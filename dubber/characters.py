"""
Who speaks each line: pitch / speaker detection and assigning a character voice. Its route.
"""
import os
import subprocess
import json
from flask import request, jsonify
from .core import (
    SPEAKER_WORKER,
    VC_DIR,
    VC_PYTHON,
    app,
    load_config,
)
from .tasks import (
    _uses_gpu,
)
from .translate import (
    _llm_chat_translate,
)
from .voices import (
    CHARACTER_ROLES,
    NATIVE_ROLE_VOICES,
    _voice_exists,
    _voice_gender,
)


def _estimate_segment_pitches(audio_path, segments, sr=16000):
    """
    Median voice pitch (Hz) of the ORIGINAL speaker for each line, or None if unclear.
    Autocorrelation on the loud, periodic frames of each line — fast, no extra models.
    """
    import numpy as np
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", audio_path, "-vn",
         "-ac", "1", "-ar", str(sr), "-af", "highpass=f=70,lowpass=f=1000", "-f", "s16le", "pipe:1"],
        capture_output=True, timeout=1800,
    )
    if r.returncode != 0 or not r.stdout:
        return [None] * len(segments)
    audio = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    frame, hop, nfft = 1024, 320, 2048
    lag_lo, lag_hi = sr // 420, sr // 70
    win = np.hanning(frame).astype(np.float32)
    out = []
    for seg in segments:
        a = int(float(seg["start"]) * sr)
        b = min(len(audio), int(float(seg["end"]) * sr))
        x = audio[a:b]
        if len(x) < frame * 2:
            out.append(None)
            continue
        idx = np.arange(0, len(x) - frame, hop)
        frames = np.stack([x[i:i + frame] for i in idx]) * win
        rms = np.sqrt((frames ** 2).mean(axis=1))
        keep = rms > max(0.01, np.percentile(rms, 50))
        frames = frames[keep]
        if len(frames) < 4:
            out.append(None)
            continue
        spec = np.fft.rfft(frames, nfft)
        acf = np.fft.irfft(np.abs(spec) ** 2, nfft)[:, :lag_hi + 2]
        acf = acf / np.maximum(acf[:, :1], 1e-9)
        f0s = []
        for row in acf:
            seg_r = row[lag_lo:lag_hi + 1]
            peak = seg_r.max()
            if peak < 0.4:
                continue  # not clearly voiced (music / noise)
            # smallest lag that is a local max close to the best peak — avoids octave errors
            for j in range(1, len(seg_r) - 1):
                if seg_r[j] >= 0.85 * peak and seg_r[j] >= seg_r[j - 1] and seg_r[j] >= seg_r[j + 1]:
                    f0s.append(sr / float(lag_lo + j))
                    break
        out.append(float(np.median(f0s)) if len(f0s) >= 4 else None)
    return out
MALE_ROLES = ("man", "boy", "old_man", "villain")
FEMALE_ROLES = ("woman", "girl", "old_woman")


@_uses_gpu("Detecting speakers")
def _detect_speakers_audio(media_path, segments, progress_cb=None):
    """Who speaks each line, from the sound: music removed (Demucs), lines grouped into
    characters by voice fingerprint, one pitch per character. GPU worker in tools/vc-env.
    Returns {"lines": [...], "speakers": [...]} or None if the engine isn't installed."""
    import tempfile
    if not (os.path.isfile(VC_PYTHON) and os.path.isfile(SPEAKER_WORKER) and media_path and os.path.isfile(media_path)):
        return None
    tmp = tempfile.mkdtemp(prefix="spk_")
    job_path, out_path = os.path.join(tmp, "job.json"), os.path.join(tmp, "out.json")
    with open(job_path, "w", encoding="utf-8") as f:
        json.dump({"audio": os.path.abspath(media_path), "out": out_path, "separate": True,
                   "segments": [[float(x["start"]), float(x["end"])] for x in segments]}, f)
    try:
        env = dict(os.environ, PYTHONIOENCODING="utf-8", TORCH_HOME=os.path.join(VC_DIR, "torch-cache"))
        proc = subprocess.Popen([VC_PYTHON, SPEAKER_WORKER, "detect", job_path], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", cwd=VC_DIR,
                                env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        tail = []
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("PROGRESS") and progress_cb:
                try:
                    _, a, b = line.split()
                    progress_cb(int(a), int(b))
                except ValueError:
                    pass
            elif line:
                tail = (tail + [line])[-12:]
                if line.startswith(("LOADED", "DONE", "WARN")):
                    print(f"[speakers] {line}")
        proc.wait()
        if proc.returncode != 0 or not os.path.isfile(out_path):
            print("[speakers failed]\n" + "\n".join(tail))
            return None
        with open(out_path, encoding="utf-8") as f:
            return json.load(f)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def _decide_character(f0, hints):
    """Role for one character. Gender comes ONLY from the voice: on a 6-character test the AI's
    guess from the dialogue text was right on 46% of lines — no better than chance — while the
    character's combined pitch was right on 84%. The text only adds flavour (old man, villain,
    grandma) when it agrees on most of the character's lines."""
    if f0 is None:
        return None
    hinted = [h for h in hints if h in CHARACTER_ROLES and h != "narrator"]
    common = max(set(hinted), key=hinted.count) if hinted else None
    sure = common is not None and len(hinted) >= 3 and hinted.count(common) >= 0.6 * len(hinted)
    if f0 < 172:
        return common if sure and common in ("old_man", "villain") else "man"
    if f0 >= 250:
        return "girl"
    return "old_woman" if sure and common == "old_woman" else "woman"


def _assign_character_voices(segments, audio_path, engine="auto", progress_state=None, native=False,
                             two_voices=False, main_voice=None, media_path=None):
    """
    Give every line a character voice.
      1. from the sound: music removed, lines grouped into characters by voice fingerprint,
         one pitch per character (falls back to per-line pitch if the GPU engine is missing)
      2. AI reads the dialogue: old / villain / child hints and the gender of unclear voices
      3. each character keeps ONE voice for all their lines
    Sets seg["voice"], seg["role"], seg["speaker"]. Returns a short summary dict.
    """
    ai_roles = {}
    try:
        numbered = [{**s, "id": i + 1} for i, s in enumerate(segments)]
        for d in _detect_speakers_task(numbered, engine=engine):
            ai_roles[d["id"]] = d.get("role")
    except Exception as e:
        print(f"[character AI hints]: {e}")

    def _cb(done, total):
        if progress_state is not None:
            progress_state["status"] = f"Finding who speaks each line {done}/{total}..."

    found = _detect_speakers_audio(media_path or audio_path, segments, progress_cb=_cb)
    if found:
        line_spk = [ln.get("speaker") for ln in found["lines"]]
        line_f0 = [ln.get("f0") for ln in found["lines"]]
        spk_f0 = {sp["id"]: sp.get("f0") for sp in found["speakers"]}
    else:
        line_f0 = _estimate_segment_pitches(audio_path, segments) if audio_path and os.path.exists(audio_path) else [None] * len(segments)
        line_spk = [None] * len(segments)
        spk_f0 = {}

    # one decision per character (lines without a character are decided alone)
    roles_by_spk = {}
    for sid, f0 in spk_f0.items():
        hints = [ai_roles.get(i + 1) for i, x in enumerate(line_spk) if x == sid and ai_roles.get(i + 1)]
        roles_by_spk[sid] = _decide_character(f0, hints)

    cfg = load_config()
    role_map = cfg.get("role_voice_map", {})
    male_pool = ["native_male_1", "native_male_2", "native_male_3"] if native else ["hero_male", "narrator_male", "villain_male", "elder_male"]
    girl_pool = ["native_girl_1", "native_girl_2"] if native else ["heroine_female", "child"]
    woman_pool = ["native_female_1", "native_female_3", "native_female_2"] if native else ["narrator_female", "soft_female", "elder_female"]
    if main_voice and _voice_gender(main_voice) == "male" and main_voice in male_pool:
        male_pool.remove(main_voice)
        male_pool.insert(0, main_voice)
    for pool in (girl_pool, woman_pool):
        if main_voice in pool:
            pool.remove(main_voice)
            pool.insert(0, main_voice)

    def _voice_for(role, sid):
        want = "male" if role in MALE_ROLES else "female"
        if two_voices:
            if main_voice and _voice_gender(main_voice) == want:
                return main_voice
            if native:
                return "native_male_1" if want == "male" else "native_girl_1"
            return "hero_male" if want == "male" else "heroine_female"
        mapped = role_map.get(role)
        if _voice_exists(mapped):
            return mapped
        if sid is None:
            return (NATIVE_ROLE_VOICES.get(role) if native else None) or CHARACTER_ROLES[role]["voice"]
        # a different, consistent voice per character: main characters get the first voices
        if want == "male":
            pool = male_pool
            if role == "old_man":
                pool = male_pool[1:] + male_pool[:1]
        else:
            pool = girl_pool if role == "girl" else woman_pool
        rank = [x for x in order if (roles_by_spk.get(x) in MALE_ROLES) == (want == "male")
                and ((roles_by_spk.get(x) == "girl") == (role == "girl") or want == "male")]
        k = rank.index(sid) if sid in rank else 0
        return pool[k % len(pool)]

    order = [sp["id"] for sp in found["speakers"]] if found else []   # most lines first
    counts, chars = {}, {}
    for i, s in enumerate(segments):
        sid = line_spk[i] if i < len(line_spk) else None
        role = roles_by_spk.get(sid) if sid else None
        if role is None:
            f0 = line_f0[i] if i < len(line_f0) else None
            hint = ai_roles.get(i + 1)
            role = _decide_character(f0, [])
        if not role:
            continue
        s["role"] = role
        s["speaker"] = sid or ""
        s["voice"] = _voice_for(role, sid)
        counts[role] = counts.get(role, 0) + 1
        if sid:
            chars[sid] = role
    return {"engine": "voice-groups" if found else "per-line pitch", "characters": len(chars),
            "pitch_found": sum(1 for f in line_f0 if f), "roles": counts}


def _detect_speakers_task(subtitles, engine="auto"):
    """
    Analyze dialogue flow, character names, pronouns, context, and tone
    to classify every subtitle cue into Girl, Boy, Man, Woman, Old Man, Old Woman, Villain, or Narrator.
    """
    if not subtitles:
        return []

    sys_prompt = """You are an expert movie voice casting director and speaker diarization AI.
Analyze this sequence of movie/video dialogue lines with timestamps.
For each numbered line, classify who is speaking into exactly one role:
- girl: Young girl, teenage girl, daughter (sweet, youthful feminine tone)
- boy: Young boy, teenage boy, son (energetic, youthful masculine tone)
- man: Adult man, hero, warrior, father, soldier (standard adult male)
- woman: Adult woman, heroine, mother, wife (standard adult female)
- old_man: Elderly grandfather, master, monk, old man (deep, raspy elder male)
- old_woman: Elderly grandmother, mature elder lady (gentle, mature elder female)
- villain: Boss, villain, enemy, monster (threatening, aggressive)
- narrator: Background narrator describing events or recap host

Analyze speaker turns, names, honorifics (e.g., Grandpa, Grandma, Dad, Sir, Girl, Boy, King, Master), questions/answers, and context.
Output ONLY a strict JSON array of objects with id and role. No Markdown outside the array, no commentary.
Format:
[
  {"id": 1, "role": "girl"},
  {"id": 2, "role": "old_man"}
]"""

    batch_size = 150
    results_map = {}

    for i in range(0, len(subtitles), batch_size):
        chunk = subtitles[i:i + batch_size]
        lines = []
        for k, s in enumerate(chunk):
            s_id = s.get("id", i + k + 1)
            txt = s.get("source") or s.get("text") or ""
            lines.append(f"{s_id}: {txt.strip()}")
        user_prompt = "\n".join(lines)

        raw_resp = None
        try:
            raw_resp, _ = _llm_chat_translate([], system_prompt=sys_prompt, user_prompt=user_prompt, engine_choice="groq")
            if not raw_resp:
                raw_resp, _ = _llm_chat_translate([], system_prompt=sys_prompt, user_prompt=user_prompt, engine_choice=engine)
        except Exception as e:
            print(f"[detect_speakers error chunk {i}]: {e}")

        if raw_resp:
            clean = raw_resp.strip()
            if "```json" in clean:
                clean = clean.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in clean:
                clean = clean.split("```", 1)[1].split("```", 1)[0].strip()

            try:
                parsed = json.loads(clean)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "id" in item:
                            results_map[item["id"]] = item.get("role", "narrator")
            except Exception as e:
                print(f"[detect_speakers json parse error]: {e} on response: {clean[:100]}")

    detected = []
    for idx, s in enumerate(subtitles, 1):
        s_id = s.get("id", idx)
        role = results_map.get(s_id)
        if not role or role not in CHARACTER_ROLES:
            txt = (s.get("source") or s.get("text") or "").lower()
            if any(k in txt for k in ["grandpa", "grandfather", "old man", "elder", "master", "លោកតា", "爺", "老者", "师父"]):
                role = "old_man"
            elif any(k in txt for k in ["grandma", "grandmother", "old lady", "elder woman", "លោកយាយ", "奶", "老妇"]):
                role = "old_woman"
            elif any(k in txt for k in ["girl", "sister", "daughter", "princess", "miss", "lady", "ក្មេងស្រី", "ក្រមុំ", "女", "妹"]):
                role = "girl"
            elif any(k in txt for k in ["boy", "brother", "son", "kid", "child", "prince", "ក្មេងប្រុស", "កំលោះ", "男", "哥", "儿"]):
                role = "boy"
            elif any(k in txt for k in ["die!", "fool", "kill", "destroy", "haha", "monster", "beast", "តួអាក្រក់"]):
                role = "villain"
            else:
                role = "narrator"

        role_info = CHARACTER_ROLES[role]
        cfg = load_config()
        role_map = cfg.get("role_voice_map", {})
        assigned_voice = role_map.get(role) or role_info["voice"]
        detected.append({
            "id": s_id,
            "role": role,
            "role_label": role_info["label"],
            "voice": assigned_voice,
            "badge": role_info["badge"]
        })

    return detected


@app.route('/auto-detect-voices', methods=['POST'])
def auto_detect_voices():
    """Auto-detect speaker character roles (Girl, Boy, Man, Woman, Old Man, Old Woman, etc.)"""
    data = request.json or {}
    subtitles = data.get('subtitles', [])
    engine = data.get('engine', 'auto')
    if not subtitles:
        return jsonify({"success": False, "error": "No subtitles provided"}), 400

    try:
        detected = _detect_speakers_task(subtitles, engine=engine)
        return jsonify({
            "success": True,
            "characters": detected,
            "count": len(detected)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
