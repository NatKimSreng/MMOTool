"""
Khmer voices: voice profiles, moods, native voice conversion (Seed-VC), cloned voices, ElevenLabs,
Edge-TTS synthesis and building the timed speech track. Voice routes.
"""
import os
import subprocess
import threading
import json
import asyncio
import edge_tts
from flask import request, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
import yt_dlp
from .core import (
    _strip_foreign_script,
    CLONED_SAMPLES_DIR,
    CLONED_VOICES_FILE,
    DEFAULT_OUTPUT_FOLDER,
    UPLOAD_FOLDER,
    VC_DIR,
    VC_PYTHON,
    VC_WORKER,
    VOICES_DIR,
    app,
    load_config,
    save_config,
)
from .tasks import (
    _uses_gpu,
    task_warn,
)
from .media import (
    _get_media_duration,
)
from .translate import (
    _llm_chat_translate,
)


# Professional Khmer Neural Voices & Character Cast Profiles
KHMER_VOICE_PROFILES = {
    "narrator_female": {
        "name": "🎙️ និទានស្រី (Female Narrator)",
        "voice": "km-KH-SreymomNeural",
        "rate": "-5%",
        "pitch": "+0Hz",
        "desc": "Warm, engaging, storytelling tone for recaps"
    },
    "narrator_male": {
        "name": "🎙️ និទានប្រុស (Male Narrator)",
        "voice": "km-KH-PisethNeural",
        "rate": "-5%",
        "pitch": "-4Hz",
        "desc": "Clear, steady, authoritative tone for recaps"
    },
    "hero_male": {
        "name": "👦 តួឯកប្រុស / កំលោះ (Hero / Young Male)",
        "voice": "km-KH-PisethNeural",
        "rate": "+0%",
        "pitch": "+5Hz",
        "desc": "Energetic, clear youth tone"
    },
    "heroine_female": {
        "name": "👧 តួឯកស្រី / ក្រមុំ (Heroine / Young Female)",
        "voice": "km-KH-SreymomNeural",
        "rate": "+0%",
        "pitch": "+7Hz",
        "desc": "Bright, sweet, youthful tone"
    },
    "elder_male": {
        "name": "👴 មនុស្សចាស់ / តា (Elder / Old Male)",
        "voice": "km-KH-PisethNeural",
        "rate": "-12%",
        "pitch": "-16Hz",
        "desc": "Slower, deep elder tone"
    },
    "elder_female": {
        "name": "👵 យាយ / ចាស់ទុំ (Elder Female)",
        "voice": "km-KH-SreymomNeural",
        "rate": "-12%",
        "pitch": "-8Hz",
        "desc": "Slower, mature elder female tone"
    },
    "villain_male": {
        "name": "🦹 តួអាក្រក់ / ម៉ឺងម៉ាត់ (Villain / Boss)",
        "voice": "km-KH-PisethNeural",
        "rate": "-8%",
        "pitch": "-22Hz",
        "desc": "Deep, commanding resonant tone"
    },
    "child": {
        "name": "👶 ក្មេងតូច (Child / Kid)",
        "voice": "km-KH-SreymomNeural",
        "rate": "+8%",
        "pitch": "+20Hz",
        "desc": "High pitch, youthful and playful"
    },
    "soft_female": {
        "name": "🌸 ស្រីស្រទន់ / កម្សត់ (Soft / Emotional)",
        "voice": "km-KH-SreymomNeural",
        "rate": "-8%",
        "pitch": "-2Hz",
        "desc": "Gentle, emotional drama tone"
    },
    "recap_fast": {
        "name": "⚡ សម្រាយរហ័ស (Fast Recap Narrator)",
        "voice": "km-KH-PisethNeural",
        "rate": "+15%",
        "pitch": "-2Hz",
        "desc": "High-tempo fast-paced movie recap"
    },
    "female": {
        "name": "Sreymom (Standard Female)",
        "voice": "km-KH-SreymomNeural",
        "rate": "-8%",
        "pitch": "+0Hz",
        "desc": "Standard female voice"
    },
    "male": {
        "name": "Piseth (Standard Male)",
        "voice": "km-KH-PisethNeural",
        "rate": "-8%",
        "pitch": "-5Hz",
        "desc": "Standard male voice"
    },
}

KHMER_VOICES = {k: v["voice"] for k, v in KHMER_VOICE_PROFILES.items()}

NATIVE_VOICES = {
    "native_male_1":   {"name": "Dara · ប្រុស",   "gender": "male",   "file": "native_male_1.wav",   "base": "narrator_male"},
    "native_male_2":   {"name": "Sokha · ប្រុស",  "gender": "male",   "file": "native_male_2.wav",   "base": "narrator_male"},
    "native_male_3":   {"name": "Vibol · ប្រុស",  "gender": "male",   "file": "native_male_3.wav",   "base": "narrator_male"},
    "native_female_1": {"name": "Sreyneang · ស្រី", "gender": "female", "file": "native_female_1.wav", "base": "narrator_female"},
    "native_female_2": {"name": "Channy · ស្រី",  "gender": "female", "file": "native_female_2.wav", "base": "narrator_female"},
    "native_female_3": {"name": "Bopha · ស្រី",   "gender": "female", "file": "native_female_3.wav", "base": "narrator_female"},
    # young, bright, high voices (like popular Khmer drama dubbing girls)
    "native_girl_1":   {"name": "Pich · ក្មេងស្រី",  "gender": "female", "file": "native_girl_1.wav",   "base": "narrator_female", "expressive": 1.5},
    "native_girl_2":   {"name": "Nita · ក្មេងស្រី",  "gender": "female", "file": "native_girl_2.wav",   "base": "narrator_female", "expressive": 1.5},
}
for _k, _v in NATIVE_VOICES.items():
    _v.setdefault("expressive", 1.2 if _v["gender"] == "male" else 1.3)

# When "different voice per character" is on and a native voice is chosen,
# characters get native voices too (by gender), each a little different.
NATIVE_ROLE_VOICES = {
    "man": "native_male_1", "boy": "native_male_3", "old_man": "native_male_2", "villain": "native_male_2",
    "woman": "native_female_1", "girl": "native_girl_1", "old_woman": "native_female_2",
    "child": "native_girl_2", "narrator": "native_male_1",
}


# How each mood changes the delivery: (speed %, pitch Hz, volume %)
EMOTION_PROSODY = {
    "neutral": (0, 0, 0),
    "excited": (10, 8, 12),
    "happy":   (6, 6, 6),
    "angry":   (8, 2, 18),
    "tense":   (3, 3, 6),
    "scared":  (7, 10, -4),
    "sad":     (-10, -6, -8),
    "calm":    (-6, -3, -10),
}


def _guess_emotion(text):
    t = (text or "").strip()
    if t.endswith("!") or "!" in t[-3:]:
        return "excited"
    return "neutral"


def _detect_line_emotions(segments):
    """AI reads the script and tags each line with a mood (Groq first, so the Gemini quota is
    kept for translation). Falls back to punctuation. Returns a list of mood names."""
    sys_prompt = (
        "You are a voice director for Khmer movie dubbing. For each numbered line, choose the mood the "
        "voice actor should use, from exactly: neutral, excited, happy, angry, tense, scared, sad, calm.\n"
        "Story narration that builds suspense is 'tense'; big reveals and action are 'excited'.\n"
        'Reply ONLY with a JSON array like [{"id": 1, "e": "excited"}, {"id": 2, "e": "sad"}].'
    )
    moods = [_guess_emotion(s.get("text", "")) for s in segments]
    batch = 150
    for i in range(0, len(segments), batch):
        chunk = segments[i: i + batch]
        user_prompt = "\n".join(f"{i + k + 1}: {(s.get('source') or s.get('text') or '').strip()}" for k, s in enumerate(chunk))
        raw = None
        try:
            raw, _ = _llm_chat_translate([], system_prompt=sys_prompt, user_prompt=user_prompt, engine_choice="groq")
            if not raw:
                raw, _ = _llm_chat_translate([], system_prompt=sys_prompt, user_prompt=user_prompt, engine_choice="auto")
        except Exception as e:
            print(f"[emotion detect]: {e}")
        if not raw:
            continue
        clean = raw.strip()
        if "```" in clean:
            clean = clean.split("```", 2)[1]
            clean = clean[4:] if clean.startswith("json") else clean
        a, b = clean.find("["), clean.rfind("]")
        try:
            for item in json.loads(clean[a: b + 1]):
                idx = int(item.get("id", 0)) - 1
                mood = str(item.get("e") or item.get("emotion") or "").lower().strip()
                if 0 <= idx < len(moods) and mood in EMOTION_PROSODY:
                    moods[idx] = mood
        except Exception as e:
            print(f"[emotion detect parse]: {e}")
    return moods


def _vc_available():
    """True when the local voice-conversion engine and its models are installed."""
    m = os.path.join(VC_DIR, "models")
    need = [
        VC_PYTHON, VC_WORKER,
        os.path.join(m, "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"),
        os.path.join(m, "bigvgan", "bigvgan_generator.pt"),
        os.path.join(m, "campplus_cn_common.bin"),
    ]
    whisper_ok = any(os.path.isfile(os.path.join(m, "whisper-small", f)) for f in ("pytorch_model.bin", "model.safetensors"))
    return whisper_ok and all(os.path.isfile(x) for x in need)


def _vc_reference_for_voice(voice_key):
    """Reference recording for a voice that should be converted, else None."""
    if not voice_key:
        return None
    if voice_key in NATIVE_VOICES:
        path = os.path.join(VOICES_DIR, NATIVE_VOICES[voice_key]["file"])
        return path if os.path.isfile(path) else None
    cv = get_cloned_voice_by_id(voice_key)
    if cv and cv.get("sample_file") and not cv.get("elevenlabs_id"):
        path = os.path.join(CLONED_SAMPLES_DIR, cv["sample_file"])
        return path if os.path.isfile(path) else None
    return None


def _voice_expressiveness(voice_key):
    """How much to widen the pitch melody for this voice (1.0 = as spoken by the TTS)."""
    cfg = load_config()
    if not cfg.get("lively_voice", True):
        return 1.0
    if "vc_expressive" in cfg:
        return float(cfg["vc_expressive"])
    if voice_key in NATIVE_VOICES:
        return float(NATIVE_VOICES[voice_key].get("expressive", 1.2))
    cv = get_cloned_voice_by_id(voice_key)
    if cv and cv.get("role") in ("girl", "child"):
        return 1.5
    return 1.2


def _voice_gender(voice_key):
    if voice_key in NATIVE_VOICES:
        return NATIVE_VOICES[voice_key]["gender"]
    cv = get_cloned_voice_by_id(voice_key)
    if cv:
        return "male" if cv.get("role") in ("man", "boy", "old_man", "villain") else "female"
    prof = KHMER_VOICE_PROFILES.get(voice_key)
    if prof:
        return "male" if "Piseth" in prof["voice"] else "female"
    return None


def _voice_exists(voice_key):
    return bool(voice_key) and (
        voice_key in KHMER_VOICE_PROFILES or voice_key in NATIVE_VOICES or get_cloned_voice_by_id(voice_key) is not None
    )


@_uses_gpu("Converting voices")
def _run_voice_conversion(items, progress_cb=None):
    """
    Convert TTS lines to reference voices. items = [{"src", "dst", "reference"}, ...].
    Runs the worker once (models load once). Returns the set of dst files created.
    """
    import tempfile
    if not items or not _vc_available():
        return set()
    cfg = load_config()
    job = {
        "items": [{"src": it["src"], "dst": it["dst"], "reference": it["reference"],
                   "f0_scale": it.get("f0_scale", 1.2)} for it in items],
        "steps": int(cfg.get("vc_steps", 12)),
        "cfg_rate": float(cfg.get("vc_cfg_rate", 0.7)),
    }
    fd, job_path = tempfile.mkstemp(suffix=".json", prefix="vc_job_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False)
    try:
        env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1", TQDM_DISABLE="1",
                   HF_HOME=os.path.join(VC_DIR, "hf-cache"))
        proc = subprocess.Popen(
            [VC_PYTHON, VC_WORKER, job_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            cwd=VC_DIR, env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
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
                tail.append(line)
                tail = tail[-15:]
                if line.startswith(("LOADED", "DONE", "ERROR")):
                    print(f"[voice conversion] {line}")
        proc.wait()
        if proc.returncode != 0:
            print("[voice conversion failed]\n" + "\n".join(tail))
    finally:
        try:
            os.remove(job_path)
        except OSError:
            pass
    return {it["dst"] for it in items if os.path.isfile(it["dst"]) and os.path.getsize(it["dst"]) > 1000}


def _convert_segments_to_voices(segments, segment_files, default_voice, progress_cb=None):
    """Swap segment files for native-voice versions where the line's voice needs conversion.
    Returns (converted_count, wanted_count)."""
    items = []
    for i, seg in enumerate(segments):
        if i >= len(segment_files) or not os.path.isfile(segment_files[i]):
            continue
        vkey = seg.get("voice") or default_voice
        ref = _vc_reference_for_voice(vkey)
        if ref:
            items.append({"src": segment_files[i], "dst": os.path.splitext(segment_files[i])[0] + "_native.wav",
                          "reference": ref, "idx": i, "f0_scale": _voice_expressiveness(vkey)})
    if not items:
        return 0, 0
    if not _vc_available():
        print("[voice conversion] engine not installed — using standard voices")
        return 0, len(items)
    done = _run_voice_conversion(items, progress_cb=progress_cb)
    for it in items:
        if it["dst"] in done:
            segment_files[it["idx"]] = it["dst"]
    return len(done), len(items)


def _native_preview_path(voice_key):
    return os.path.join(VOICES_DIR, "previews", f"{voice_key}.mp3")


def _make_voice_previews(voice_keys, text=None):
    """Pre-render short samples for voices that need conversion (shown on the ▶ buttons)."""
    import tempfile
    text = text or "សួស្តីបងប្អូនទាំងអស់គ្នា! ថ្ងៃនេះខ្ញុំនឹងនិទានរឿងមួយដ៏គួរឱ្យរំភើប ដែលអ្នកមិនធ្លាប់ឮពីមុនមក!"
    os.makedirs(os.path.join(VOICES_DIR, "previews"), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="vprev_")
    items = []
    for k in voice_keys:
        ref = _vc_reference_for_voice(k)
        if not ref:
            continue
        src = os.path.join(tmp, f"{k}.mp3")
        asyncio.run(_synthesize_text_to_file(text, k, src, custom_rate="+3%", emotion="excited"))
        if os.path.isfile(src):
            items.append({"src": src, "dst": os.path.join(tmp, f"{k}.wav"), "reference": ref, "key": k,
                          "f0_scale": _voice_expressiveness(k)})
    done = _run_voice_conversion(items)
    for it in items:
        if it["dst"] in done:
            subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", it["dst"],
                            "-af", "loudnorm=I=-16:TP=-1.5", "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "128k",
                            _native_preview_path(it["key"])], capture_output=True, timeout=120)
    return [it["key"] for it in items if os.path.isfile(_native_preview_path(it["key"]))]
KHMER_VOICE_PITCH = {k: v["pitch"] for k, v in KHMER_VOICE_PROFILES.items()}


def get_voice_settings(voice_key, custom_rate=None, custom_pitch=None):
    """Resolve neural voice ID, rate, and pitch from profile key with custom overrides."""
    profile = KHMER_VOICE_PROFILES.get(voice_key)
    if profile:
        voice = profile["voice"]
        rate = custom_rate if (custom_rate and custom_rate != "default") else profile["rate"]
        pitch = custom_pitch if (custom_pitch and custom_pitch != "default") else profile["pitch"]
        return voice, rate, pitch
    voice = KHMER_VOICES.get(voice_key, "km-KH-SreymomNeural")
    pitch = custom_pitch if (custom_pitch and custom_pitch != "default") else "+0Hz"
    rate = custom_rate if (custom_rate and custom_rate != "default") else "-8%"
    return voice, rate, pitch


# ═══════════════════════════════════════════════════════════
# VOICE CLONING (INTERNET AUDIO / YOUTUBE / ELEVENLABS / MP3)
# ═══════════════════════════════════════════════════════════

def load_cloned_voices():
    """Load user's cloned voices from cloned_voices.json."""
    if os.path.exists(CLONED_VOICES_FILE):
        try:
            with open(CLONED_VOICES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            print(f"[cloned_voices load failed]: {e}")
    return []


def save_cloned_voices(voices_list):
    """Save user's cloned voices to cloned_voices.json."""
    try:
        with open(CLONED_VOICES_FILE, "w", encoding="utf-8") as f:
            json.dump(voices_list, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[cloned_voices save failed]: {e}")
        return False


def get_cloned_voice_by_id(voice_id):
    """Retrieve cloned voice dict by ID."""
    if not voice_id:
        return None
    for v in load_cloned_voices():
        if v.get("id") == voice_id:
            return v
    return None


def _elevenlabs_clone_voice(name, audio_path, api_key, description="Cloned via DAI Dubber Pro"):
    """Clone a voice directly into user's ElevenLabs account using reference audio."""
    import requests
    url = "https://api.elevenlabs.io/v1/voices/add"
    headers = {"xi-api-key": api_key}
    try:
        with open(audio_path, "rb") as f:
            files = {"files": (os.path.basename(audio_path), f, "audio/mpeg")}
            data = {"name": name, "description": description}
            r = requests.post(url, headers=headers, data=data, files=files, timeout=60)
        if r.status_code in (200, 201):
            res = r.json()
            return True, res.get("voice_id", "")
        else:
            try:
                err = r.json().get("detail", {}).get("message", r.text)
            except Exception:
                err = r.text
            return False, err
    except Exception as e:
        return False, str(e)


def _elevenlabs_tts(text, voice_id, api_key, out_path):
    """Generate high-fidelity human speech via ElevenLabs Multilingual V2."""
    text = _strip_foreign_script(text)
    import requests
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.50,
            "similarity_boost": 0.85,
            "style": 0.35,
            "use_speaker_boost": True
        }
    }
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=45)
        if r.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(r.content)
            return True, ""
        else:
            try:
                err = r.json().get("detail", {}).get("message", r.text)
            except Exception:
                err = r.text
            return False, err
    except Exception as e:
        return False, str(e)


def extract_voice_sample_from_url(url, start_sec=0, duration_sec=40):
    """Download audio from YouTube / internet link using yt-dlp and extract clean sample via FFmpeg."""
    import tempfile
    import uuid
    import subprocess
    sample_id = f"cloned_{uuid.uuid4().hex[:8]}"
    out_sample_mp3 = os.path.join(CLONED_SAMPLES_DIR, f"{sample_id}.mp3")

    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_tmpl = os.path.join(tmp_dir, "dl_audio.%(ext)s")
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': raw_tmpl,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        candidates = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.startswith("dl_audio")]
        if not candidates:
            raise RuntimeError("Failed to extract audio from the internet URL")
        dl_file = candidates[0]

        st = max(0.0, float(start_sec))
        dur = min(120.0, max(10.0, float(duration_sec)))
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(st),
            "-t", str(dur),
            "-i", dl_file,
            "-ac", "1",
            "-ar", "44100",
            "-b:a", "192k",
            out_sample_mp3
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        if r.returncode != 0 or not os.path.exists(out_sample_mp3):
            raise RuntimeError("FFmpeg audio sample extraction failed")

    return sample_id, f"{sample_id}.mp3", out_sample_mp3


def extract_voice_sample_from_file(uploaded_file, start_sec=0, duration_sec=40):
    """Extract clean speech sample from an uploaded video or audio file."""
    import tempfile
    import uuid
    import subprocess
    sample_id = f"cloned_{uuid.uuid4().hex[:8]}"
    out_sample_mp3 = os.path.join(CLONED_SAMPLES_DIR, f"{sample_id}.mp3")

    with tempfile.TemporaryDirectory() as tmp_dir:
        orig_name = secure_filename(uploaded_file.filename or "audio.mp3")
        in_path = os.path.join(tmp_dir, orig_name)
        uploaded_file.save(in_path)

        st = max(0.0, float(start_sec))
        dur = min(120.0, max(10.0, float(duration_sec)))
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(st),
            "-t", str(dur),
            "-i", in_path,
            "-ac", "1",
            "-ar", "44100",
            "-b:a", "192k",
            out_sample_mp3
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        if r.returncode != 0 or not os.path.exists(out_sample_mp3):
            raise RuntimeError("FFmpeg audio extraction from uploaded file failed")

    return sample_id, f"{sample_id}.mp3", out_sample_mp3


def _master_speech_audio(input_path, output_path=None, soft_highs=False):
    """
    Light voice polish for TTS speech. Edge-TTS is already compressed and bright,
    so we only clean rumble, add a touch of presence and catch peaks — heavy EQ /
    compression on top made the voice harsh and pumpy.
    Works on .wav (lossless, preferred) or .mp3.
    """
    import subprocess
    if not input_path or not os.path.exists(input_path):
        return input_path

    ext = os.path.splitext(input_path)[1].lower() or ".wav"
    in_place = False
    if not output_path:
        output_path = input_path + ".master" + ext
        in_place = True

    af = (
        "highpass=f=70,"
        "equalizer=f=200:t=q:w=1.0:g=1.0,"
        "equalizer=f=3200:t=q:w=1.2:g=1.2,"
        "acompressor=threshold=0.125:ratio=2:attack=10:release=150:makeup=1,"
        "alimiter=limit=0.89:attack=5:release=60"
    )
    if soft_highs:
        # studio-crisp highs give AI voices away; real dubbing over a scene is a bit softer
        af = af.replace("highpass=f=70,", "highpass=f=70,equalizer=f=9500:t=h:w=4000:g=-3.5,")
    codec = ["-c:a", "pcm_s16le"] if ext == ".wav" else ["-c:a", "libmp3lame", "-b:a", "192k"]
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", input_path, "-af", af, *codec, output_path]
    try:
        dur = _get_media_duration(input_path) or 60
        r = subprocess.run(cmd, capture_output=True, timeout=max(180, int(dur / 5)))
        if r.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            if in_place:
                os.replace(output_path, input_path)
                return input_path
            return output_path
        if os.path.exists(output_path):
            os.remove(output_path)
        print(f"[mastering failed]: {(r.stderr or b'').decode('utf-8', errors='ignore')[-300:]}")
    except Exception as e:
        print(f"[mastering failed]: {e}")
    return input_path


async def _synthesize_text_to_file(text, voice_key, out_mp3_path, custom_rate=None, custom_pitch=None, slot_duration=None, speed_boost_pct=0, emotion=None):
    """
    Centralized speech synthesis:
    1. Supports ElevenLabs cloned voices (via API).
    2. Resolves character role mappings from role_voice_map.
    3. Falls back smoothly to Edge-TTS neural character profiles.
    """
    text = _strip_foreign_script(text)
    cfg = load_config()
    role_map = cfg.get("role_voice_map", {})

    # Check if voice_key is a role name (e.g., 'girl', 'boy', 'old_man')
    if voice_key in CHARACTER_ROLES:
        voice_key = role_map.get(voice_key) or CHARACTER_ROLES[voice_key]["voice"]

    # Native voices: Khmer words from the matching Edge voice; the pipeline converts the sound afterwards
    if voice_key in NATIVE_VOICES:
        voice_key = NATIVE_VOICES[voice_key]["base"]

    # Check if voice_key is a user's cloned voice
    cloned_voice = get_cloned_voice_by_id(voice_key)
    el_key = (cfg.get("ELEVENLABS_API_KEY") or "").strip()

    if cloned_voice and cloned_voice.get("elevenlabs_id") and el_key:
        ok, err = _elevenlabs_tts(text, cloned_voice["elevenlabs_id"], el_key, out_mp3_path)
        if ok and os.path.exists(out_mp3_path) and os.path.getsize(out_mp3_path) > 100:
            return True
        task_warn(f"ElevenLabs voice failed: {err} — used the free neural voice")

    # If cloned voice had no ElevenLabs key, resolve to its assigned character role
    if cloned_voice:
        base_role = cloned_voice.get("role", "narrator")
        if base_role in ("man", "boy", "old_man", "villain", "narrator_male"):
            voice_key = "narrator_male"
        elif base_role in ("woman", "girl", "old_woman", "child", "narrator_female"):
            voice_key = "narrator_female"
        else:
            voice_key = CHARACTER_ROLES.get(base_role, {}).get("voice", "narrator_female")

    v_id, v_rate_base, v_pitch = get_voice_settings(voice_key, custom_rate=custom_rate, custom_pitch=custom_pitch)
    seg_rate = _estimate_tts_rate(text, slot_duration, base_rate=v_rate_base, natural=True) if slot_duration else v_rate_base
    if speed_boost_pct:
        try:
            base_pct = int(str(seg_rate).replace("%", "").replace("+", "") or "0")
        except ValueError:
            base_pct = 0
        seg_rate = f"{max(-50, min(100, base_pct + int(speed_boost_pct))):+d}%"

    # mood of the line: a little faster/higher/louder for excitement, slower/lower for sadness…
    seg_volume = "+0%"
    d_rate, d_pitch, d_vol = EMOTION_PROSODY.get(emotion or "", (0, 0, 0))
    if (text or "").strip().endswith("?"):
        d_pitch += 4
    if d_rate or d_pitch or d_vol:
        def _num(v, unit):
            try:
                return int(str(v).replace(unit, "").replace("+", "") or "0")
            except ValueError:
                return 0
        seg_rate = f"{max(-50, min(100, _num(seg_rate, '%') + d_rate)):+d}%"
        v_pitch = f"{_num(v_pitch, 'Hz') + d_pitch:+d}Hz"
        seg_volume = f"{max(-50, min(50, d_vol)):+d}%"

    try:
        await edge_tts.Communicate(text, v_id, rate=seg_rate, pitch=v_pitch, volume=seg_volume).save(out_mp3_path)
        return True
    except Exception as e:
        print(f"[edge_tts failed for {v_id}]: {e}")
        try:
            await edge_tts.Communicate(text, "km-KH-SreymomNeural", rate=seg_rate, pitch="+0Hz").save(out_mp3_path)
            return True
        except Exception as e2:
            task_warn(f"Voice generation failed: {e2}")
            return False

@app.route('/generate-segment-voice', methods=['POST'])
def generate_segment_voice():
    data = request.json
    text = data.get('text')
    segment_id = data.get('id')
    output_dir = data.get('outputDir', DEFAULT_OUTPUT_FOLDER)
    voice_key = data.get('voice', 'female')
    rate = data.get('rate', '-8%')

    if not text:
        return jsonify({"success": False, "error": "Text is required"}), 400

    try:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"segment_{segment_id}.mp3")
        voice = KHMER_VOICES.get(voice_key, KHMER_VOICES["female"])
        pitch = KHMER_VOICE_PITCH.get(voice_key, "+0Hz")

        async def speak():
            communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            await communicate.save(output_file)

        asyncio.run(speak())

        return jsonify({
            "success": True,
            "path": output_file,
            "filename": f"segment_{segment_id}.mp3"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def _estimate_tts_rate(text, slot_sec, base_rate="+0%", natural=True):
    """
    Pick Edge TTS rate so spoken length is closer to the video slot.
    Khmer ~ 8–11 chars/sec at normal rate (rough).
    When natural=True (default): keep rate close to base so voice stays
    easy to listen to — never jump more than ~±15% from user choice.
    """
    text = (text or "").strip()
    try:
        base_pct = int(str(base_rate).replace("%", "").replace("+", "") or "0")
    except ValueError:
        base_pct = 0

    if not text or slot_sec <= 0.2:
        return f"{base_pct:+d}%"

    chars = len(text.replace(" ", ""))
    # natural duration estimate (slightly conservative for clarity)
    natural_sec = max(0.45, chars / 8.5)
    need = natural_sec / slot_sec  # >1 means need faster

    if natural:
        # Only nudge a little toward the slot — prefer consistent pace
        # Keep voice sounding natural; avoid big speed jumps that feel "not matching"
        pct = int((need - 1.0) * 40)  # even softer mapping
        pct = max(-12, min(12, pct))
        pct = max(-22, min(22, pct + base_pct))
        return f"{pct:+d}%"

    # Strict mode: try harder to fit (old aggressive behavior, still capped)
    pct = int((need - 1.0) * 100)
    pct = max(-30, min(50, pct))
    pct = max(-40, min(60, pct + base_pct // 2))
    return f"{pct:+d}%"


def _fit_audio_to_ms(audio, target_ms, max_speedup=1.12, min_speedup=0.88):
    """
    Gently adjust audio length. Prefer natural pace over exact fit.
    - Only stretch within [min_speedup, max_speedup] (default ±12%).
    - Outside that range: pad with silence if short, soft-trim if long
      (caller should give more room via gaps when possible).
    """
    from pydub import AudioSegment
    import subprocess
    import tempfile

    if target_ms <= 0:
        return AudioSegment.silent(duration=0)
    cur = len(audio)
    if cur <= 0:
        return AudioSegment.silent(duration=target_ms)

    # Close enough — tiny pad/trim only
    if abs(cur - target_ms) <= 80:
        if cur > target_ms:
            return audio[:target_ms]
        return audio + AudioSegment.silent(duration=target_ms - cur)

    ratio = cur / float(target_ms)  # >1 = audio longer than slot → need speedup

    # Elastic auto-fit speech pacing
    if ratio > max_speedup:
        # Allow natural tempo speedup up to 1.30x so words stay intact
        ratio = min(ratio, 1.30)
    elif ratio < min_speedup:
        ratio = min_speedup

    # ffmpeg atempo (high quality pitch-preserving time stretch)
    if 0.5 <= ratio <= 2.0 and abs(ratio - 1.0) > 0.03:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
                in_path = f_in.name
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
                out_path = f_out.name
            audio.export(in_path, format="wav")
            atempo = max(0.5, min(2.0, ratio))
            cmd = [
                "ffmpeg", "-y", "-i", in_path,
                "-filter:a", f"atempo={atempo:.4f}",
                out_path,
            ]
            r = subprocess.run(cmd, capture_output=True, timeout=120)
            if r.returncode == 0 and os.path.exists(out_path):
                sped = AudioSegment.from_file(out_path)
                try:
                    os.remove(in_path)
                    os.remove(out_path)
                except OSError:
                    pass
                if len(sped) > target_ms:
                    over = len(sped) - target_ms
                    # If slightly longer, let it breathe into the gap so last words are never cut off
                    if over <= 350:
                        return sped
                    else:
                        sped = sped[:target_ms + 200].fade_out(min(45, (target_ms + 200) // 4))
                elif len(sped) < target_ms:
                    sped = sped + AudioSegment.silent(duration=target_ms - len(sped))
                return sped
            try:
                os.remove(in_path)
                os.remove(out_path)
            except OSError:
                pass
        except Exception as e:
            print(f"[atempo]: {e}")

    # pydub frame-rate fallback (mild only)
    if 0.85 <= ratio <= 1.18:
        new_rate = int(audio.frame_rate * ratio)
        sped = audio._spawn(audio.raw_data, overrides={"frame_rate": new_rate})
        sped = sped.set_frame_rate(audio.frame_rate)
        if len(sped) > target_ms:
            over = len(sped) - target_ms
            if over <= 300:
                return sped
            sped = sped[:target_ms + 150].fade_out(35)
        elif len(sped) < target_ms:
            sped = sped + AudioSegment.silent(duration=target_ms - len(sped))
        return sped

    # No aggressive stretch — soft blend into following gap
    if cur > target_ms:
        over = cur - target_ms
        if over <= 300:
            return audio
        out = audio[:target_ms + 150]
        if len(out) > 50:
            out = out.fade_out(min(35, len(out) // 5))
        return out
    return audio + AudioSegment.silent(duration=target_ms - cur)


def _room_blend(track, sr, amount=0.16):
    """Give the dry studio voice a little room sound, like it was recorded in the scene.
    Small-room impulse response (~0.3 s), applied in 60 s blocks to keep memory low."""
    import numpy as np
    from scipy.signal import oaconvolve
    rng = np.random.default_rng(7)
    n = int(0.32 * sr)
    t = np.arange(n, dtype=np.float32) / sr
    ir = rng.standard_normal(n).astype(np.float32) * np.exp(-t / 0.06).astype(np.float32)
    ir[: int(0.005 * sr)] = 0.0                      # 5 ms before the first reflection
    k = np.exp(-np.arange(24, dtype=np.float32) / 6.0)  # rooms soak up the highs
    ir = np.convolve(ir, k / k.sum())[:n].astype(np.float32)
    ir /= float(np.sqrt((ir ** 2).sum()) or 1.0)
    wet = np.zeros(len(track) + n, dtype=np.float32)
    block = sr * 60
    for a in range(0, len(track), block):
        piece = track[a: a + block]
        if not piece.any():
            continue
        wet[a: a + len(piece) + n - 1] += oaconvolve(piece, ir, mode="full").astype(np.float32)
    return track * 0.94 + wet[: len(track)] * amount


def _build_timed_speech_track(segments, segment_files, video_duration_s, out_mp3_path, natural=True, blend=None):
    """
    Build a speech track exactly as long as the video.

    - Each line starts at its subtitle start time.
    - Lines are never cut off and never overlap: if a line is still talking when the
      next one should start, the next line waits (a tiny delay you can't hear is much
      better than chopped words or two voices at once).
    - Only lines that still don't fit get a mild time-stretch (max 1.15x, 1.25x when
      catching up after a delay) — TTS rate should already have done most of the work.
    - Built in memory and written as WAV (no MP3 generation loss). Also writes the
      MP3 at out_mp3_path for download. The WAV sits next to it (same name, .wav).

    natural=False allows stronger squeezing (up to 1.35x) for tight sync.
    Returns (out_mp3_path, duration_seconds).
    """
    import subprocess
    import wave
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from pydub import AudioSegment

    SR = 24000  # Edge-TTS native rate — no resampling for the common case
    video_len = max(SR, int(float(video_duration_s) * SR))
    gap = int(0.06 * SR)
    max_speed = 1.15 if natural else 1.35
    catchup_speed = 1.25 if natural else 1.35

    def _load(path):
        try:
            a = AudioSegment.from_file(path).set_channels(1).set_frame_rate(SR).set_sample_width(2)
            x = np.frombuffer(a.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as e:
            # pydub's error carries ffmpeg's whole banner — keep the log to one line
            print(f"[speech load {os.path.basename(path)}]: {str(e).splitlines()[0] if str(e) else e}")
            task_warn("Some voice lines came out as empty audio files — those lines are silent in the dub. "
                      "Try another voice, or Resume to make them again.")
            return None
        # trim TTS lead-in / tail silence (keeps lines from wasting their slot)
        idx = np.where(np.abs(x) > 0.004)[0]
        if len(idx) == 0:
            return None
        pad = int(0.03 * SR)
        return x[max(0, idx[0] - pad): min(len(x), idx[-1] + pad)]

    def _atempo(x, speed):
        try:
            pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()
            r = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error",
                 "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
                 "-filter:a", f"atempo={speed:.4f}", "-f", "s16le", "pipe:1"],
                input=pcm, capture_output=True, timeout=60,
            )
            if r.returncode == 0 and r.stdout:
                return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as e:
            print(f"[atempo]: {e}")
        return x

    files = list(segment_files[: len(segments)])
    with ThreadPoolExecutor(max_workers=8) as pool:
        clips = list(pool.map(_load, files))

    track = np.zeros(video_len, dtype=np.float32)
    fade_in, fade_out = int(0.005 * SR), int(0.012 * SR)
    cursor = 0
    n = len(clips)
    late_lines = 0

    for i, clip in enumerate(clips):
        if clip is None or len(clip) == 0:
            continue
        want = max(0, int(float(segments[i]["start"]) * SR))
        if want >= video_len:
            continue
        start = max(want, cursor + gap if cursor else want)
        if start >= video_len:
            break
        late = start - want
        if late > int(0.25 * SR):
            late_lines += 1

        next_want = int(float(segments[i + 1]["start"]) * SR) if i + 1 < n else video_len
        avail = max(int(0.3 * SR), next_want - start - gap)
        if len(clip) > avail:
            cap = catchup_speed if late > int(0.5 * SR) else max_speed
            speed = min(len(clip) / float(avail), cap)
            if speed > 1.03:
                clip = _atempo(clip, speed)

        clip = clip.copy()
        if len(clip) > fade_in + fade_out:
            clip[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)
            clip[-fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)
        end = min(video_len, start + len(clip))
        seg = clip[: end - start]
        if end == video_len and len(clip) > len(seg) and len(seg) > fade_out:
            seg[-fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)
        track[start:end] += seg
        cursor = end

    if late_lines:
        print(f"[speech track] {late_lines} lines started >0.25s late to avoid overlap (no words cut)")

    if blend is None:
        blend = load_config().get("blend_scene", True)
    if blend:
        try:
            track = _room_blend(track, SR)
        except Exception as e:
            print(f"[room blend]: {e}")

    wav_path = os.path.splitext(out_mp3_path)[0] + ".wav"
    pcm = (np.clip(track, -1, 1) * 32767).astype(np.int16)
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())

    cfg = load_config()
    if cfg.get("studio_cinema_dsp", True):
        _master_speech_audio(wav_path, soft_highs=bool(blend))

    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", wav_path,
         "-c:a", "libmp3lame", "-b:a", "192k", out_mp3_path],
        capture_output=True, timeout=max(300, int(video_len / SR / 5)),
    )
    return out_mp3_path, video_len / float(SR)


@app.route('/preview-voice-sample', methods=['POST'])
def preview_voice_sample():
    """Generate a quick 1-sentence audio sample to preview a voice in browser."""
    data = request.json or {}
    text = data.get("text", "ជម្រាបសួរ! នេះជាការសាកល្បងសំឡេងរបស់ DAI Dubber Pro។")
    voice_key = data.get("voice", "narrator_female")
    rate = data.get("rate")
    pitch = data.get("pitch")

    sample_dir = os.path.join(UPLOAD_FOLDER, "_preview")
    os.makedirs(sample_dir, exist_ok=True)
    sample_file = os.path.join(sample_dir, f"prev_{datetime.now().strftime('%H%M%S%f')}.mp3")

    try:
        async def _speak():
            await _synthesize_text_to_file(text, voice_key, sample_file, custom_rate=rate, custom_pitch=pitch)
        asyncio.run(_speak())

        if not os.path.exists(sample_file) or os.path.getsize(sample_file) == 0:
            return jsonify({"success": False, "error": "Failed to generate preview audio"}), 500

        cfg = load_config()
        if cfg.get("studio_cinema_dsp", True):
            _master_speech_audio(sample_file)

        import base64
        with open(sample_file, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        try:
            os.remove(sample_file)
        except Exception:
            pass

        return jsonify({"success": True, "audio": f"data:audio/mp3;base64,{b64}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/voices', methods=['GET'])
def get_all_voices():
    """Return all voices: built-in neural profiles + user's cloned voices."""
    cfg = load_config()
    cloned = load_cloned_voices()
    role_map = cfg.get("role_voice_map", {})

    builtin = []
    for k, v in KHMER_VOICE_PROFILES.items():
        builtin.append({
            "key": k,
            "name": v["name"],
            "type": "builtin",
            "desc": v.get("desc", "")
        })

    cloned_list = []
    for cv in cloned:
        role_label = CHARACTER_ROLES.get(cv.get("role", "narrator"), {}).get("label", cv.get("role", ""))
        cloned_list.append({
            "key": cv["id"],
            "name": f"🎙️ {cv['name']} ({role_label})",
            "type": "cloned",
            "role": cv.get("role", "narrator"),
            "has_elevenlabs": bool(cv.get("elevenlabs_id")),
            "sample_file": cv.get("sample_file", ""),
            "desc": cv.get("desc", "")
        })

    return jsonify({
        "success": True,
        "voices": cloned_list + builtin,
        "cloned": cloned,
        "characters": CHARACTER_ROLES,
        "role_voice_map": role_map,
        "has_elevenlabs": bool((cfg.get("ELEVENLABS_API_KEY") or "").strip())
    })


@app.route('/api/native-voices', methods=['GET'])
def api_native_voices():
    """Native (converted) voices + the user's own voices, with preview links."""
    out = []
    for k, v in NATIVE_VOICES.items():
        out.append({"key": k, "name": v["name"], "gender": v["gender"], "type": "native",
                    "preview": f"/voice-preview/{k}" if os.path.isfile(_native_preview_path(k)) else ""})
    for cv in load_cloned_voices():
        if _vc_reference_for_voice(cv.get("id")) or cv.get("elevenlabs_id"):
            g = "male" if cv.get("role") in ("man", "boy", "old_man", "villain") else "female"
            out.append({"key": cv["id"], "name": cv.get("name", "My voice"), "gender": g, "type": "mine",
                        "preview": f"/voice-preview/{cv['id']}" if os.path.isfile(_native_preview_path(cv["id"])) else ""})
    return jsonify({"success": True, "voices": out, "engine_ready": _vc_available()})


@app.route('/voice-preview/<voice_key>')
def voice_preview(voice_key):
    from flask import send_file
    path = _native_preview_path(secure_filename(voice_key))
    if not os.path.isfile(path):
        return jsonify({"success": False, "error": "No preview yet"}), 404
    return send_file(path, mimetype="audio/mpeg")


@app.route('/api/cloned-voices', methods=['GET'])
def get_cloned_voices_api():
    """Return array of cloned voices."""
    return jsonify({"success": True, "cloned": load_cloned_voices()})


@app.route('/api/cloned-voices/<voice_id>', methods=['DELETE'])
def delete_cloned_voice_api(voice_id):
    """Delete a cloned voice and its sample file."""
    voices = load_cloned_voices()
    updated = []
    deleted_item = None
    for v in voices:
        if v.get("id") == voice_id:
            deleted_item = v
        else:
            updated.append(v)

    if deleted_item:
        save_cloned_voices(updated)
        sample_path = os.path.join(CLONED_SAMPLES_DIR, deleted_item.get("sample_file", ""))
        if os.path.exists(sample_path):
            try:
                os.remove(sample_path)
            except Exception:
                pass
        return jsonify({"success": True, "message": f"Deleted voice '{deleted_item.get('name')}'"})
    return jsonify({"success": False, "error": "Voice not found"}), 404


@app.route('/cloned-voice-sample/<filename>')
def serve_cloned_voice_sample(filename):
    """Serve sample audio file for in-browser playback."""
    from flask import send_from_directory
    return send_from_directory(CLONED_SAMPLES_DIR, secure_filename(filename))


@app.route('/api/clone-voice', methods=['POST'])
def clone_voice_api():
    """
    Clone voice from YouTube / internet link or uploaded audio/video file.
    Creates clean speech sample in cloned_voices/samples/,
    optionally clones via ElevenLabs API, and saves profile to cloned_voices.json.
    """
    mode = request.form.get("mode", "url")  # 'url' or 'upload'
    name = (request.form.get("name") or "").strip()
    role = request.form.get("role", "narrator")
    start_sec = float(request.form.get("startSec", 0) or 0)
    duration_sec = float(request.form.get("durationSec", 40) or 40)
    clone_el = request.form.get("cloneToElevenlabs", "true") == "true"
    set_role_default = request.form.get("setAsRoleDefault", "true") == "true"

    if not name:
        name = f"Cloned {role.replace('_', ' ').title()}"

    try:
        if mode == "url":
            url = (request.form.get("url") or "").strip()
            if not url:
                return jsonify({"success": False, "error": "YouTube or audio URL is required"}), 400
            sample_id, sample_fname, sample_abs_path = extract_voice_sample_from_url(
                url, start_sec=start_sec, duration_sec=duration_sec
            )
            source_desc = f"Internet URL: {url[:60]}..."
        else:
            if "audio_file" not in request.files:
                return jsonify({"success": False, "error": "Audio/video file is required"}), 400
            file_obj = request.files["audio_file"]
            if not file_obj or not file_obj.filename:
                return jsonify({"success": False, "error": "No file selected"}), 400
            sample_id, sample_fname, sample_abs_path = extract_voice_sample_from_file(
                file_obj, start_sec=start_sec, duration_sec=duration_sec
            )
            source_desc = f"Upload: {file_obj.filename}"

        cfg = load_config()
        el_key = (cfg.get("ELEVENLABS_API_KEY") or "").strip()
        elevenlabs_id = ""
        el_status = ""

        if clone_el and el_key:
            ok, el_res = _elevenlabs_clone_voice(
                name, sample_abs_path, el_key,
                description=f"DAI Dubber Pro clone ({role}) - {source_desc}"
            )
            if ok:
                elevenlabs_id = el_res
                el_status = "Cloned to ElevenLabs ✓"
            else:
                el_status = f"ElevenLabs clone note: {el_res}"

        voices = load_cloned_voices()
        new_voice = {
            "id": sample_id,
            "name": name,
            "role": role,
            "sample_file": sample_fname,
            "elevenlabs_id": elevenlabs_id,
            "source_type": mode,
            "desc": source_desc,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        voices.insert(0, new_voice)
        save_cloned_voices(voices)
        if not elevenlabs_id and _vc_available():
            threading.Thread(target=_make_voice_previews, args=([sample_id],), daemon=True).start()

        if set_role_default:
            role_map = cfg.get("role_voice_map", {})
            role_map[role] = sample_id
            save_config({"role_voice_map": role_map})

        return jsonify({
            "success": True,
            "voice": new_voice,
            "elevenlabs_id": elevenlabs_id,
            "message": f"Successfully cloned voice '{name}'! {el_status}"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/set-role-voice-map', methods=['POST'])
def set_role_voice_map_api():
    """Save character role voice assignments (e.g., girl -> cloned_123, old_man -> elder_male)."""
    data = request.json or {}
    role_map = data.get("role_voice_map", {})
    cfg = save_config({"role_voice_map": role_map})
    return jsonify({"success": True, "role_voice_map": cfg.get("role_voice_map", {})})


# ═══════════════════════════════════════════════════════════
# AUTO SPEAKER & CHARACTER DETECTION (Girl, Boy, Man, Old Man...)
# ═══════════════════════════════════════════════════════════

CHARACTER_ROLES = {
    "girl": {
        "label": "👧 Girl (ក្មេងស្រី)",
        "voice": "heroine_female",
        "badge": "bg-pink-950/70 text-pink-300 border-pink-500/40"
    },
    "boy": {
        "label": "👦 Boy (ក្មេងប្រុស)",
        "voice": "hero_male",
        "badge": "bg-cyan-950/70 text-cyan-300 border-cyan-500/40"
    },
    "man": {
        "label": "👨 Man (បុរស)",
        "voice": "narrator_male",
        "badge": "bg-blue-950/70 text-blue-300 border-blue-500/40"
    },
    "woman": {
        "label": "👩 Woman (ស្ត្រី)",
        "voice": "soft_female",
        "badge": "bg-purple-950/70 text-purple-300 border-purple-500/40"
    },
    "old_man": {
        "label": "👴 Old Man (លោកតា)",
        "voice": "elder_male",
        "badge": "bg-amber-950/70 text-amber-300 border-amber-500/40"
    },
    "old_woman": {
        "label": "👵 Old Woman (លោកយាយ)",
        "voice": "elder_female",
        "badge": "bg-orange-950/70 text-orange-300 border-orange-500/40"
    },
    "villain": {
        "label": "🦹 Villain (តួអាក្រក់)",
        "voice": "villain_male",
        "badge": "bg-red-950/70 text-red-300 border-red-500/40"
    },
    "child": {
        "label": "👶 Child (ក្មេងតូច)",
        "voice": "child",
        "badge": "bg-teal-950/70 text-teal-300 border-teal-500/40"
    },
    "narrator": {
        "label": "🎙️ Narrator (អ្នកនិទាន)",
        "voice": "narrator_female",
        "badge": "bg-emerald-950/70 text-emerald-300 border-emerald-500/40"
    }
}
