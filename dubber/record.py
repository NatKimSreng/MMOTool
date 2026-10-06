"""
Record my voice: re-voice a finished dub with your own recordings. Its page and routes.
"""
import os
import subprocess
import json
import asyncio
from flask import render_template, request, jsonify
from datetime import datetime
from .core import (
    app,
    load_history,
)
from .tasks import (
    _ACTIVE_STATES,
    _TASKS,
    _TASKS_LOCK,
    _start_task,
    _task_state,
    register_task_runner,
)
from .media import (
    _burn_subtitles,
    _get_media_duration,
    _mix_audio_tracks,
    _mux_video_with_audio,
)
from .voices import (
    NATIVE_VOICES,
    _build_timed_speech_track,
    _run_voice_conversion,
    _synthesize_text_to_file,
    _vc_available,
    _vc_reference_for_voice,
    _voice_exists,
    _voice_expressiveness,
    _voice_gender,
)


# ═══════════════════════════════════════════════════════════
# RECORD MY VOICE — act the lines yourself; the app turns your recording into
# the character's voice (human timing + emotion, AI only changes the sound).
# ═══════════════════════════════════════════════════════════
REC_DIRNAME = "04_my_recordings"


def _save_job_segments(job_dir, segments, segment_files):
    try:
        rows = []
        for i, seg in enumerate(segments):
            f = segment_files[i] if i < len(segment_files) else ""
            rows.append({
                "i": i + 1, "start": round(float(seg["start"]), 3), "end": round(float(seg["end"]), 3),
                "text": seg.get("text", ""), "source": seg.get("source", ""), "voice": seg.get("voice", ""),
                "role": seg.get("role", ""), "emotion": seg.get("emotion", ""), "speaker": seg.get("speaker", ""),
                "file": os.path.relpath(f, job_dir) if f else "",
            })
        with open(os.path.join(job_dir, "segments.json"), "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=1)
    except Exception as e:
        print(f"[save segments]: {e}")


def _save_job_settings(job_dir, data):
    try:
        with open(os.path.join(job_dir, "job.json"), "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1)
    except Exception as e:
        print(f"[save job]: {e}")


def _load_record_job(job_dir):
    """job.json + segments.json of a finished dub, or None if it can't be re-voiced."""
    if not job_dir or not os.path.isdir(job_dir):
        return None
    jp, sp = os.path.join(job_dir, "job.json"), os.path.join(job_dir, "segments.json")
    if not (os.path.isfile(jp) and os.path.isfile(sp)):
        return None
    with open(jp, encoding="utf-8") as fh:
        job = json.load(fh)
    with open(sp, encoding="utf-8") as fh:
        segs = json.load(fh)
    return job, segs


def _recording_path(job_dir, idx, kind=""):
    return os.path.join(job_dir, REC_DIRNAME, f"line_{int(idx):04d}{kind}.wav")


def _revoice_lines(job_dir, job, segs, idxs, voice_key, progress_cb=None):
    """Speak the given lines again with another voice (TTS → native conversion). Updates segs."""
    speech_dir = os.path.join(job_dir, "03_speech")
    os.makedirs(speech_dir, exist_ok=True)
    rate = job.get("rate", "+3%")
    stamp = datetime.now().strftime("%H%M%S")
    outs = {i: os.path.join(speech_dir, f"segment_{i + 1:04d}_{voice_key}_{stamp}.mp3") for i in idxs}

    async def _speak_all():
        sem = asyncio.Semaphore(8)

        async def one(i):
            async with sem:
                await _synthesize_text_to_file(segs[i]["text"], voice_key, outs[i], custom_rate=rate,
                                               emotion=segs[i].get("emotion"))
        await asyncio.gather(*(one(i) for i in idxs))
    asyncio.run(_speak_all())

    items = []
    ref = _vc_reference_for_voice(voice_key)
    for i in idxs:
        if not os.path.isfile(outs[i]):
            continue
        segs[i]["voice"] = voice_key
        segs[i]["file"] = os.path.relpath(outs[i], job_dir)
        if ref:
            items.append({"src": outs[i], "dst": os.path.splitext(outs[i])[0] + "_native.wav", "reference": ref,
                          "idx": i, "f0_scale": _voice_expressiveness(voice_key)})
    done = _run_voice_conversion(items, progress_cb=progress_cb) if items else set()
    for it in items:
        if it["dst"] in done:
            segs[it["idx"]]["file"] = os.path.relpath(it["dst"], job_dir)


def _rebuild_with_recordings_task(job_dir, target="auto", changes=None):
    """Apply character voice changes + recorded lines, then rebuild the video."""
    progress_state = _task_state()
    try:
        loaded = _load_record_job(job_dir)
        if not loaded:
            raise ValueError("This project can't be re-voiced (made before this feature existed).")
        job, segs = loaded
        changes = {k: v for k, v in (changes or {}).items() if k and _voice_exists(v)}
        for n, (spk, vkey) in enumerate(changes.items(), 1):
            idxs = [i for i, x in enumerate(segs) if x.get("speaker") == spk and x.get("voice") != vkey]
            if not idxs:
                continue
            progress_state["status"] = f"New voice for character {spk} ({len(idxs)} lines)..."
            progress_state["percent"] = 5

            def _cb(done, total):
                progress_state["status"] = f"New voice for character {spk}: {done}/{total} lines..."
                progress_state["percent"] = 5 + int(20 * done / max(total, 1))
            _revoice_lines(job_dir, job, segs, idxs, vkey, progress_cb=_cb)
        if changes:
            with open(os.path.join(job_dir, "segments.json"), "w", encoding="utf-8") as fh:
                json.dump(segs, fh, ensure_ascii=False, indent=1)
        files = [os.path.join(job_dir, s["file"]) if s.get("file") else "" for s in segs]
        items, used = [], 0
        for i, s in enumerate(segs):
            rec = _recording_path(job_dir, i + 1)
            if not os.path.isfile(rec):
                continue
            used += 1
            if target == "original":
                files[i] = rec
                continue
            vkey = target if target not in ("auto", "") else (s.get("voice") or job.get("voice_key"))
            if not _vc_reference_for_voice(vkey):
                g = _voice_gender(vkey) or "female"
                vkey = "native_male_1" if g == "male" else "native_girl_1"
            ref = _vc_reference_for_voice(vkey)
            if ref:
                # f0_scale 1.0: your own acting already has the right melody
                items.append({"src": rec, "dst": _recording_path(job_dir, i + 1, "_voiced"), "reference": ref,
                              "idx": i, "f0_scale": 1.0})
            else:
                files[i] = rec
        if not used and not changes:
            raise ValueError("Nothing to change yet — pick a new voice for a character, or record a line.")

        if used:
            progress_state["status"] = f"Turning your {used} recorded lines into the character voices..."
        progress_state["percent"] = max(progress_state.get("percent", 0), 25)

        def _cb(done, total):
            progress_state["status"] = f"Converting your voice {done}/{total}..."
            progress_state["percent"] = 25 + int(25 * done / max(total, 1))
        done = _run_voice_conversion(items, progress_cb=_cb) if items else set()
        for it in items:
            files[it["idx"]] = it["dst"] if it["dst"] in done else it["src"]

        progress_state["status"] = "Building the voice track..."
        progress_state["percent"] = 55
        speech_dir = os.path.join(job_dir, "03_speech")
        os.makedirs(speech_dir, exist_ok=True)
        full_mp3 = os.path.join(speech_dir, "full_my_voice.mp3")
        _build_timed_speech_track(segs, files, job.get("video_duration") or 0, full_mp3, blend=job.get("blend_scene", True))
        full_wav = os.path.splitext(full_mp3)[0] + ".wav"

        progress_state["status"] = "Mixing with the movie sound..."
        progress_state["percent"] = 70
        mix_wav = os.path.join(speech_dir, "audio_my_voice_mix.wav")
        video = job["video"]
        if _mix_audio_tracks(video, full_wav, mix_wav, dubbing_mode=job.get("dubbing_mode", "duck"),
                             bgm_volume=job.get("bgm_volume", 0.25), voice_volume=job.get("voice_volume", 1.0)):
            audio, mode = mix_wav, "premixed"
        else:
            audio, mode = full_wav, job.get("dubbing_mode", "duck")

        progress_state["status"] = "Writing the video with your voice..."
        progress_state["percent"] = 80
        out = os.path.join(job_dir, "06_video_my_voice.mp4")
        logo = job.get("logo_path") if job.get("logo_path") and os.path.isfile(job.get("logo_path") or "") else None
        if job.get("burn_subs") and job.get("srt") and os.path.isfile(job["srt"]):
            _burn_subtitles(video, job["srt"], out, audio_path=audio, dubbing_mode=mode,
                            bgm_volume=job.get("bgm_volume", 0.25), voice_volume=job.get("voice_volume", 1.0),
                            font_size=job.get("sub_size", 22), sub_color=job.get("sub_color", "yellow"),
                            sub_box=job.get("sub_box", False), sub_pos=job.get("sub_pos", "bottom"),
                            logo_path=logo, logo_pos=job.get("logo_pos", "top-right"),
                            logo_size=job.get("logo_size", "medium"), logo_opacity=job.get("logo_opacity", 0.85),
                            aspect_ratio=job.get("aspect_ratio", "original"), hide_box=job.get("hide_box", ""))
        else:
            _mux_video_with_audio(video, audio, out, dubbing_mode=mode, logo_path=logo,
                                  logo_pos=job.get("logo_pos", "top-right"), logo_size=job.get("logo_size", "medium"),
                                  logo_opacity=job.get("logo_opacity", 0.85), aspect_ratio=job.get("aspect_ratio", "original"),
                                  hide_box=job.get("hide_box", ""))
        for tmp in (mix_wav, full_wav):
            try:
                os.remove(tmp)
            except OSError:
                pass
        what = ", ".join(x for x in (f"{len(changes)} character voice(s) changed" if changes else "",
                                     f"{used} lines in your voice" if used else "") if x)
        progress_state.update({
            "status": f"Done! {what}: {os.path.basename(job_dir)}/06_video_my_voice.mp4",
            "percent": 100, "final_video": out, "result_path": job_dir, "job_dir": job_dir,
            "final_srt": job.get("srt", ""), "final_audio": full_mp3,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        progress_state["status"] = f"Error: {e}"
        progress_state["percent"] = 0
    finally:
        progress_state["is_processing"] = False


@app.route('/record')
def record_page():
    return render_template('record.html')


@app.route('/api/record/jobs')
def api_record_jobs():
    items = []
    for h in reversed(load_history()[-40:]):
        d = h.get("output_path", "")
        loaded = _load_record_job(d) if d else None
        if not loaded:
            continue
        rec_dir = os.path.join(d, REC_DIRNAME)
        n_rec = len([f for f in os.listdir(rec_dir) if f.endswith(".wav") and "_" not in f[5:]]) if os.path.isdir(rec_dir) else 0
        items.append({"dir": d, "name": h.get("filename", ""), "date": h.get("date", ""),
                      "lines": len(loaded[1]), "recorded": n_rec})
    return jsonify({"success": True, "items": items})


@app.route('/api/record/job')
def api_record_job():
    d = request.args.get("dir", "")
    loaded = _load_record_job(d)
    if not loaded:
        return jsonify({"success": False, "error": "This project can't be re-voiced. Dub it again with the new version first."}), 404
    job, segs = loaded
    chars = {}
    for s in segs:
        s["recorded"] = os.path.isfile(_recording_path(d, s["i"]))
        s["voice_name"] = (NATIVE_VOICES.get(s.get("voice"), {}) or {}).get("name") or s.get("voice", "")
        s["gender"] = _voice_gender(s.get("voice")) or ""
        sp = s.get("speaker")
        if sp:
            c = chars.setdefault(sp, {"id": sp, "lines": 0, "voices": {}, "first": s["start"], "sample": None})
            c["lines"] += 1
            c["voices"][s.get("voice", "")] = c["voices"].get(s.get("voice", ""), 0) + 1
            dur = s["end"] - s["start"]
            if dur >= 1.5 and (c["sample"] is None or abs(dur - 3.0) < abs(c["sample"][1] - c["sample"][0] - 3.0)):
                c["sample"] = [s["start"], s["end"]]
    characters = []
    for c in sorted(chars.values(), key=lambda c: -c["lines"]):
        voice = max(c["voices"], key=c["voices"].get)
        characters.append({"id": c["id"], "lines": c["lines"], "voice": voice, "gender": _voice_gender(voice) or "",
                           "voice_name": (NATIVE_VOICES.get(voice, {}) or {}).get("name") or voice,
                           "sample": c["sample"] or [c["first"], c["first"] + 3]})
    final = ""
    for name in ("06_video_my_voice.mp4", "05_video_khmer_voice_subs.mp4", "04_video_khmer_voice.mp4"):
        if os.path.isfile(os.path.join(d, name)):
            final = os.path.join(d, name)
            break
    return jsonify({"success": True, "name": job.get("name", os.path.basename(d)), "video": job.get("video", ""),
                    "final_video": final, "segments": segs, "characters": characters, "engine_ready": _vc_available()})


@app.route('/api/record/line', methods=['POST'])
def api_record_line():
    d = request.form.get("dir", "")
    idx = int(request.form.get("idx", "0") or 0)
    if not _load_record_job(d) or idx < 1 or "audio" not in request.files:
        return jsonify({"success": False, "error": "Bad request"}), 400
    os.makedirs(os.path.join(d, REC_DIRNAME), exist_ok=True)
    raw = _recording_path(d, idx, "_raw").replace(".wav", ".webm")
    request.files["audio"].save(raw)
    out = _recording_path(d, idx)
    # clean up a phone/laptop mic: rumble, steady noise, level; trim silence at both ends
    af = ("highpass=f=80,afftdn=nf=-28,"
          "silenceremove=start_periods=1:start_threshold=-42dB:start_silence=0.08,areverse,"
          "silenceremove=start_periods=1:start_threshold=-42dB:start_silence=0.12,areverse,"
          "loudnorm=I=-18:TP=-2")
    r = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", raw, "-af", af,
                        "-ac", "1", "-ar", "44100", out], capture_output=True, timeout=120)
    try:
        os.remove(raw)
    except OSError:
        pass
    for stale in (_recording_path(d, idx, "_voiced"),):
        if os.path.isfile(stale):
            os.remove(stale)
    if r.returncode != 0 or not os.path.isfile(out):
        return jsonify({"success": False, "error": "Could not read the recording"}), 500
    return jsonify({"success": True, "duration": round(_get_media_duration(out) or 0, 2)})


@app.route('/api/record/delete', methods=['POST'])
def api_record_delete():
    data = request.json or {}
    d, idx = data.get("dir", ""), int(data.get("idx", 0) or 0)
    if not _load_record_job(d):
        return jsonify({"success": False}), 400
    for kind in ("", "_voiced"):
        f = _recording_path(d, idx, kind)
        if os.path.isfile(f):
            os.remove(f)
    return jsonify({"success": True})


@app.route('/api/record/take')
def api_record_take():
    from flask import send_file
    d, idx = request.args.get("dir", ""), int(request.args.get("idx", "0") or 0)
    f = _recording_path(d, idx)
    if not _load_record_job(d) or not os.path.isfile(f):
        return jsonify({"success": False}), 404
    return send_file(f, mimetype="audio/wav")


@app.route('/api/record/build', methods=['POST'])
def api_record_build():
    data = request.json or {}
    d = data.get("dir", "")
    if not _load_record_job(d):
        return jsonify({"success": False, "error": "Project not found"}), 404
    with _TASKS_LOCK:
        busy = any(t.get("state") in _ACTIVE_STATES and (t.get("params") or {}).get("job_dir") == d
                   for t in _TASKS.values())
    if busy:
        return jsonify({"success": False, "busy": True, "error": "This video is already being rebuilt."}), 409
    changes = data.get("changes") or {}
    st = _start_task("_rebuild_with_recordings_task", dict(job_dir=d, target=data.get("target", "auto"), changes=changes),
                     name=f"Re-voice: {os.path.basename(d.rstrip(os.sep))}", kind="record")
    return jsonify({"success": True, "task_id": st["id"]})


register_task_runner(_rebuild_with_recordings_task)
