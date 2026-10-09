"""
The main dub job: video → transcript → Khmer script → voices → dubbed video, with checkpoints
so a paused job resumes where it stopped. Dub routes.
"""
import os
import subprocess
import threading
import json
import asyncio
from flask import request, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
import uuid
import yt_dlp
from .core import (
    _strip_foreign_script,
    DEFAULT_OUTPUT_FOLDER,
    UPLOAD_FOLDER,
    _MEDIA_EXT,
    _REJECT_EXT,
    app,
    load_config,
    save_config,
    save_to_history,
)
from .tasks import (
    TaskState,
    TaskStopped,
    _ACTIVE_REVIEW_EVENTS,
    _REVIEW_EDITS,
    _set_task_param,
    _start_task,
    _task_state,
    _wait_for_user,
    register_task_runner,
)
from .media import (
    _burn_subtitles,
    _expand_slots_into_gaps,
    _get_media_duration,
    _merge_video_files,
    _mix_audio_tracks,
    _mux_video_with_audio,
    _normalize_segment_timings,
    _write_bilingual_srt,
    _write_srt,
)
from .asr import (
    _extract_audio_wav,
    _has_khmer,
    _transcribe_source_lang,
)
from .translate import (
    _translate_to_khmer,
)
from .voices import (
    _build_timed_speech_track,
    _convert_segments_to_voices,
    _detect_line_emotions,
    _synthesize_text_to_file,
    _vc_reference_for_voice,
    get_voice_settings,
)
from .characters import (
    _assign_character_voices,
)
from .record import (
    _save_job_segments,
    _save_job_settings,
)


def _safe_job_folder_name(name, fallback="job"):
    """Clean user/folder name for Windows/mac paths."""
    name = (name or "").strip()
    if not name:
        name = fallback
    # keep letters, numbers, Khmer, spaces, dash, underscore
    cleaned = []
    for ch in name:
        if ch.isalnum() or ch in " -_." or ("\u1780" <= ch <= "\u17ff"):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    name = "".join(cleaned).strip(" ._")
    name = " ".join(name.split())  # collapse spaces
    if not name:
        name = fallback
    return name[:80]


RESUME_FILE = "_resume.json"


def _load_checkpoint(job_dir):
    """Steps an earlier run of this job already finished (see pipeline_cn_en_to_khmer_task)."""
    try:
        with open(os.path.join(job_dir, RESUME_FILE), encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_checkpoint(job_dir, ck):
    try:
        path = os.path.join(job_dir, RESUME_FILE)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(ck, f, ensure_ascii=False, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
        os.replace(path + ".tmp", path)
    except Exception as e:
        print(f"[checkpoint] save failed: {e}")


def _clear_checkpoint(job_dir):
    try:
        os.remove(os.path.join(job_dir, RESUME_FILE))
    except OSError:
        pass


def pipeline_cn_en_to_khmer_task(
    file_path,
    custom_output_dir,
    source_lang="auto",
    do_tts=False,
    voice_key="narrator_female",
    rate="-5%",
    pitch="+0Hz",
    job_name="",
    burn_subs=True,
    dubbing_mode="duck",
    bgm_volume=0.25,
    voice_volume=1.0,
    translate_engine="auto",
    translate_style="recap",
    sub_color="yellow",
    sub_size=22,
    sub_box=False,
    sub_pos="bottom",
    bilingual_subs=False,
    auto_detect_voice=True,
    logo_path=None,
    logo_pos="top-right",
    logo_size="medium",
    logo_opacity=0.85,
    aspect_ratio="original",
    hide_box="",
    review_script=False,
    cast_mode="two",
    blend_scene=True,
    resume_dir=None,
):
    """
    Full DAI Dubber Pro Style pipeline (resume_dir = job folder of an earlier run: finished
    steps saved in its _resume.json are skipped):
      outputs/{job_name}/
        00_auto_merged.mp4             (if multiple clips uploaded)
        01_source.srt
        02_khmer.srt
        02_bilingual.srt               (if bilingual_subs enabled)
        03_speech/
          full.mp3                     (Khmer speech track)
          audio_full_mix.mp3           (Khmer speech + Smart ducked BGM)
          segment_xxxx.mp3
        04_video_khmer_voice.mp4       (Video dubbed with chosen mode: duck/mute/mix)
        05_video_khmer_voice_subs.mp4  (Video + Dubbed audio + Styled Khmer / Bilingual subtitles + Logo)
        info.txt
    """
    progress_state = _task_state()
    audio_path = None
    try:
        progress_state["is_processing"] = True
        progress_state["status"] = "Starting pipeline..."
        progress_state["percent"] = 5

        clips = list(file_path) if isinstance(file_path, list) else [file_path]
        is_multi_clip = len(clips) > 1
        if resume_dir and os.path.isdir(resume_dir):
            # continue an earlier run in its own folder
            job_dir = os.path.abspath(resume_dir)
            job_folder_name = os.path.basename(job_dir)
            ck = _load_checkpoint(job_dir)
            folder_label = ck.get("folder_label") or job_folder_name
            progress_state["status"] = "Resuming — reusing the steps already done..."
        else:
            stem = os.path.splitext(os.path.basename(clips[0]))[0]
            if "_" in stem and stem.rsplit("_", 1)[-1].isdigit():
                stem = stem.rsplit("_", 1)[0]
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_label = _safe_job_folder_name(
                job_name, fallback=f"{stem}_merged" if is_multi_clip else (stem or "job"))
            job_folder_name = f"{folder_label}_{stamp}"
            base_root = custom_output_dir if custom_output_dir else DEFAULT_OUTPUT_FOLDER
            job_dir = os.path.join(base_root, job_folder_name)
            ck = {"folder_label": folder_label}
        speech_dir = os.path.join(job_dir, "03_speech")
        os.makedirs(speech_dir, exist_ok=True)
        _save_checkpoint(job_dir, ck)
        # from here on Pause / app close can be resumed in this same folder
        _set_task_param("resume_dir", os.path.abspath(job_dir))
        progress_state["job_dir"] = os.path.abspath(job_dir)
        progress_state["result_path"] = os.path.abspath(job_dir)

        if is_multi_clip:
            merged_video = os.path.join(job_dir, "00_auto_merged.mp4")
            # merge_v 2 = mixed audio formats are normalized; older merges may have broken audio
            if ck.get("merged") and ck.get("merge_v") == 2 and os.path.isfile(merged_video):
                progress_state["status"] = "Using the merged video from before..."
            else:
                progress_state["status"] = f"Auto-merging {len(clips)} clips into 1 full continuous video..."
                progress_state["percent"] = 5
                _merge_video_files(clips, merged_video, progress_cb=lambda s: progress_state.update({"status": s}))
                ck["merged"] = True
                ck["merge_v"] = 2
                _save_checkpoint(job_dir, ck)
            file_path = merged_video
        else:
            file_path = clips[0]

        video_duration = _get_media_duration(file_path)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if not video_duration or video_duration < 0.5:
            raise ValueError(
                f"Cannot read video duration ({os.path.basename(file_path)}). "
                "File may be corrupt — re-merge with Safe re-encode, then upload merged.mp4."
            )
        progress_state["status"] = (
            f"Video OK: {video_duration/60:.1f} min · {size_mb:.0f} MB → extracting audio…"
        )
        progress_state["percent"] = 10

        def _audio():
            # the speech audio is only extracted when a step still needs it (not on most resumes)
            nonlocal audio_path
            if not audio_path:
                audio_path = _extract_audio_wav(file_path, progress_state)
            return audio_path

        if ck.get("src_segments"):
            src_segments, engine = ck["src_segments"], ck.get("asr_engine", "saved")
            progress_state["status"] = f"Using the saved transcript ({len(src_segments)} lines)..."
        else:
            src_segments, engine = _transcribe_source_lang(
                _audio(), progress_state, lang=source_lang
            )
            if not src_segments:
                raise ValueError("No speech detected in video")

            src_segments = _normalize_segment_timings(src_segments, video_duration)
            ck.update(src_segments=src_segments, asr_engine=engine)
            _save_checkpoint(job_dir, ck)

        progress_state["status"] = f"Job folder: {job_folder_name}"
        progress_state["percent"] = 20

        # 1) Source SRT
        src_srt = os.path.join(job_dir, "01_source.srt")
        _write_srt(src_srt, src_segments)
        last_t = src_segments[-1]["end"] if src_segments else 0
        progress_state["status"] = (
            f"Source SRT OK ({engine}) — 0→{last_t:.1f}s / video {video_duration:.1f}s"
        )
        progress_state["percent"] = 50

        # 2) Khmer SRT
        if ck.get("km_segments"):
            km_segments = ck["km_segments"]
            progress_state["status"] = f"Using the saved Khmer script ({len(km_segments)} lines)..."
        else:
            def _save_translation(partial):
                ck["translate_partial"] = partial
                _save_checkpoint(job_dir, ck)

            km_segments = _translate_to_khmer(
                src_segments, progress_state, engine=translate_engine, style=translate_style,
                saved=ck.get("translate_partial"), save_partial=_save_translation,
            )
            for i, s in enumerate(src_segments):
                if i < len(km_segments):
                    km_segments[i]["start"] = s["start"]
                    km_segments[i]["end"] = s["end"]
            km_segments = _normalize_segment_timings(km_segments, video_duration)
            ck["km_segments"] = km_segments
            ck.pop("translate_partial", None)
            _save_checkpoint(job_dir, ck)

        for s in km_segments:
            s["text"] = _strip_foreign_script(s.get("text", ""))
        km_srt = os.path.join(job_dir, "02_khmer.srt")
        _write_srt(km_srt, km_segments)
        n_kh = sum(1 for s in km_segments if _has_khmer(s.get("text", "")))

        # Optional Bilingual SRT
        bi_srt = ""
        if bilingual_subs:
            bi_srt = os.path.join(job_dir, "02_bilingual.srt")
            _write_bilingual_srt(bi_srt, km_segments, src_segments)

        progress_state["status"] = (
            f"Khmer SRT ({n_kh}/{len(km_segments)}) — timed to {video_duration:.1f}s"
        )
        progress_state["percent"] = 85

        result_path = os.path.abspath(job_dir)

        # 3) Speech matched to video
        full_mp3 = ""
        full_mix_mp3 = ""
        dubbed_path = ""
        final_audio = ""
        final_mode = dubbing_mode
        actual_dur = 0.0
        if do_tts:
            if ck.get("cast_done") or ck.get("detected"):
                progress_state["status"] = "Using the saved voice for each line..."
            elif auto_detect_voice:
                progress_state["status"] = "Detecting who speaks each line (voice pitch + AI)..."
                try:
                    summary = _assign_character_voices(
                        km_segments, _audio(), engine=translate_engine, progress_state=progress_state,
                        native=bool(_vc_reference_for_voice(voice_key)),
                        two_voices=(cast_mode == "two"), main_voice=voice_key, media_path=file_path,
                    )
                    print(f"[characters] {summary}")
                    progress_state["status"] = f"{summary.get('characters') or 'Some'} characters found: " + ", ".join(
                        f"{n} lines {r.replace('_', ' ')}" for r, n in sorted(summary["roles"].items(), key=lambda x: -x[1])
                    )
                except Exception as detect_err:
                    print(f"[auto_detect in pipeline error]: {detect_err}")
            if not ck.get("detected"):
                ck.update(km_segments=km_segments, detected=True)
                _save_checkpoint(job_dir, ck)

            # Checkpoint: Interactive Script & Voice Review before Synthesis
            if review_script and not ck.get("cast_done"):
                review_id = str(uuid.uuid4())[:8]
                review_evt = threading.Event()
                _ACTIVE_REVIEW_EVENTS[review_id] = review_evt
                progress_state["waiting_for_review"] = True
                progress_state["review_job_id"] = review_id
                progress_state["review_segments"] = [
                    {
                        "id": i + 1,
                        "start": round(float(s.get("start", 0)), 2),
                        "end": round(float(s.get("end", 0)), 2),
                        "source": s.get("source", ""),
                        "text": s.get("text", ""),
                        "voice": s.get("voice", voice_key)
                    }
                    for i, s in enumerate(km_segments)
                ]
                progress_state["status"] = ("Waiting for you to check the Khmer script — no time limit. "
                                            "Other jobs keep running meanwhile.")
                print(f"[review] Job {review_id} paused for user script review.")
                confirmed = _wait_for_user(review_evt)
                progress_state["waiting_for_review"] = False
                progress_state["review_job_id"] = ""
                progress_state["review_segments"] = []
                if review_id in _ACTIVE_REVIEW_EVENTS:
                    del _ACTIVE_REVIEW_EVENTS[review_id]
                if getattr(progress_state, "stop_requested", False):
                    raise TaskStopped()   # paused during the review: ask again on resume
                if confirmed and review_id in _REVIEW_EDITS:
                    edits = _REVIEW_EDITS.pop(review_id, None)
                    if edits and isinstance(edits, list):
                        for item in edits:
                            idx = int(item.get("id", 0)) - 1
                            if 0 <= idx < len(km_segments):
                                if "text" in item and item["text"] is not None:
                                    km_segments[idx]["text"] = _strip_foreign_script(str(item["text"]))
                                if "voice" in item and item["voice"]:
                                    km_segments[idx]["voice"] = str(item["voice"]).strip()
                        _write_srt(km_srt, km_segments)
                        if bilingual_subs and bi_srt:
                            _write_bilingual_srt(bi_srt, km_segments, src_segments)
                        print(f"[review] Job {review_id} resumed with {len(edits)} verified lines!")
            if not ck.get("cast_done"):
                ck.update(km_segments=km_segments, cast_done=True)
                _save_checkpoint(job_dir, ck)

            progress_state["status"] = "Generating Khmer character voices..."
            progress_state["percent"] = 86
            default_voice, def_rate, def_pitch = get_voice_settings(
                voice_key, custom_rate=rate, custom_pitch=pitch
            )

            # Expand each cue into silence gaps so voice has room (still starts on time)
            new_timing = not ck.get("timed_segs")
            timed_segs = ck["timed_segs"] if not new_timing else _expand_slots_into_gaps(km_segments, video_duration)
            segment_files = []

            # Many lines at once. Every line is spoken at the voice's normal pace; a line
            # that is too long for its gap is re-spoken faster by the TTS engine itself
            # (sounds natural) instead of being stretched or chopped afterwards.
            cfg_now = load_config()
            tts_workers = int(cfg_now.get("tts_workers", 8))
            if new_timing:
                if cfg_now.get("lively_voice", True):
                    progress_state["status"] = "Giving each line the right mood (excited, sad, angry…)..."
                    for blk, mood in zip(timed_segs, _detect_line_emotions(timed_segs)):
                        blk["emotion"] = mood
                ck.update(timed_segs=timed_segs, tts_done=[])
                _save_checkpoint(job_dir, ck)
            segment_files = [
                os.path.join(speech_dir, f"segment_{i:04d}.mp3") for i in range(1, len(timed_segs) + 1)
            ]
            # lines already spoken in an earlier run (file finished) are kept
            tts_done = {i for i in ck.get("tts_done", [])
                        if i < len(segment_files) and os.path.isfile(segment_files[i])
                        and os.path.getsize(segment_files[i]) > 0}
            done_count = [len(tts_done)]
            saved_voices = ck.get("segment_files") if ck.get("voices_done") else None
            voices_ready = bool(saved_voices) and len(saved_voices) == len(timed_segs) and all(
                os.path.isfile(f) for f in saved_voices)

            async def _gen():
                sem = asyncio.Semaphore(tts_workers)

                async def _one(idx, blk):
                    if idx in tts_done:
                        return
                    async with sem:
                        v_key = blk.get("voice") or voice_key
                        out = segment_files[idx]
                        nxt = float(timed_segs[idx + 1]["start"]) if idx + 1 < len(timed_segs) else video_duration
                        window = max(0.4, nxt - float(blk["start"]) - 0.1)
                        for attempt in range(3):
                            ok = await _synthesize_text_to_file(
                                blk["text"], v_key, out, custom_rate=rate, custom_pitch=pitch,
                                emotion=blk.get("emotion"),
                            )
                            if ok:
                                break
                            await asyncio.sleep(1.5 * (attempt + 1))
                        spoken = await asyncio.to_thread(_get_media_duration, out) if os.path.exists(out) else 0
                        if spoken and spoken - 0.25 > window * 1.05:
                            boost = min(35, int(((spoken - 0.25) / window - 1.0) * 100) + 3)
                            await _synthesize_text_to_file(
                                blk["text"], v_key, out, custom_rate=rate, custom_pitch=pitch,
                                speed_boost_pct=boost, emotion=blk.get("emotion"),
                            )
                        done_count[0] += 1
                        if os.path.isfile(out):
                            tts_done.add(idx)
                            if len(tts_done) % 10 == 0:
                                ck["tts_done"] = sorted(tts_done)
                                _save_checkpoint(job_dir, ck)
                        progress_state["status"] = f"Voice {done_count[0]}/{len(timed_segs)}..."
                        progress_state["percent"] = 86 + int(3 * done_count[0] / max(len(timed_segs), 1))

                try:
                    await asyncio.gather(*(_one(i, b) for i, b in enumerate(timed_segs)))
                finally:
                    ck["tts_done"] = sorted(tts_done)
                    _save_checkpoint(job_dir, ck)

            if voices_ready:
                segment_files = list(saved_voices)
                progress_state["status"] = f"Using the {len(segment_files)} voice lines made before..."
            else:
                if len(tts_done) < len(timed_segs):
                    if tts_done:
                        progress_state["status"] = (
                            f"Continuing voices — {len(tts_done)}/{len(timed_segs)} already made...")
                    asyncio.run(_gen())

                # Native voices: convert the TTS lines to the real speaker's voice (GPU)
                def _vc_progress(done, total):
                    progress_state["status"] = f"Making voices sound native {done}/{total}..."
                    progress_state["percent"] = 89 + int(3 * done / max(total, 1))
                n_conv, n_want = _convert_segments_to_voices(timed_segs, segment_files, voice_key, progress_cb=_vc_progress)
                if n_want and n_conv < n_want:
                    print(f"[voice conversion] {n_conv}/{n_want} lines converted — the rest use the standard voice")
                if getattr(progress_state, "stop_requested", False):
                    raise TaskStopped()   # conversion was cut short: redo it on resume
                ck.update(segment_files=segment_files, voices_done=True)
                _save_checkpoint(job_dir, ck)
            _save_job_segments(job_dir, timed_segs, segment_files)
            progress_state["status"] = "Building timed Khmer voice track..."

            full_mp3 = os.path.join(speech_dir, "full.mp3")
            try:
                path, actual_dur = _build_timed_speech_track(
                    timed_segs, segment_files, video_duration or last_t, full_mp3, blend=blend_scene
                )
            except Exception as merge_err:
                print(f"[tts timed merge failed]: {merge_err}")
                full_mp3 = ""

            # Mix the final soundtrack ONCE (lossless WAV); video encodes just copy it in
            full_wav = os.path.splitext(full_mp3)[0] + ".wav" if full_mp3 else ""
            voice_src = full_wav if (full_wav and os.path.exists(full_wav)) else full_mp3
            if voice_src and os.path.exists(voice_src):
                progress_state["status"] = f"Mixing Khmer voice with background ({dubbing_mode})..."
                progress_state["percent"] = 93
                mix_wav = os.path.join(speech_dir, "audio_full_mix.wav")
                if _mix_audio_tracks(
                    file_path, voice_src, mix_wav,
                    dubbing_mode=dubbing_mode, bgm_volume=bgm_volume, voice_volume=voice_volume,
                    progress_cb=lambda st: progress_state.update({"status": st}),
                ):
                    final_audio, final_mode = mix_wav, "premixed"
                    full_mix_mp3 = os.path.join(speech_dir, "audio_full_mix.mp3")
                    subprocess.run(
                        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", mix_wav,
                         "-c:a", "libmp3lame", "-b:a", "192k", full_mix_mp3],
                        capture_output=True, timeout=1800,
                    )
                else:
                    final_audio, final_mode = voice_src, dubbing_mode

            # 04 (no subtitles) only when subtitles are off — saves a whole extra encode
            if final_audio and not burn_subs:
                progress_state["status"] = "Writing dubbed video..."
                progress_state["percent"] = 94
                dubbed_path = os.path.join(job_dir, "04_video_khmer_voice.mp4")
                try:
                    _mux_video_with_audio(
                        file_path, final_audio, dubbed_path,
                        dubbing_mode=final_mode, bgm_volume=bgm_volume, voice_volume=voice_volume,
                        logo_path=logo_path, logo_pos=logo_pos, logo_size=logo_size, logo_opacity=logo_opacity,
                        aspect_ratio=aspect_ratio, hide_box=hide_box
                    )
                except Exception as mux_err:
                    print(f"[mux video]: {mux_err}")
                    progress_state["status"] = f"Video export problem: {mux_err}"
                    dubbed_path = ""

        # 4) Burn Subtitles
        subs_path = ""
        target_srt = bi_srt if (bilingual_subs and os.path.isfile(bi_srt)) else km_srt
        if burn_subs and os.path.isfile(target_srt):
            progress_state["status"] = "Burning Khmer subtitles on video (styled)..."
            progress_state["percent"] = 97
            try:
                subs_path = os.path.join(job_dir, "05_video_khmer_voice_subs.mp4")
                _burn_subtitles(
                    file_path,
                    target_srt,
                    subs_path,
                    audio_path=final_audio if (do_tts and final_audio) else None,
                    dubbing_mode=final_mode,
                    bgm_volume=bgm_volume,
                    voice_volume=voice_volume,
                    font_size=sub_size,
                    sub_color=sub_color,
                    sub_box=sub_box,
                    sub_pos=sub_pos,
                    logo_path=logo_path,
                    logo_pos=logo_pos,
                    logo_size=logo_size,
                    logo_opacity=logo_opacity,
                    aspect_ratio=aspect_ratio,
                    hide_box=hide_box
                )
            except Exception as burn_err:
                print(f"[burn subs]: {burn_err}")
                progress_state["status"] = f"Subtitle video problem: {burn_err}"
                subs_path = ""

        # big lossless intermediates are no longer needed (MP3 copies are kept)
        for tmp in (os.path.join(speech_dir, "audio_full_mix.wav"),
                    os.path.splitext(full_mp3)[0] + ".wav" if full_mp3 else ""):
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

        # job.json: everything needed to rebuild this video later (e.g. with your own recorded voice)
        if do_tts:
            _save_job_settings(job_dir, {
                "video": os.path.abspath(file_path), "video_duration": video_duration, "voice_key": voice_key,
                "dubbing_mode": dubbing_mode, "bgm_volume": bgm_volume, "voice_volume": voice_volume,
                "burn_subs": burn_subs, "srt": os.path.abspath(target_srt) if burn_subs else "",
                "sub_color": sub_color, "sub_size": sub_size, "sub_box": sub_box, "sub_pos": sub_pos,
                "logo_path": logo_path, "logo_pos": logo_pos, "logo_size": logo_size, "logo_opacity": logo_opacity,
                "aspect_ratio": aspect_ratio, "hide_box": hide_box, "blend_scene": blend_scene, "name": folder_label, "rate": rate,
            })

        # info.txt summary
        info_path = os.path.join(job_dir, "info.txt")
        with open(info_path, "w", encoding="utf-8") as f:
            f.write(f"Job name        : {folder_label}\n")
            f.write(f"Folder          : {job_folder_name}\n")
            f.write(f"Created         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Video file      : {os.path.basename(file_path)}\n")
            f.write(f"Duration        : {video_duration:.2f}s\n")
            f.write(f"Source Language : {source_lang}\n")
            f.write(f"ASR Engine      : {engine}\n")
            f.write(f"Translate Engine: {translate_engine}\n")
            f.write(f"Translate Style : {translate_style}\n")
            f.write(f"Lines           : {len(km_segments)} (Khmer script: {n_kh})\n")
            f.write(f"TTS             : {'yes' if do_tts else 'no'}\n")
            if do_tts:
                f.write(f"Voice Profile   : {voice_key}\n")
                f.write(f"Dubbing Mode    : {dubbing_mode} (BGM: {bgm_volume*100:.0f}%, Voice: {voice_volume*100:.0f}%)\n")
                f.write(f"Speech Duration : {actual_dur:.2f}s (matched to video)\n")
            f.write(f"Burn Subtitles  : {'yes' if burn_subs else 'no'} (Color: {sub_color}, Size: {sub_size}px, Box: {sub_box})\n")
            f.write(f"Bilingual Subs  : {'yes' if bilingual_subs else 'no'}\n")
            f.write("\nGenerated Files:\n")
            f.write("  01_source.srt                    — original speech subtitles\n")
            f.write("  02_khmer.srt                     — Khmer translated subtitles\n")
            if bi_srt:
                f.write("  02_bilingual.srt                 — Khmer + original dual subtitles\n")
            if do_tts:
                f.write("  03_speech/full.mp3               — pure Khmer voice track\n")
                if full_mix_mp3 and os.path.isfile(full_mix_mp3):
                    f.write("  03_speech/audio_full_mix.mp3     — full mix (speech + ducked BGM)\n")
                if dubbed_path and os.path.isfile(dubbed_path):
                    f.write("  04_video_khmer_voice.mp4         — dubbed video (audio synced)\n")
            if subs_path and os.path.isfile(subs_path):
                f.write(f"  {os.path.basename(subs_path):<30} — READY (dubbed audio + burned subtitles)\n")

        if subs_path and os.path.isfile(subs_path):
            progress_state["status"] = (
                f"Done! Open: {job_folder_name}/{os.path.basename(subs_path)} "
                f"(dubbed voice + subtitles on video)"
            )
        elif do_tts and dubbed_path and os.path.exists(dubbed_path):
            progress_state["status"] = (
                f"Done! Open: {job_folder_name}/04_video_khmer_voice.mp4 "
                f"(video {video_duration:.1f}s = voice {actual_dur:.1f}s)"
            )
        else:
            progress_state["status"] = f"Done! Folder: {job_folder_name}"

        final_video_file = subs_path if (subs_path and os.path.exists(subs_path)) else (dubbed_path if (dubbed_path and os.path.exists(dubbed_path)) else file_path)
        progress_state["percent"] = 100
        progress_state["result_path"] = result_path
        progress_state["final_video"] = final_video_file
        progress_state["final_srt"] = km_srt if (km_srt and os.path.exists(km_srt)) else ""
        progress_state["final_audio"] = full_mix_mp3 if (full_mix_mp3 and os.path.exists(full_mix_mp3)) else (full_mp3 if (full_mp3 and os.path.exists(full_mp3)) else "")
        progress_state["job_dir"] = result_path
        save_to_history(job_folder_name, result_path, len(km_segments))
        _clear_checkpoint(job_dir)

    except Exception as e:
        progress_state["status"] = f"Error: {str(e)}"
        progress_state["percent"] = 0
    finally:
        # temp speech audio only — an uploaded audio file is the job's input (needed to resume)
        if audio_path and audio_path != file_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass
        progress_state["is_processing"] = False


@app.route('/pipeline', methods=['POST'])
def pipeline():
    """Video (CN/EN/...) → one job folder with source SRT + Khmer SRT + speech + dubbed video."""
    video_files = request.files.getlist('videos')
    if not video_files or not any(f.filename for f in video_files):
        if 'video' in request.files:
            video_files = [request.files['video']]
        else:
            return jsonify({"success": False, "error": "No video uploaded"}), 400

    video_files = [f for f in video_files if f and f.filename]
    if not video_files:
        return jsonify({"success": False, "error": "No valid video file selected"}), 400

    source_lang = request.form.get('sourceLang', 'auto')
    do_tts = request.form.get('doTts', 'false') == 'true'
    voice_key = request.form.get('voice', 'narrator_female')
    rate = request.form.get('rate', '-5%')
    pitch = request.form.get('pitch', 'default')
    custom_output_dir = request.form.get('outputDir', '')
    job_name = request.form.get('jobName', '').strip()
    burn_subs = request.form.get('burnSubs', 'true') == 'true'
    dubbing_mode = request.form.get('dubbingMode', 'duck')
    bgm_volume = float(request.form.get('bgmVolume', 0.25))
    voice_volume = float(request.form.get('voiceVolume', 1.0))
    translate_engine = request.form.get('translateEngine', 'auto')
    translate_style = request.form.get('translateStyle', 'recap')
    sub_color = request.form.get('subColor', 'yellow')
    sub_size = int(request.form.get('subSize', 22))
    sub_box = request.form.get('subBox', 'false') == 'true'
    sub_pos = request.form.get('subPos', 'bottom')
    bilingual_subs = request.form.get('bilingualSubs', 'false') == 'true'
    auto_detect_voice = request.form.get('autoDetectVoice', 'true') == 'true'

    # Watermark Logo settings
    logo_enabled = request.form.get('logoEnabled', 'false') == 'true'
    logo_pos = request.form.get('logoPos', 'top-right')
    logo_size = request.form.get('logoSize', 'medium')
    try:
        logo_opacity = float(request.form.get('logoOpacity', 0.85))
    except (ValueError, TypeError):
        logo_opacity = 0.85

    logo_path = None
    if logo_enabled:
        if 'logo' in request.files and request.files['logo'].filename:
            logo_f = request.files['logo']
            l_ext = os.path.splitext(logo_f.filename)[1].lower()
            if l_ext in ('.png', '.jpg', '.jpeg', '.webp'):
                logo_dest = os.path.abspath(os.path.join(UPLOAD_FOLDER, f"channel_logo{l_ext}"))
                logo_f.save(logo_dest)
                save_config({
                    "logo_path": logo_dest,
                    "logo_enabled": True,
                    "logo_pos": logo_pos,
                    "logo_size": logo_size,
                    "logo_opacity": logo_opacity
                })
                logo_path = logo_dest
        if not logo_path:
            cfg = load_config()
            saved_logo = cfg.get("logo_path", "")
            if saved_logo and os.path.exists(saved_logo):
                logo_path = saved_logo

    order_raw = request.form.get("order", "[]")
    try:
        order = json.loads(order_raw)
        if not isinstance(order, list):
            order = []
    except Exception:
        order = []

    stamp = datetime.now().strftime("%H%M%S")
    saved = {}
    for i, vf in enumerate(video_files):
        raw_name = vf.filename or f"clip_{i}.mp4"
        orig_ext = os.path.splitext(raw_name)[1].lower() or ".mp4"
        if orig_ext in _REJECT_EXT:
            continue
        safe_name = secure_filename(raw_name) or f"clip_{i}.mp4"
        stem, ex = os.path.splitext(safe_name)
        if not ex or ex.lower() in _REJECT_EXT:
            ex = orig_ext if orig_ext in _MEDIA_EXT else ".mp4"
        target_path = os.path.abspath(
            os.path.join(UPLOAD_FOLDER, f"{stem}_{stamp}_{i:02d}{ex}")
        )
        vf.save(target_path)
        saved[raw_name] = target_path
        saved[safe_name] = target_path
        saved[os.path.basename(raw_name)] = target_path

    if not saved:
        return jsonify({"success": False, "error": "No valid video files uploaded"}), 400

    if len(saved) == 1:
        chosen_input = list(saved.values())[0]
        if not job_name:
            job_name = os.path.splitext(os.path.basename(chosen_input))[0]
    else:
        # Multiple clips to auto-merge
        ordered_paths = []
        if order:
            for name in order:
                p = saved.get(name) or saved.get(os.path.basename(name))
                if p and p not in ordered_paths:
                    ordered_paths.append(p)
        for p in saved.values():
            if p not in ordered_paths:
                ordered_paths.append(p)
        chosen_input = ordered_paths
        if not job_name:
            first_name = os.path.splitext(os.path.basename(ordered_paths[0]))[0]
            job_name = f"{first_name}_merged"

    aspect_ratio = request.form.get('aspectRatio', 'original')
    hide_box = request.form.get('hideBox', '')
    review_script = request.form.get('reviewScript', 'false').lower() == 'true'
    cast_mode = request.form.get('castMode', 'two')
    blend_scene = request.form.get('blendScene', 'true') == 'true'

    st = _start_task("pipeline_cn_en_to_khmer_task", dict(
        file_path=chosen_input, custom_output_dir=custom_output_dir, source_lang=source_lang, do_tts=do_tts,
        voice_key=voice_key, rate=rate, pitch=pitch, job_name=job_name, burn_subs=burn_subs,
        dubbing_mode=dubbing_mode, bgm_volume=bgm_volume, voice_volume=voice_volume,
        translate_engine=translate_engine, translate_style=translate_style,
        sub_color=sub_color, sub_size=sub_size, sub_box=sub_box, sub_pos=sub_pos, bilingual_subs=bilingual_subs,
        auto_detect_voice=auto_detect_voice,
        logo_path=logo_path, logo_pos=logo_pos, logo_size=logo_size, logo_opacity=logo_opacity,
        aspect_ratio=aspect_ratio, hide_box=hide_box, review_script=review_script, cast_mode=cast_mode, blend_scene=blend_scene,
    ), name=job_name, kind="dub")
    return jsonify({
        "success": True,
        "task_id": st["id"],
        "message": f"Pipeline started ({len(saved)} clip{'s' if len(saved) > 1 else ''})"
    })


@app.route('/pipeline-from-url', methods=['POST'])
def pipeline_from_url():
    """Download video from YouTube or internet URL and immediately start AI dubbing."""
    data = request.json or {}
    url = (data.get('url') or '').strip()
    if not url:
        return jsonify({"success": False, "error": "Please provide a valid YouTube or video link"}), 400

    source_lang = data.get('sourceLang', 'auto')
    translate_engine = data.get('translateEngine', 'auto')
    translate_style = data.get('translateStyle', 'recap')
    dubbing_mode = data.get('dubbingMode', 'duck')
    voice_key = data.get('voice', 'narrator_female')
    job_name = (data.get('jobName') or '').strip()
    burn_subs = data.get('burnSubs', True)
    sub_box = data.get('subBox', False)
    bilingual_subs = data.get('bilingualSubs', False)
    auto_detect_voice = data.get('autoDetectVoice', True)
    sub_color = data.get('subColor', 'yellow')
    sub_size = int(data.get('subSize', 22))
    bgm_volume = float(data.get('bgmVolume', 0.25))
    voice_volume = float(data.get('voiceVolume', 1.0))

    aspect_ratio = data.get('aspectRatio', 'original')
    hide_box = data.get('hideBox', '')
    review_script = bool(data.get('reviewScript', False))

    logo_enabled = bool(data.get('logoEnabled', False))
    logo_pos = data.get('logoPos', 'top-right')
    logo_size = data.get('logoSize', 'medium')
    logo_opacity = float(data.get('logoOpacity', 0.85))
    logo_path = None
    if logo_enabled:
        cfg = load_config()
        saved_logo = cfg.get("logo_path", "")
        if saved_logo and os.path.exists(saved_logo):
            logo_path = saved_logo

    st = _start_task("url_pipeline_task", dict(
        url=url, source_lang=source_lang, do_tts=True,
        voice_key=voice_key, rate=data.get("rate", "-5%"), pitch="default",
        job_name=job_name, burn_subs=burn_subs,
        dubbing_mode=dubbing_mode, bgm_volume=bgm_volume, voice_volume=voice_volume,
        translate_engine=translate_engine, translate_style=translate_style,
        sub_color=sub_color, sub_size=sub_size, sub_box=sub_box,
        sub_pos=data.get('subPos', 'bottom'), bilingual_subs=bilingual_subs,
        auto_detect_voice=auto_detect_voice,
        logo_path=logo_path, logo_pos=logo_pos, logo_size=logo_size, logo_opacity=logo_opacity,
        aspect_ratio=aspect_ratio, hide_box=hide_box, review_script=review_script,
        cast_mode=data.get("castMode", "two"),
        blend_scene=bool(data.get("blendScene", True)),
    ), name=job_name or url[:60], kind="dub")
    return jsonify({"success": True, "task_id": st["id"], "message": "Downloading video and starting AI Dubbing..."})


def url_pipeline_task(url, downloaded_file=None, **opts):
    """Download a video link with yt-dlp, then dub it. On resume the earlier download is reused."""
    progress_state = _task_state()
    try:
        dl_file = downloaded_file if (downloaded_file and os.path.isfile(downloaded_file)) else ""
        if not dl_file:
            progress_state["percent"] = 5
            progress_state["status"] = "Downloading video from URL with yt-dlp..."
            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': os.path.join(UPLOAD_FOLDER, '%(id)s_%(title)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
                'nocheckcertificate': True,
                'merge_output_format': 'mp4',
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                dl_file = ydl.prepare_filename(info)
                if not os.path.exists(dl_file):
                    stem = os.path.splitext(dl_file)[0]
                    for ex in ('.mp4', '.mkv', '.webm'):
                        if os.path.exists(stem + ex):
                            dl_file = stem + ex
                            break
            if not os.path.exists(dl_file):
                raise RuntimeError("Failed to locate downloaded video")
            _set_task_param("downloaded_file", dl_file)
            if not opts.get("job_name"):
                opts["job_name"] = secure_filename(info.get('title', 'youtube_video'))[:30]
                _set_task_param("job_name", opts["job_name"])
            if isinstance(progress_state, TaskState):
                progress_state["name"] = opts["job_name"]
    except Exception as e:
        progress_state["status"] = f"Error: {str(e)}"
        progress_state["percent"] = 0
        progress_state["is_processing"] = False
        return
    pipeline_cn_en_to_khmer_task(dl_file, "", **opts)


@app.route('/api/pipeline-review-confirm', methods=['POST'])
def pipeline_review_confirm():
    """Receive user script review corrections and resume paused pipeline."""
    data = request.json or {}
    job_id = data.get("job_id", "").strip()
    segments = data.get("segments", [])
    if job_id and job_id in _ACTIVE_REVIEW_EVENTS:
        _REVIEW_EDITS[job_id] = segments
        _ACTIVE_REVIEW_EVENTS[job_id].set()
        return jsonify({"success": True, "message": "Review confirmed, pipeline resuming"})
    elif len(_ACTIVE_REVIEW_EVENTS) == 1:
        k = list(_ACTIVE_REVIEW_EVENTS.keys())[0]
        _REVIEW_EDITS[k] = segments
        _ACTIVE_REVIEW_EVENTS[k].set()
        return jsonify({"success": True, "message": "Review confirmed, pipeline resuming"})
    return jsonify({"success": False, "error": "Review event expired or not found"}), 404


register_task_runner(pipeline_cn_en_to_khmer_task, resumable=True)
register_task_runner(url_pipeline_task, resumable=True)
