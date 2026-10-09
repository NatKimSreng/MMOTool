"""
The other jobs: subtitles only, cut video, SRT → speech, download, Studio export, batch merge.
Their routes.
"""
import os
import json
import asyncio
import edge_tts
from flask import request, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
import yt_dlp
from .core import (
    DEFAULT_OUTPUT_FOLDER,
    DOWNLOAD_FOLDER,
    UPLOAD_FOLDER,
    _REJECT_EXT,
    _safe_upload_name,
    _strip_foreign_script,
    app,
    format_timestamp,
    load_config,
    save_to_history,
)
from .tasks import (
    _start_task,
    _task_state,
    register_task_runner,
    task_warn,
)
from .media import (
    _burn_subtitles,
    _expand_slots_into_gaps,
    _get_media_duration,
    _mux_video_with_audio,
    _normalize_segment_timings,
    _write_bilingual_srt,
    _write_srt,
    parse_srt_blocks,
)
from .asr import (
    _extract_audio_wav,
    _transcribe_faster_whisper,
    _transcribe_groq,
    _transcribe_kiri,
    _transcribe_openai_whisper,
)
from .translate import (
    _translate_to_khmer,
)
from .voices import (
    _build_timed_speech_track,
    _estimate_tts_rate,
    _synthesize_text_to_file,
    get_voice_settings,
)
from .characters import (
    _detect_speakers_task,
)
from .pipeline import (
    _safe_job_folder_name,
)


def srt_generation_task(file_path, custom_output_dir, model_choice="auto"):
    """
    Generate Khmer SRT — quality ranking:
      1. kiri       — Kiri API (Khmer-first cloud) BEST quality cloud
      2. khmer-ft   — PhanithLIM small Khmer local BEST free quality
      3. khmer-large— Tnaot large-v3 Khmer local (heavier, better)
      4. khmer-tiny — fast local, lower quality
      5. groq       — fast but BAD on Khmer (stock Whisper)
      6. medium     — stock Whisper, BAD on Khmer
    """
    progress_state = _task_state()
    audio_path = None
    try:
        progress_state["is_processing"] = True
        progress_state["status"] = "កំពុងត្រៀម..."
        progress_state["percent"] = 5

        cfg = load_config()
        if model_choice == "auto":
            # Prefer quality over speed for Khmer
            if (cfg.get("KIRI_API_KEY") or "").strip():
                model_choice = "kiri"
            else:
                model_choice = "khmer-ft"

        audio_path = _extract_audio_wav(file_path, progress_state)
        segments = []
        used_engine = ""

        if model_choice == "kiri":
            segments, used_engine = _transcribe_kiri(audio_path, progress_state)
        elif model_choice == "khmer-large":
            try:
                segments, used_engine = _transcribe_faster_whisper(
                    audio_path,
                    "Tnaot/whisper-large-v3-khmer-ct2",
                    progress_state,
                    "khmer-large",
                )
            except Exception as e:
                task_warn(f"Khmer large model failed: {e}")
        elif model_choice == "khmer-tiny":
            try:
                segments, used_engine = _transcribe_faster_whisper(
                    audio_path,
                    "PhanithLIM/whisper-tiny-khmer-ct2",
                    progress_state,
                    "khmer-tiny",
                )
            except Exception as e:
                task_warn(f"Khmer tiny model failed: {e}")
        elif model_choice in ("khmer-ft", "auto"):
            try:
                segments, used_engine = _transcribe_faster_whisper(
                    audio_path,
                    "PhanithLIM/whisper-small-khmer-ct2",
                    progress_state,
                    "khmer-ft",
                )
            except Exception as e:
                task_warn(f"Khmer model failed: {e} — trying the next engine")
        elif model_choice == "groq":
            segments, used_engine = _transcribe_groq(audio_path, progress_state)
        elif model_choice == "medium":
            segments, used_engine = _transcribe_openai_whisper(audio_path, progress_state)

        # Quality-first fallbacks (avoid stock Whisper / Groq if possible)
        if not segments and model_choice != "khmer-ft":
            try:
                segments, used_engine = _transcribe_faster_whisper(
                    audio_path,
                    "PhanithLIM/whisper-small-khmer-ct2",
                    progress_state,
                    "khmer-ft-fallback",
                )
            except Exception as e:
                task_warn(f"Khmer model failed: {e}")
        if not segments and model_choice != "kiri" and (cfg.get("KIRI_API_KEY") or "").strip():
            try:
                segments, used_engine = _transcribe_kiri(audio_path, progress_state)
            except Exception as e:
                task_warn(f"Kiri failed: {e}")
        if not segments and model_choice != "groq" and (cfg.get("GROQ_API_KEY") or "").strip():
            try:
                segments, used_engine = _transcribe_groq(audio_path, progress_state)
            except Exception as e:
                task_warn(f"Groq failed: {e}")
        if not segments and model_choice != "medium":
            try:
                segments, used_engine = _transcribe_openai_whisper(audio_path, progress_state)
            except Exception as e:
                task_warn(f"Whisper medium failed: {e}")

        if not segments:
            raise ValueError(
                "No speech / empty SRT. Install: pip install faster-whisper "
                "then use engine = Local Khmer fine-tuned (NOT Groq). "
                "Or set KIRI_API_KEY for Khmer-first cloud STT."
            )

        progress_state["status"] = "កំពុងបង្កើតឯកសារ SRT..."
        progress_state["percent"] = 90

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        root_output = custom_output_dir if custom_output_dir else DEFAULT_OUTPUT_FOLDER
        os.makedirs(root_output, exist_ok=True)
        output_path = os.path.join(root_output, f"{base_name}.srt")

        _write_srt(output_path, segments)

        progress_state["percent"] = 100
        progress_state["status"] = f"Success! ({used_engine})"
        progress_state["result_path"] = os.path.abspath(output_path)

    except Exception as e:
        progress_state["status"] = f"Error: {str(e)}"
        progress_state["percent"] = 0
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass
        progress_state["is_processing"] = False

def cut_video_task(file_path, num_segments, quality, keep_audio, custom_output_dir):
    """Split video into N equal parts via ffmpeg (stream-copy first). Safe for 1–2h files."""
    progress_state = _task_state()
    import subprocess

    try:
        progress_state["is_processing"] = True
        progress_state["status"] = "Initializing..."
        progress_state["percent"] = 5

        duration = _get_media_duration(file_path)
        if duration <= 0:
            raise ValueError("Could not read video duration")

        segment_duration = duration / max(1, int(num_segments))
        filename = os.path.basename(file_path)
        name_ext = os.path.splitext(filename)
        base_name = name_ext[0]
        extension = name_ext[1] if name_ext[1] else ".mp4"

        root_output = custom_output_dir if custom_output_dir else DEFAULT_OUTPUT_FOLDER
        output_subfolder = os.path.join(root_output, base_name)
        os.makedirs(output_subfolder, exist_ok=True)

        # quality → CRF (lower = better). Stream-copy ignores this unless we re-encode.
        crf_map = {"high": "18", "medium": "23", "low": "28"}
        crf = crf_map.get(quality, "23")

        def run_ff(cmd, timeout=600):
            return subprocess.run(cmd, capture_output=True, timeout=timeout)

        for i in range(int(num_segments)):
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, duration)
            progress_state["status"] = f"Processing Part {i+1}/{num_segments}..."
            progress_state["percent"] = 5 + int((i / max(num_segments, 1)) * 90)

            output_filename = f"part_{i+1}_{base_name}{extension}"
            output_path = os.path.join(output_subfolder, output_filename)

            # Prefer stream copy (fast, no quality loss)
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{start_time:.3f}",
                "-to", f"{end_time:.3f}",
                "-i", file_path,
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                "-movflags", "+faststart",
            ]
            if not keep_audio:
                cmd.extend(["-an"])
            cmd.append(output_path)

            part_timeout = max(120, int((end_time - start_time) * 2) + 60)
            r = run_ff(cmd, timeout=part_timeout)
            bad = (
                r.returncode != 0
                or not os.path.exists(output_path)
                or os.path.getsize(output_path) < 500
            )
            if bad:
                # Fallback: re-encode this part
                try:
                    if os.path.exists(output_path):
                        os.remove(output_path)
                except Exception:
                    pass
                cmd = [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", f"{start_time:.3f}",
                    "-to", f"{end_time:.3f}",
                    "-i", file_path,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", crf,
                    "-movflags", "+faststart",
                ]
                if keep_audio:
                    cmd.extend(["-c:a", "aac", "-b:a", "192k"])
                else:
                    cmd.append("-an")
                cmd.append(output_path)
                r = run_ff(cmd, timeout=max(300, int((end_time - start_time) * 4) + 120))
                if r.returncode != 0 or not os.path.exists(output_path) or os.path.getsize(output_path) < 500:
                    err = (r.stderr or b"").decode("utf-8", errors="ignore")[-500:]
                    raise ValueError(f"Failed part {i+1}: {err}")

        final_path = os.path.abspath(output_subfolder)
        progress_state["percent"] = 100
        progress_state["status"] = "Success!"
        progress_state["result_path"] = final_path

        save_to_history(filename, final_path, num_segments)

    except Exception as e:
        progress_state["status"] = f"Error: {str(e)}"
        progress_state["percent"] = 0
        print(f"[cut_video] {e}")
    finally:
        progress_state["is_processing"] = False


def srt_to_speech_task(srt_path, custom_output_dir, voice_key="narrator_female", rate="-8%", merge=True):
    """SRT → Khmer speech using Edge TTS (best free Khmer quality)."""
    progress_state = _task_state()
    try:
        progress_state["is_processing"] = True
        progress_state["status"] = "កំពុងអានឯកសារ SRT..."
        progress_state["percent"] = 8

        blocks = parse_srt_blocks(srt_path)
        if not blocks:
            raise ValueError("No text found in SRT file.")

        base_name = os.path.splitext(os.path.basename(srt_path))[0]
        root_output = custom_output_dir if custom_output_dir else DEFAULT_OUTPUT_FOLDER
        output_folder = os.path.join(root_output, f"{base_name}_speech")
        os.makedirs(output_folder, exist_ok=True)

        voice, voice_rate, voice_pitch = get_voice_settings(voice_key, custom_rate=rate)
        total = len(blocks)
        segment_files = []

        async def generate_audio():
            for i, blk in enumerate(blocks, start=1):
                progress_state["status"] = f"កំពុងបង្កើតសំឡេងភាគ {i}/{total}..."
                progress_state["percent"] = 10 + int((i / total) * (70 if merge else 85))
                text = blk["text"]
                out = os.path.join(output_folder, f"segment_{i:04d}.mp3")
                # rate slightly slower by default = clearer; pitch tuned per voice
                communicate = edge_tts.Communicate(text, voice, rate=voice_rate, pitch=voice_pitch)
                await communicate.save(out)
                segment_files.append(out)

        asyncio.run(generate_audio())

        # Merge into one track matching SRT timeline length
        if merge and segment_files:
            progress_state["status"] = "កំពុងរួមបញ្ចូលសំឡេង (match SRT time)..."
            progress_state["percent"] = 85
            try:
                last_end = max(float(b.get("end", 0)) for b in blocks) if blocks else 0
                total_dur = max(last_end, max(float(b.get("start", 0)) for b in blocks) + 1)
                merged_path = os.path.join(output_folder, f"{base_name}_full.mp3")
                path, actual = _build_timed_speech_track(
                    blocks, segment_files, total_dur, merged_path
                )
                progress_state["status"] = f"ជោគជ័យ! Speech {actual:.1f}s matches SRT"
            except Exception as merge_err:
                print(f"[merge skip]: {merge_err}")

        progress_state["percent"] = 100
        progress_state["status"] = "ជោគជ័យ!"
        progress_state["result_path"] = os.path.abspath(output_folder)

    except Exception as e:
        progress_state["status"] = f"Error: {str(e)}"
        progress_state["percent"] = 0
    finally:
        progress_state["is_processing"] = False

def _is_bilibili_url(url):
    u = (url or "").lower()
    return "bilibili.com" in u or "b23.tv" in u or "bilibili.tv" in u


def _ydl_opts(url, quality="best"):
    """yt-dlp options shared by the episode list and the download."""
    is_bili = _is_bilibili_url(url)
    cap = f"[height<={int(quality)}]" if str(quality).isdigit() else ""
    opts = {
        # Bilibili serves separate video/audio streams; take the best of each and merge
        'format': (f'bestvideo*{cap}+bestaudio/best{cap}/best' if is_bili
                   else f'bestvideo{cap}[ext=mp4]+bestaudio[ext=m4a]/best{cap}[ext=mp4]/best{cap}/best'),
        'quiet': True,
        'no_warnings': True,
        'nocheckcertificate': True,
        'merge_output_format': 'mp4',
        'retries': 10,
        'fragment_retries': 10,
        'concurrent_fragment_downloads': 4,
        'windowsfilenames': True,
    }
    if is_bili:
        # without a browser User-Agent + Referer Bilibili answers HTTP 412
        opts['http_headers'] = {
            'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                           '(KHTML, like Gecko) Chrome/130.0 Safari/537.36'),
            'Referer': 'https://www.bilibili.com/',
            'Origin': 'https://www.bilibili.com',
        }
    # cookies.txt from your own logged-in account unlocks 1080p and what your account may watch
    cookies = (load_config().get("download_cookies_file") or "").strip().strip('"')
    if cookies and os.path.isfile(cookies):
        opts['cookiefile'] = cookies
    return opts


def _explain_download_error(url, e):
    msg = str(e)
    if _is_bilibili_url(url):
        low = msg.lower()
        if "412" in msg:
            msg += " — Bilibili blocked the request. Wait a few minutes, or add a cookies.txt in Settings."
        elif "geo" in low or "region" in low or "area" in low:
            msg += " — this video is region-locked on Bilibili."
        elif "premium" in low or "vip" in low or "大会员" in msg or "login" in low:
            msg += " — this video needs a Bilibili login/VIP. Add a cookies.txt from your account in Settings."
    return msg


def _bili_season_items(url):
    """Bilibili movie / series pages: the real episode names (yt-dlp's list only says 'Part N')."""
    import re
    import requests
    m = re.search(r"/bangumi/play/(ss|ep)(\d+)", url or "")
    if not m:
        return None
    q = ("season_id=" if m.group(1) == "ss" else "ep_id=") + m.group(2)
    opts = _ydl_opts(url)
    headers = opts.get('http_headers', {})
    cookies = None
    if opts.get('cookiefile'):
        import http.cookiejar
        cookies = http.cookiejar.MozillaCookieJar(opts['cookiefile'])
        cookies.load(ignore_discard=True, ignore_expires=True)
    data = requests.get(f"https://api.bilibili.com/pgc/view/web/season?{q}",
                        headers=headers, cookies=cookies, timeout=20).json()
    res = data.get("result") or {}
    eps = res.get("episodes") or []
    if data.get("code") != 0 or not eps:
        return None
    items = []
    for i, e in enumerate(eps, start=1):
        title = e.get("show_title") or " ".join(x for x in (e.get("title"), e.get("long_title")) if x)
        items.append({
            "index": i,
            "title": title or f"Episode {i}",
            "url": e.get("link") or f"https://www.bilibili.com/bangumi/play/ep{e.get('id')}",
            "duration": (e.get("duration") or 0) / 1000 or None,
            "thumbnail": e.get("cover"),
            "badge": e.get("badge") or "",   # e.g. 会员 = needs VIP on your account
        })
    return {"title": res.get("title") or "Series", "items": items, "heights": []}


def _list_video_items(url):
    """Every episode / part behind a link (a movie, a series, a multi-part video, a playlist)."""
    try:
        found = _bili_season_items(url)
        if found:
            return found
    except Exception as e:
        print(f"[bilibili season list]: {e}")
    opts = _ydl_opts(url)
    opts['extract_flat'] = 'in_playlist'
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = [e for e in (info.get('entries') or []) if e] if info.get('_type') == 'playlist' else []
    if not entries:
        heights = sorted({f.get('height') for f in info.get('formats') or [] if f.get('height')}, reverse=True)
        return {"title": info.get('title') or 'Video', "items": [{
            "index": 1, "title": info.get('title') or 'Video', "url": info.get('webpage_url') or url,
            "duration": info.get('duration'), "thumbnail": info.get('thumbnail'),
        }], "heights": heights}
    items = []
    for i, e in enumerate(entries, start=1):
        items.append({
            "index": i,
            "title": e.get('title') or f"Part {i}",
            "url": e.get('url') or e.get('webpage_url') or url,
            "duration": e.get('duration'),
            "thumbnail": (e.get('thumbnails') or [{}])[-1].get('url') or e.get('thumbnail'),
        })
    return {"title": info.get('title') or 'Playlist', "items": items, "heights": []}


_PREVIEW_STREAMS = {}   # token -> (stream url, request headers), for the on-page player


def _pick_preview_formats(formats):
    """A small browser-playable stream: one file with sound if the site has it,
    else H.264 video ≤480p plus a separate audio track (Bilibili)."""
    def h(f):
        return f.get('height') or 0
    playable = [f for f in formats if f.get('url') and (f.get('protocol') or 'https').startswith('http')]
    both = [f for f in playable if f.get('vcodec') not in (None, 'none') and f.get('acodec') not in (None, 'none')
            and f.get('ext') in ('mp4', 'webm')]
    if both:
        small = [f for f in both if h(f) <= 480]
        return (max(small, key=h) if small else min(both, key=h)), None
    video = [f for f in playable if f.get('vcodec', 'none') != 'none' and f.get('acodec') in (None, 'none')]
    audio = [f for f in playable if f.get('acodec', 'none') != 'none' and f.get('vcodec') in (None, 'none')]
    avc = [f for f in video if str(f.get('vcodec', '')).startswith('avc')] or video
    if not avc:
        raise ValueError("No playable preview stream for this video")
    small = [f for f in avc if h(f) <= 480]
    v = max(small, key=h) if small else min(avc, key=h)
    a = min(audio, key=lambda f: f.get('abr') or f.get('tbr') or 0) if audio else None
    return v, a


def _preview_token(fmt):
    import secrets
    tok = secrets.token_urlsafe(12)
    if len(_PREVIEW_STREAMS) > 200:
        _PREVIEW_STREAMS.clear()
    _PREVIEW_STREAMS[tok] = (fmt['url'], dict(fmt.get('http_headers') or {}))
    return f"/download-preview/stream/{tok}"


@app.route('/download-preview', methods=['POST'])
def download_preview():
    """Watch an episode on the page before downloading it."""
    url = ((request.json or {}).get('url') or '').strip()
    if not url:
        return jsonify({"success": False, "error": "URL is required"}), 400
    try:
        opts = _ydl_opts(url)
        opts['noplaylist'] = True
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
        if info.get('_type') == 'playlist' and info.get('entries'):
            info = next(e for e in info['entries'] if e)
        v, a = _pick_preview_formats(info.get('formats') or [info])
        return jsonify({
            "success": True,
            "title": info.get('title') or 'Video',
            "duration": info.get('duration'),
            "video": _preview_token(v),
            "audio": _preview_token(a) if a else None,
        })
    except Exception as e:
        return jsonify({"success": False, "error": _explain_download_error(url, e)}), 500


@app.route('/download-preview/stream/<tok>')
def download_preview_stream(tok):
    """Relay the stream with the site's Referer/User-Agent (the browser can't send those itself)."""
    from flask import Response, stream_with_context
    import requests
    found = _PREVIEW_STREAMS.get(tok)
    if not found:
        return jsonify({"success": False, "error": "Preview expired — press ▶ again"}), 404
    src, headers = found
    if request.headers.get('Range'):
        headers = {**headers, 'Range': request.headers['Range']}
    up = requests.get(src, headers=headers, stream=True, timeout=30)
    out = {k: up.headers[k] for k in ('Content-Type', 'Content-Length', 'Content-Range') if k in up.headers}
    out['Accept-Ranges'] = 'bytes'
    if out.get('Content-Type', '').startswith(('application/octet-stream', 'video/x-m4s')):
        out['Content-Type'] = 'video/mp4'

    def _body():
        try:
            for chunk in up.iter_content(256 * 1024):
                yield chunk
        finally:
            up.close()
    return Response(stream_with_context(_body()), status=up.status_code, headers=out)


def download_yt_task(url, custom_output_dir, items=None, quality="best"):
    """Download one link, or only the chosen episodes/parts (items = their URLs)."""
    progress_state = _task_state()
    targets = [u for u in (items or []) if u] or [url]
    saved, failed = [], []
    try:
        progress_state["is_processing"] = True
        progress_state["status"] = "Fetching video info..."
        progress_state["percent"] = 5

        root_output = custom_output_dir if custom_output_dir else DOWNLOAD_FOLDER
        os.makedirs(root_output, exist_ok=True)
        n = len(targets)

        for k, target in enumerate(targets):
            label = f"[{k + 1}/{n}] " if n > 1 else ""

            def _hook(d, k=k, label=label):
                if d.get("status") == "downloading":
                    total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                    part = d.get("downloaded_bytes", 0) / total if total else 0
                    progress_state["percent"] = 5 + int(90 * (k + part) / n)
                    progress_state["status"] = f"{label}Downloading… {d.get('_percent_str', '').strip()}"

            opts = _ydl_opts(target, quality)
            opts['outtmpl'] = os.path.join(root_output, '%(title).150B [%(id)s].%(ext)s')
            opts['noplaylist'] = True    # a ?p=3 link downloads part 3 only, not the whole series
            opts['progress_hooks'] = [_hook]
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(target, download=True)
                    if info.get('_type') == 'playlist' and info.get('entries'):
                        info = next(e for e in info['entries'] if e)
                    done = info.get('requested_downloads') or []
                    filename = (done[0].get('filepath') if done else None) or ydl.prepare_filename(info)
                saved.append(os.path.abspath(filename))
                progress_state["status"] = f"{label}Downloaded: {info.get('title', 'Video')} ({info.get('height') or '?'}p)"
            except Exception as e:
                if n == 1:
                    raise
                print(f"[download {target}]: {e}")
                failed.append(_explain_download_error(target, e))

        if not saved:
            raise ValueError(failed[0] if failed else "Nothing was downloaded")
        progress_state["percent"] = 100
        progress_state["result_path"] = saved[0] if len(saved) == 1 else os.path.abspath(root_output)
        if n > 1:
            progress_state["status"] = f"Downloaded {len(saved)}/{n} videos" + (
                f" — {len(failed)} failed: {failed[0]}" if failed else "")

    except Exception as e:
        progress_state["status"] = f"Error: {_explain_download_error(url, e)}"
        progress_state["percent"] = 0
    finally:
        progress_state["is_processing"] = False

@app.route('/cut', methods=['POST'])
def cut():
    if 'video' not in request.files:
        return jsonify({"success": False, "error": "No video uploaded"}), 400

    video_file = request.files['video']
    num_segments = int(request.form.get('segments', 6))
    quality = request.form.get('quality', 'medium')
    keep_audio = request.form.get('keepAudio') == 'true'
    custom_output_dir = request.form.get('outputDir', '')

    if video_file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400

    filename = secure_filename(video_file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    video_file.save(file_path)

    st = _start_task("cut_video_task", dict(
        file_path=file_path, num_segments=num_segments, quality=quality, keep_audio=keep_audio,
        custom_output_dir=custom_output_dir), name=f"Cut: {filename}", kind="cut")

    return jsonify({"success": True, "message": "Processing started", "task_id": st["id"]})

@app.route('/srt-to-speech', methods=['POST'])
def srt_to_speech():
    if 'srt_file' not in request.files:
        return jsonify({"success": False, "error": "No SRT file uploaded"}), 400

    srt_file = request.files['srt_file']
    custom_output_dir = request.form.get('outputDir', '')
    voice_key = request.form.get('voice', 'female')  # female | male
    rate = request.form.get('rate', '-8%')          # default slightly slower = clearer / more natural
    merge = request.form.get('merge', 'true') == 'true'

    if srt_file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400

    filename = secure_filename(srt_file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    srt_file.save(file_path)

    st = _start_task("srt_to_speech_task", dict(
        srt_path=file_path, custom_output_dir=custom_output_dir, voice_key=voice_key, rate=rate, merge=merge),
        name=f"Voice: {filename}", kind="speech")

    return jsonify({"success": True, "message": "Speech generation started", "task_id": st["id"]})

def studio_export_task(file_path, clips_data, custom_output_dir, dub_options=None):
    """
    Cut and concatenate clips in order (CapCut-style export) with optional
    Khmer Character Voiceover dubbing and stylized subtitle burn-in.
    Uses pure ffmpeg (stream-copy first, re-encode fallback) so 1–2h
    videos stay fast and do not OOM like MoviePy.
    """
    progress_state = _task_state()
    import subprocess
    import shutil
    import tempfile

    temp_dir = None
    try:
        progress_state["is_processing"] = True
        progress_state["status"] = "Preparing cuts..."
        progress_state["percent"] = 5

        if not os.path.exists(file_path):
            raise ValueError(f"Source missing: {file_path}")

        total = len(clips_data)
        if total == 0:
            raise ValueError("No clips to export")

        # Validate & normalize clip times
        segments = []
        for i, c in enumerate(clips_data):
            start = float(c.get("sourceStart", 0))
            end = float(c.get("sourceEnd", 0))
            if end <= start:
                raise ValueError(f"Clip {i+1}: end ({end}) must be > start ({start})")
            if start < 0:
                start = 0.0
            segments.append((start, end))

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        root_output = custom_output_dir if custom_output_dir else DEFAULT_OUTPUT_FOLDER
        os.makedirs(root_output, exist_ok=True)
        output_path = os.path.join(root_output, f"{base_name}_studio_export.mp4")

        # Work in a temp folder so we can clean up partial cuts
        temp_dir = tempfile.mkdtemp(prefix="studio_export_")
        part_paths = []

        # Estimate timeout from total source span (min 10 min, scale with hours)
        total_span = sum(e - s for s, e in segments)
        base_timeout = max(600, int(total_span * 3) + 120)

        def run_ff(cmd, timeout=base_timeout):
            return subprocess.run(
                cmd, capture_output=True, timeout=timeout
            )

        # ── 1) Extract each segment with stream copy (very fast) ──
        for i, (start, end) in enumerate(segments):
            progress_state["status"] = f"Cutting clip {i+1}/{total} (fast copy)..."
            progress_state["percent"] = 5 + int((i / max(total, 1)) * 55)

            part = os.path.join(temp_dir, f"part_{i:04d}.mp4")
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{start:.3f}",
                "-to", f"{end:.3f}",
                "-i", file_path,
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                "-movflags", "+faststart",
                part,
            ]
            r = run_ff(cmd, timeout=max(120, int((end - start) * 2) + 60))
            if r.returncode != 0 or not os.path.exists(part) or os.path.getsize(part) < 500:
                err = (r.stderr or b"").decode("utf-8", errors="ignore")[-300:]
                print(f"[studio] copy failed part {i+1}, re-encoding: {err}")
                progress_state["status"] = f"Re-encoding clip {i+1}/{total}..."
                cmd = [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", f"{start:.3f}",
                    "-to", f"{end:.3f}",
                    "-i", file_path,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart",
                    part,
                ]
                r = run_ff(cmd, timeout=max(300, int((end - start) * 4) + 120))
                if r.returncode != 0 or not os.path.exists(part) or os.path.getsize(part) < 500:
                    err2 = (r.stderr or b"").decode("utf-8", errors="ignore")[-500:]
                    raise ValueError(f"Failed to cut clip {i+1}: {err2}")
            part_paths.append(part)

        # ── 2) Concatenate ──
        progress_state["status"] = f"Joining {total} clips..."
        progress_state["percent"] = 65

        joined_export = os.path.join(temp_dir, "joined_base.mp4")
        if total == 1:
            shutil.copy(part_paths[0], joined_export)
        else:
            list_file = os.path.join(temp_dir, "concat_list.txt")
            with open(list_file, "w", encoding="utf-8") as f:
                for pth in part_paths:
                    abs_pth = os.path.abspath(pth).replace("\\", "/").replace("'", "'\\''")
                    f.write(f"file '{abs_pth}'\n")

            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-i", list_file,
                "-c", "copy",
                "-movflags", "+faststart",
                joined_export,
            ]
            r = run_ff(cmd, timeout=base_timeout)
            bad = (
                r.returncode != 0
                or not os.path.exists(joined_export)
                or os.path.getsize(joined_export) < 1000
            )
            if bad:
                err = (r.stderr or b"").decode("utf-8", errors="ignore")[-400:]
                print(f"[studio] concat copy failed, re-encoding join: {err}")
                progress_state["status"] = "Re-encoding final join (one pass)..."
                progress_state["percent"] = 70
                try:
                    if os.path.exists(joined_export):
                        os.remove(joined_export)
                except Exception:
                    pass
                cmd = [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "concat", "-safe", "0",
                    "-i", list_file,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart",
                    joined_export,
                ]
                r = run_ff(cmd, timeout=max(base_timeout, int(total_span * 5) + 180))
                if r.returncode != 0 or not os.path.exists(joined_export) or os.path.getsize(joined_export) < 1000:
                    err2 = (r.stderr or b"").decode("utf-8", errors="ignore")[-600:]
                    raise ValueError(f"ffmpeg join failed: {err2}")

        # ── 3) Optional Dubbing & Subtitle Burn-In ──
        dub_opt = dub_options or {}
        apply_dubbing = bool(dub_opt.get("apply_dubbing", False))
        burn_subs = bool(dub_opt.get("burn_subtitles", False))
        subtitles = dub_opt.get("subtitles", [])

        current_video = joined_export
        speech_audio_path = None

        if (apply_dubbing or burn_subs) and subtitles:
            # Generate speech track if dubbing requested
            if apply_dubbing:
                progress_state["status"] = "Generating Khmer Character Voiceover..."
                progress_state["percent"] = 75

                v_dur = _get_media_duration(joined_export) or 0
                timed_segs = _normalize_segment_timings(subtitles, v_dur)
                timed_segs = _expand_slots_into_gaps(timed_segs, v_dur)
                default_v = dub_opt.get("default_voice", "narrator_female")
                c_rate = dub_opt.get("voice_rate", "-5%")
                c_pitch = dub_opt.get("voice_pitch", "+0Hz")

                speech_parts = []
                async def _gen_speech():
                    for i, blk in enumerate(timed_segs, start=1):
                        txt = blk.get("text", "").strip()
                        if not txt:
                            continue
                        v_key = blk.get("voice") or default_v
                        v_id, v_rate, v_pitch = get_voice_settings(v_key, custom_rate=c_rate, custom_pitch=c_pitch)
                        slot = max(0.35, float(blk["end"]) - float(blk["start"]))
                        seg_rate = _estimate_tts_rate(txt, slot, base_rate=v_rate, natural=True)

                        seg_mp3 = os.path.join(temp_dir, f"seg_{i:04d}.mp3")
                        await edge_tts.Communicate(txt, v_id, rate=seg_rate, pitch=v_pitch).save(seg_mp3)
                        speech_parts.append(seg_mp3)

                asyncio.run(_gen_speech())
                if speech_parts:
                    built_speech = os.path.join(temp_dir, "speech_full.mp3")
                    speech_audio_path, _ = _build_timed_speech_track(
                        timed_segs, speech_parts, v_dur, built_speech
                    )

            # If burning subtitles
            if burn_subs:
                progress_state["status"] = "Burning Khmer stylized subtitles..."
                progress_state["percent"] = 88
                srt_path = os.path.join(temp_dir, "export_subs.srt")
                is_bilingual = bool(dub_opt.get("bilingual", False))
                if is_bilingual:
                    _write_bilingual_srt(srt_path, subtitles, subtitles)
                else:
                    _write_srt(srt_path, [{**s, "text": s.get("text", "")} for s in subtitles])

                sub_col = dub_opt.get("sub_color", "yellow")
                sub_sz = int(dub_opt.get("sub_size", 22))
                sub_bx = bool(dub_opt.get("sub_box", False))
                dub_mode = dub_opt.get("dubbing_mode", "duck")

                _burn_subtitles(
                    current_video,
                    srt_path,
                    output_path,
                    audio_path=speech_audio_path,
                    dubbing_mode=dub_mode,
                    font_size=sub_sz,
                    sub_color=sub_col,
                    sub_box=sub_bx
                )
            elif speech_audio_path:
                progress_state["status"] = "Mixing audio with Smart BGM Ducking..."
                progress_state["percent"] = 88
                dub_mode = dub_opt.get("dubbing_mode", "duck")
                _mux_video_with_audio(current_video, speech_audio_path, output_path, dubbing_mode=dub_mode)
            else:
                shutil.copy(current_video, output_path)
        else:
            shutil.copy(joined_export, output_path)

        progress_state["percent"] = 100
        progress_state["status"] = "Success!"
        progress_state["result_path"] = os.path.abspath(output_path)

        save_to_history(os.path.basename(file_path), output_path, total)

    except Exception as e:
        progress_state["status"] = f"Error: {str(e)}"
        progress_state["percent"] = 0
        print(f"[studio_export] {e}")
    finally:
        progress_state["is_processing"] = False
        if temp_dir and os.path.isdir(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
@app.route('/studio-upload', methods=['POST'])
def studio_upload():
    """
    Upload source video once when user imports in Studio.
    Export then only sends clip JSON + server path (no multi-GB re-upload).
    """
    if 'video' not in request.files:
        return jsonify({"success": False, "error": "No video uploaded"}), 400
    video_file = request.files['video']
    if not video_file or video_file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400

    filename = _safe_upload_name(video_file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        video_file.save(file_path)
    except Exception as e:
        return jsonify({"success": False, "error": f"Save failed: {e}"}), 500

    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    dur = _get_media_duration(file_path)
    return jsonify({
        "success": True,
        "serverPath": os.path.abspath(file_path),
        "filename": filename,
        "sizeMb": round(size_mb, 1),
        "duration": dur,
    })


@app.route('/studio-export', methods=['POST'])
def studio_export():
    clips_json = request.form.get('clips', '[]')
    custom_output_dir = request.form.get('outputDir', '')
    server_path = (request.form.get('serverPath') or '').strip()
    dub_options_json = request.form.get('dubOptions', '')

    try:
        clips_data = json.loads(clips_json)
    except json.JSONDecodeError:
        return jsonify({"success": False, "error": "Invalid clips data"}), 400

    if not clips_data:
        return jsonify({"success": False, "error": "No clips to export"}), 400

    dub_options = None
    if dub_options_json:
        try:
            dub_options = json.loads(dub_options_json)
        except Exception:
            dub_options = None

    file_path = None
    # Prefer path from earlier /studio-upload (fast path for long videos)
    if server_path:
        abs_path = os.path.abspath(server_path)
        # Only allow files under uploads/ for safety
        uploads_abs = os.path.abspath(UPLOAD_FOLDER)
        if abs_path.startswith(uploads_abs + os.sep) and os.path.isfile(abs_path):
            file_path = abs_path
        else:
            return jsonify({
                "success": False,
                "error": "Invalid server path. Re-import the video.",
            }), 400

    if not file_path:
        if 'video' not in request.files:
            return jsonify({
                "success": False,
                "error": "No video. Import the file again (upload may have failed).",
            }), 400
        video_file = request.files['video']
        if not video_file or video_file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        filename = _safe_upload_name(video_file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        video_file.save(file_path)

    st = _start_task("studio_export_task", dict(
        file_path=file_path, clips_data=clips_data, custom_output_dir=custom_output_dir, dub_options=dub_options),
        name=f"Studio export: {os.path.basename(file_path)}", kind="export")

    return jsonify({"success": True, "message": "Export started", "task_id": st["id"]})

@app.route('/get-srt-content')
def get_srt_content():
    path = request.args.get('path')
    if not path or not os.path.exists(path):
        return jsonify({"success": False, "error": "File not found"}), 404
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({"success": True, "content": content})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/save-srt', methods=['POST'])
def save_srt():
    data = request.json
    path = data.get('path')
    content = data.get('content')
    if not path or not content:
        return jsonify({"success": False, "error": "Missing data"}), 400
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write("\n".join(_strip_foreign_script(ln) for ln in content.split("\n")))
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/download', methods=['POST'])
def download():
    url = request.form.get('url')
    custom_output_dir = request.form.get('outputDir', '')

    if not url:
        return jsonify({"success": False, "error": "URL is required"}), 400
    try:
        items = json.loads(request.form.get('items') or '[]')
    except ValueError:
        items = []
    quality = request.form.get('quality', 'best')
    name = f"Download {len(items)} videos: {url[:50]}" if len(items) > 1 else f"Download: {url[:60]}"

    st = _start_task("download_yt_task", dict(url=url, custom_output_dir=custom_output_dir,
                                              items=items, quality=quality),
                     name=name, kind="download")

    return jsonify({"success": True, "message": "Download started", "task_id": st["id"]})


@app.route('/download-info', methods=['POST'])
def download_info():
    """List the episodes / parts behind a link so the user can pick which to download."""
    url = ((request.json or {}).get('url') or '').strip()
    if not url:
        return jsonify({"success": False, "error": "URL is required"}), 400
    try:
        return jsonify({"success": True, **_list_video_items(url)})
    except Exception as e:
        return jsonify({"success": False, "error": _explain_download_error(url, e)}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'video' not in request.files:
        return jsonify({"success": False, "error": "No video uploaded"}), 400

    video_file = request.files['video']
    custom_output_dir = request.form.get('outputDir', '')
    model_choice = request.form.get('model', 'auto')  # auto | groq | khmer-ft | khmer-tiny | medium

    if video_file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400

    ext = os.path.splitext(video_file.filename or "")[1].lower()
    if ext in _REJECT_EXT:
        return jsonify({
            "success": False,
            "error": f"Upload a VIDEO file, not {ext}"
        }), 400

    filename = secure_filename(video_file.filename)
    stem, ex = os.path.splitext(filename)
    file_path = os.path.join(UPLOAD_FOLDER, f"{stem}_{datetime.now().strftime('%H%M%S')}{ex}")
    video_file.save(file_path)

    st = _start_task("srt_generation_task", dict(
        file_path=file_path, custom_output_dir=custom_output_dir, model_choice=model_choice),
        name=f"Subtitles: {filename}", kind="srt")

    return jsonify({"success": True, "message": "Transcription started", "task_id": st["id"]})


@app.route('/studio-auto-translate', methods=['POST'])
def studio_auto_translate():
    """Auto-translate subtitles inside Studio Editor using selected engine & style with optional speaker detection."""
    data = request.json or {}
    subtitles = data.get('subtitles', [])
    engine = data.get('engine', 'auto')
    style = data.get('style', 'recap')
    detect_characters = bool(data.get('detectCharacters', True))

    if not subtitles:
        return jsonify({"success": False, "error": "No subtitles provided"}), 400

    segments = []
    for s in subtitles:
        segments.append({
            "start": float(s.get("start", 0)),
            "end": float(s.get("end", 0)),
            "text": s.get("text", ""),
            "source": s.get("source", s.get("text", "")),
            "id": s.get("id"),
        })

    dummy_state = {"percent": 0, "status": "Translating..."}
    try:
        translated = _translate_to_khmer(segments, dummy_state, engine=engine, style=style)
        
        detected_map = {}
        if detect_characters:
            try:
                char_list = _detect_speakers_task(segments, engine=engine)
                for c in char_list:
                    detected_map[c["id"]] = c
            except Exception as e:
                print(f"[auto_detect in translate warning]: {e}")

        res = []
        for i, t in enumerate(translated):
            orig_sub = subtitles[i] if i < len(subtitles) else {}
            s_id = orig_sub.get("id", i + 1)
            char_info = detected_map.get(s_id, {})

            res.append({
                "id": s_id,
                "start": t["start"],
                "end": t["end"],
                "text": t["text"],
                "source": t.get("source", orig_sub.get("text", "")),
                "voice": char_info.get("voice") or orig_sub.get("voice", "narrator_female"),
                "role": char_info.get("role", "narrator"),
                "role_label": char_info.get("role_label", ""),
                "badge": char_info.get("badge", ""),
            })
        return jsonify({"success": True, "subtitles": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/studio-generate-tts', methods=['POST'])
def studio_generate_tts():
    """Batch generate TTS audio for subtitles in Studio Editor with character voices."""
    data = request.json or {}
    subtitles = data.get('subtitles', [])
    default_voice = data.get('defaultVoice', 'narrator_female')
    rate = data.get('rate', '-5%')
    pitch = data.get('pitch', '+0Hz')
    video_duration = float(data.get('videoDuration', 0))

    if not subtitles:
        return jsonify({"success": False, "error": "No subtitles to generate speech for"}), 400

    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(DEFAULT_OUTPUT_FOLDER, f"studio_tts_{job_id}")
    speech_dir = os.path.join(out_dir, "speech")
    os.makedirs(speech_dir, exist_ok=True)

    segment_files = []
    timed_segs = _normalize_segment_timings(subtitles, video_duration)
    timed_segs = _expand_slots_into_gaps(timed_segs, video_duration)

    try:
        async def _gen_all():
            for i, blk in enumerate(timed_segs, start=1):
                txt = blk.get("text", "").strip()
                v_key = blk.get("voice") or default_voice
                slot = max(0.35, float(blk["end"]) - float(blk["start"]))

                out = os.path.join(speech_dir, f"segment_{i:04d}.mp3")
                await _synthesize_text_to_file(
                    txt, v_key, out,
                    custom_rate=rate, custom_pitch=pitch, slot_duration=slot
                )
                segment_files.append(out)

        asyncio.run(_gen_all())

        full_mp3 = os.path.join(speech_dir, "full.mp3")
        last_t = timed_segs[-1]["end"] if timed_segs else 0
        path, actual_dur = _build_timed_speech_track(
            timed_segs, segment_files, video_duration or last_t, full_mp3
        )
        return jsonify({
            "success": True,
            "count": len(segment_files),
            "fullAudioPath": full_mp3,
            "duration": actual_dur,
            "outputDir": out_dir
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ═══════════════════════════════════════════════════════════
# BATCH MERGE — reorder many videos → one long video + audio
# ═══════════════════════════════════════════════════════════

def batch_merge_task(file_paths, job_name, custom_output_dir, reencode=True):
    """
    Concatenate videos in given order into one file.
    file_paths: list of absolute paths already ordered.
    """
    progress_state = _task_state()
    list_file = None
    try:
        progress_state["is_processing"] = True
        progress_state["status"] = f"Merging {len(file_paths)} videos..."
        progress_state["percent"] = 5

        if not file_paths:
            raise ValueError("No videos to merge")

        for p in file_paths:
            if not os.path.exists(p):
                raise ValueError(f"Missing file: {p}")
            ext = os.path.splitext(p)[1].lower()
            if ext in _REJECT_EXT:
                raise ValueError(f"Not a video: {p}")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_label = _safe_job_folder_name(job_name, fallback="batch_merge")
        job_folder = f"{folder_label}_{stamp}"
        base_root = custom_output_dir if custom_output_dir else DEFAULT_OUTPUT_FOLDER
        job_dir = os.path.join(base_root, job_folder)
        os.makedirs(job_dir, exist_ok=True)

        list_file = os.path.join(job_dir, "concat_list.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for pth in file_paths:
                # ffmpeg concat resolves relative paths relative to the list file
                # location — always write absolute paths so uploads/ is found.
                abs_pth = os.path.abspath(pth)
                safe = abs_pth.replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{safe}'\n")

        progress_state["status"] = f"List ready ({len(file_paths)} clips) → ffmpeg..."
        progress_state["percent"] = 15

        out_video = os.path.join(job_dir, "merged.mp4")
        import subprocess

        def run_ffmpeg(cmd):
            return subprocess.run(
                cmd, capture_output=True,
                timeout=max(600, 90 * len(file_paths)),
            )

        def _has_audio(path):
            """Quick check that output has at least one audio stream."""
            try:
                p = subprocess.run(
                    [
                        "ffprobe", "-v", "error", "-select_streams", "a",
                        "-show_entries", "stream=codec_type",
                        "-of", "csv=p=0", path,
                    ],
                    capture_output=True, timeout=30,
                )
                out = (p.stdout or b"").decode("utf-8", errors="ignore").strip()
                return p.returncode == 0 and "audio" in out.lower()
            except Exception:
                return False

        used_reencode = reencode
        if not reencode:
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", list_file, "-c", "copy", out_video,
            ]
            r = run_ffmpeg(cmd)
            bad = (
                r.returncode != 0
                or not os.path.exists(out_video)
                or os.path.getsize(out_video) < 1000
                or not _has_audio(out_video)
            )
            if bad:
                used_reencode = True
                print("[batch] copy failed or no audio, re-encoding...")
                try:
                    if os.path.exists(out_video):
                        os.remove(out_video)
                except Exception:
                    pass

        if used_reencode:
            progress_state["status"] = f"Re-encoding {len(file_paths)} videos (may take time)..."
            progress_state["percent"] = 25
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", list_file,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                out_video,
            ]
            r = run_ffmpeg(cmd)
            if r.returncode != 0 or not os.path.exists(out_video) or os.path.getsize(out_video) < 1000:
                err = (r.stderr or b"").decode("utf-8", errors="ignore")[-600:]
                raise ValueError(f"ffmpeg merge failed: {err}")
            if not _has_audio(out_video):
                raise ValueError(
                    "Merged file has no audio. "
                    "Check that your source clips have sound, then try again."
                )

        progress_state["percent"] = 85
        progress_state["status"] = "Writing order list + info..."

        order_path = os.path.join(job_dir, "order.txt")
        with open(order_path, "w", encoding="utf-8") as f:
            for i, pth in enumerate(file_paths, 1):
                f.write(f"{i:03d}. {os.path.basename(pth)}\n")

        dur = _get_media_duration(out_video)
        info_path = os.path.join(job_dir, "info.txt")
        with open(info_path, "w", encoding="utf-8") as f:
            f.write(f"Job       : {folder_label}\n")
            f.write(f"Folder    : {job_folder}\n")
            f.write(f"Created   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Clips     : {len(file_paths)}\n")
            f.write(f"Duration  : {dur:.1f}s ({dur/60:.1f} min)\n")
            f.write(f"Output    : merged.mp4\n")
            f.write("Order     : see order.txt\n")

        size_mb = os.path.getsize(out_video) / (1024 * 1024)
        abs_out = os.path.abspath(out_video)
        progress_state["percent"] = 100
        progress_state["status"] = (
            f"Done! merged.mp4 — "
            f"{len(file_paths)} clips · {dur/60:.1f} min · {size_mb:.0f} MB"
            f" · upload this file in Auto later"
        )
        progress_state["result_path"] = abs_out
        save_to_history(job_folder, abs_out, len(file_paths))

    except Exception as e:
        progress_state["status"] = f"Error: {str(e)}"
        progress_state["percent"] = 0
    finally:
        progress_state["is_processing"] = False


@app.route('/batch-merge', methods=['POST'])
def batch_merge():
    """
    Upload many videos + order (JSON list of filenames).
    Form fields:
      videos   — multiple files
      order    — JSON array of original filenames in desired order
      jobName  — folder name
      reencode — 'true' (default, safer) | 'false' (fast copy)
    """
    files = request.files.getlist("videos")
    if not files:
        return jsonify({"success": False, "error": "No videos uploaded"}), 400

    order_raw = request.form.get("order", "[]")
    try:
        order = json.loads(order_raw)
        if not isinstance(order, list):
            order = []
    except Exception:
        order = []

    job_name = request.form.get("jobName", "").strip() or "batch_merge"
    reencode = request.form.get("reencode", "true") != "false"
    custom_output_dir = request.form.get("outputDir", "")

    saved = {}
    stamp = datetime.now().strftime("%H%M%S")
    for i, vf in enumerate(files):
        if not vf or not vf.filename:
            continue
        orig = vf.filename
        ext = os.path.splitext(orig)[1].lower()
        if ext in _REJECT_EXT:
            continue
        safe = secure_filename(orig) or f"clip_{i}.mp4"
        stem, ex = os.path.splitext(safe)
        path = os.path.abspath(os.path.join(UPLOAD_FOLDER, f"batch_{stamp}_{i:03d}_{stem}{ex}"))
        vf.save(path)
        saved[os.path.basename(orig)] = path
        saved[orig] = path
        saved[safe] = path

    if not saved:
        return jsonify({"success": False, "error": "No valid video files"}), 400

    ordered_paths = []
    if order:
        for name in order:
            base = os.path.basename(name)
            path = saved.get(name) or saved.get(base)
            if path and path not in ordered_paths:
                ordered_paths.append(path)
        for path in saved.values():
            if path not in ordered_paths:
                ordered_paths.append(path)
    else:
        ordered_paths = list(dict.fromkeys(saved.values()))

    st = _start_task("batch_merge_task", dict(
        file_paths=ordered_paths, job_name=job_name, custom_output_dir=custom_output_dir, reencode=reencode),
        name=f"Merge: {job_name or str(len(ordered_paths)) + ' videos'}", kind="merge")
    return jsonify({
        "success": True,
        "message": f"Merging {len(ordered_paths)} videos",
        "count": len(ordered_paths),
        "task_id": st["id"],
    })


register_task_runner(srt_generation_task)
register_task_runner(cut_video_task)
register_task_runner(srt_to_speech_task)
register_task_runner(download_yt_task)
register_task_runner(studio_export_task)
register_task_runner(batch_merge_task)
