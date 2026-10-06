"""
Subtitle files, timing, and ffmpeg work: durations, mixing, Demucs music separation, merging,
muxing and burning subtitles into video.
"""
import os
from moviepy import VideoFileClip
from .core import (
    VC_DIR,
    VC_PYTHON,
    _BASE_DIR,
    format_timestamp,
    load_config,
)
from .tasks import (
    _uses_gpu,
)


_FAST_ENCODER_CACHE = None


def parse_srt_blocks(srt_path):
    """Parse SRT into list of {start, end, text} (seconds)."""
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = []
    for block in content.strip().split("\n\n"):
        lines = [ln.strip() for ln in block.strip().split("\n") if ln.strip()]
        if len(lines) < 2:
            continue
        # Skip index line if present
        if lines[0].isdigit() and len(lines) >= 3:
            time_line = lines[1]
            text = " ".join(lines[2:])
        else:
            time_line = lines[0]
            text = " ".join(lines[1:])
        if "-->" not in time_line:
            continue
        left, right = time_line.split("-->")
        start = _srt_time_to_sec(left.strip())
        end = _srt_time_to_sec(right.strip())
        text = text.strip()
        if text:
            blocks.append({"start": start, "end": end, "text": text})
    return blocks


def _srt_time_to_sec(t):
    """00:00:01,500 or 00:00:01.500 → seconds"""
    t = t.replace(",", ".")
    parts = t.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(parts[0])


# ═══════════════════════════════════════════════════════════
# PIPELINE: Video (CN/EN) → SRT → Translate Khmer → TTS
# ═══════════════════════════════════════════════════════════

def _write_srt(path, segments):
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(
                f"{i}\n{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n"
                f"{seg['text'].strip()}\n\n"
            )


def _get_media_duration(file_path):
    """Return duration in seconds (float) for video/audio."""
    import subprocess
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path,
            ],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode == 0 and r.stdout.strip():
            return float(r.stdout.strip())
    except Exception as e:
        print(f"[duration ffprobe]: {e}")
    try:
        clip = VideoFileClip(file_path)
        d = float(clip.duration or 0)
        clip.close()
        if d > 0:
            return d
    except Exception as e:
        print(f"[duration moviepy]: {e}")
    return 0.0


def _normalize_segment_timings(segments, video_duration):
    """
    Make subtitle times consistent with video length:
    - monotonic non-overlapping
    - each line has a sensible minimum duration
    - last cue ends near video end (2 min video → SRT covers ~2 min)
    """
    if not segments:
        return segments
    segs = []
    for s in segments:
        start = max(0.0, float(s.get("start", 0)))
        end = max(start + 0.3, float(s.get("end", start + 1.0)))
        segs.append({**s, "start": start, "end": end})

    segs.sort(key=lambda x: x["start"])

    # Fix overlaps & tiny gaps
    for i in range(1, len(segs)):
        if segs[i]["start"] < segs[i - 1]["end"]:
            mid = (segs[i - 1]["end"] + segs[i]["start"]) / 2.0
            segs[i - 1]["end"] = mid
            segs[i]["start"] = mid
        # ensure min 0.4s duration
        if segs[i]["end"] - segs[i]["start"] < 0.4:
            segs[i]["end"] = segs[i]["start"] + 0.4

    if segs and segs[0]["end"] - segs[0]["start"] < 0.4:
        segs[0]["end"] = segs[0]["start"] + 0.4

    # Stretch last end to video duration if within 3s short (cover full video)
    if video_duration and video_duration > 0 and segs:
        if segs[-1]["end"] < video_duration:
            # if last ends early, extend last cue (or leave silence — prefer extend a bit)
            gap = video_duration - segs[-1]["end"]
            if gap < 5.0:
                segs[-1]["end"] = video_duration
        # clamp any cue past video end
        for s in segs:
            if s["start"] > video_duration:
                s["start"] = max(0, video_duration - 0.5)
            if s["end"] > video_duration:
                s["end"] = video_duration
            if s["end"] <= s["start"]:
                s["end"] = min(video_duration, s["start"] + 0.5)

    return segs


def _expand_slots_into_gaps(segments, video_duration):
    """
    Give each line more room by borrowing silence between cues.
    Keeps start times (when speech begins in video) — only extends ends.
    More aggressive borrow so TTS rarely needs to speed up (easier listening).
    """
    if not segments:
        return segments
    segs = [dict(s) for s in segments]
    n = len(segs)
    for i in range(n):
        start = float(segs[i]["start"])
        end = float(segs[i]["end"])
        # next boundary
        if i + 1 < n:
            next_start = float(segs[i + 1]["start"])
            # use up to ~95% of the gap — leave a short breath before next line
            room = start + (next_start - start) * 0.95
            if room > end:
                segs[i]["end"] = room
        else:
            # last line: extend toward video end
            if video_duration and video_duration > end:
                segs[i]["end"] = min(
                    video_duration,
                    end + max(0.8, (video_duration - end) * 0.9),
                )
        if segs[i]["end"] <= segs[i]["start"]:
            segs[i]["end"] = segs[i]["start"] + 0.6
    return segs


def _video_has_audio(video_path):
    """Check if video file contains at least one audio stream."""
    import subprocess
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        video_path
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=12)
        return bool(r.stdout.strip())
    except Exception:
        return True


def _dub_audio_filter(dubbing_mode, bgm_volume, voice_volume, bg_in="[0:a]", voice_in="[1:a]",
                      has_bg=True, center_cancel=False):
    """
    Filter graph that mixes dubbed voice over the original background, ending in [aout].
    - duck / isolate_bgm: background dips gently while the voice talks (no hard pumping)
    - mix: background at a fixed level
    - mute (or no background): voice only
    Always finishes with loudness normalisation + a true-peak limiter so nothing clips.
    """
    final = "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000,alimiter=limit=0.89:attack=5:release=80[aout]"
    voice = f"{voice_in}aformat=sample_rates=48000:channel_layouts=stereo,volume={voice_volume:.2f}"
    if not has_bg or dubbing_mode == "mute":
        return f"{voice},{final}"
    bg = f"{bg_in}aformat=sample_rates=48000:channel_layouts=stereo"
    if center_cancel:
        # fallback vocal reduction: lowers centre-panned dialogue, keeps stereo music/effects
        bg += ",stereotools=mlev=0.12"
    bg += f",volume={bgm_volume:.2f}"
    if dubbing_mode in ("duck", "isolate_bgm"):
        return (
            f"{voice},asplit=2[sc_voice][mix_voice];"
            f"{bg}[bg];"
            f"[bg][sc_voice]sidechaincompress=threshold=0.05:ratio=4:attack=20:release=300[ducked_bg];"
            f"[ducked_bg][mix_voice]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,{final}"
        )
    return (
        f"{voice}[mix_voice];{bg}[bg];"
        f"[bg][mix_voice]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,{final}"
    )


@_uses_gpu("Separating music from voices")
def _separate_background_demucs(media_path, work_dir, progress_cb=None):
    """
    Real vocal removal with Demucs (htdemucs, two stems). Returns path to a WAV of the
    background (music + effects) or None. Runs only on an NVIDIA GPU unless
    config "demucs_allow_cpu" is true — on CPU a 2h movie takes 1–2 hours.
    """
    import subprocess
    import sys
    import glob
    import shutil
    cfg = load_config()
    py = sys.executable
    env = dict(os.environ)
    if os.path.isfile(VC_PYTHON):
        py, has_cuda = VC_PYTHON, True     # GPU install made for the voice engine
        env["TORCH_HOME"] = os.path.join(VC_DIR, "torch-cache")
    else:
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except Exception:
            return None
    if not has_cuda and not cfg.get("demucs_allow_cpu", False):
        return None

    sep_dir = os.path.join(work_dir, "_demucs")
    os.makedirs(sep_dir, exist_ok=True)
    dur = _get_media_duration(media_path) or 600
    try:
        if progress_cb:
            progress_cb("Removing original voices (Demucs AI)…")
        # 10-minute chunks keep RAM / VRAM use low on long movies
        r = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", media_path, "-vn",
             "-ac", "2", "-ar", "44100", "-f", "segment", "-segment_time", "600",
             os.path.join(sep_dir, "chunk_%03d.wav")],
            capture_output=True, timeout=max(600, int(dur / 3)),
        )
        chunks = sorted(glob.glob(os.path.join(sep_dir, "chunk_*.wav")))
        if r.returncode != 0 or not chunks:
            return None
        r = subprocess.run(
            [py, "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs",
             "-d", "cuda" if has_cuda else "cpu", "-o", sep_dir, *chunks],
            capture_output=True, timeout=max(1800, int(dur * (1 if has_cuda else 3))), env=env,
        )
        if r.returncode != 0:
            print(f"[demucs failed]: {(r.stderr or b'').decode('utf-8', errors='ignore')[-400:]}")
            return None
        parts = []
        for c in chunks:
            nv = os.path.join(sep_dir, "htdemucs", os.path.splitext(os.path.basename(c))[0], "no_vocals.wav")
            if not os.path.isfile(nv):
                return None
            parts.append(nv)
        list_file = os.path.join(sep_dir, "list.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for pth in parts:
                f.write("file '" + pth.replace(chr(92), "/").replace("'", "'" + chr(92) + "''") + "'\n")
        out = os.path.join(work_dir, "background_no_vocals.wav")
        r = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", list_file, "-c", "copy", out],
            capture_output=True, timeout=600,
        )
        return out if r.returncode == 0 and os.path.isfile(out) else None
    except Exception as e:
        print(f"[demucs]: {e}")
        return None
    finally:
        for c in glob.glob(os.path.join(sep_dir, "chunk_*.wav")):
            try:
                os.remove(c)
            except OSError:
                pass
        shutil.rmtree(os.path.join(sep_dir, "htdemucs"), ignore_errors=True)


def _mix_audio_tracks(orig_video_path, tts_audio_path, out_mix_path, dubbing_mode="duck", bgm_volume=0.25, voice_volume=1.0, progress_cb=None):
    """
    Produce the final dubbed soundtrack (background + Khmer voice) once, so the video
    encode only has to copy it in. Output format follows the extension (.wav / .mp3 / .m4a).
    """
    import subprocess
    has_audio = _video_has_audio(orig_video_path)
    bg_source = orig_video_path
    center_cancel = False
    if has_audio and dubbing_mode == "isolate_bgm":
        sep = _separate_background_demucs(
            orig_video_path, os.path.dirname(os.path.abspath(out_mix_path)), progress_cb=progress_cb
        )
        if sep:
            bg_source = sep
        else:
            center_cancel = True

    fc = _dub_audio_filter(
        dubbing_mode, bgm_volume, voice_volume,
        has_bg=has_audio, center_cancel=center_cancel,
    )
    ext = os.path.splitext(out_mix_path)[1].lower()
    if ext == ".wav":
        codec = ["-c:a", "pcm_s16le"]
    elif ext in (".m4a", ".aac"):
        codec = ["-c:a", "aac", "-b:a", "192k"]
    else:
        codec = ["-c:a", "libmp3lame", "-b:a", "192k"]
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", bg_source, "-i", tts_audio_path,
        "-filter_complex", fc, "-map", "[aout]", "-vn",
        *codec, out_mix_path,
    ]
    try:
        dur = _get_media_duration(tts_audio_path) or 600
        r = subprocess.run(cmd, capture_output=True, timeout=max(300, int(dur / 3)))
        if r.returncode != 0:
            print(f"[mix audio]: {(r.stderr or b'').decode('utf-8', errors='ignore')[-400:]}")
        return out_mix_path if r.returncode == 0 and os.path.exists(out_mix_path) else ""
    except Exception as e:
        print(f"[mix audio tracks failed]: {e}")
        return ""


def _get_fastest_video_encoder():
    """Detect the best working hardware H.264 encoder (tested once, then cached)."""
    global _FAST_ENCODER_CACHE
    if _FAST_ENCODER_CACHE is not None:
        return _FAST_ENCODER_CACHE
    import subprocess
    candidates = [
        ("h264_nvenc", ["-preset", "p5", "-rc", "vbr", "-cq", "23", "-b:v", "0", "-maxrate", "8M", "-bufsize", "16M"]),
        ("h264_qsv", ["-preset", "medium", "-global_quality", "23", "-maxrate", "8M", "-bufsize", "16M"]),
        ("h264_mf", ["-b:v", "4500k", "-rate_control", "quality", "-quality", "70"]),
    ]
    for enc, args in candidates:
        try:
            # 320x240: some hardware encoders reject tiny test frames
            cmd = ["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "testsrc=duration=0.2:size=320x240:rate=30",
                   "-c:v", enc, *args, "-f", "null", "-"]
            p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
            if p.returncode == 0:
                _FAST_ENCODER_CACHE = (enc, args)
                print(f"[Hardware Acceleration]: Using {enc} hardware video encoder")
                return _FAST_ENCODER_CACHE
        except Exception:
            pass
    _FAST_ENCODER_CACHE = ("libx264", ["-preset", "veryfast", "-crf", "21"])
    print("[Hardware Acceleration]: Using libx264 CPU video encoder")
    return _FAST_ENCODER_CACHE


def _output_duration_ok(out_path, expected_sec, tolerance=0.98):
    """True if the output file exists and is (almost) as long as expected."""
    if not out_path or not os.path.isfile(out_path):
        return False
    if not expected_sec:
        return True
    got = _get_media_duration(out_path) or 0
    if got < expected_sec * tolerance:
        print(f"[encode check] {os.path.basename(out_path)} is {got:.1f}s, expected {expected_sec:.1f}s — output truncated")
        return False
    return True


def _parse_hide_box(hide_box):
    """'x,y,w,h' fractions of the source frame → tuple, or None when empty / invalid."""
    if not hide_box:
        return None
    try:
        x, y, w, h = (float(v) for v in str(hide_box).split(","))
    except ValueError:
        return None
    x, y = max(0.0, min(0.98, x)), max(0.0, min(0.98, y))
    w, h = max(0.02, min(1.0 - x, w)), max(0.02, min(1.0 - y, h))
    return x, y, w, h


def _get_aspect_ratio_filter(aspect_ratio="original", hide_box=None):
    """
    Returns FFmpeg filter chain for canvas framing (e.g. 9:16 Shorts with blurred background canvas),
    optionally blurring a box of the source picture first (hide_box "x,y,w,h" — e.g. the original
    burned-in subtitles).
    Returns (filter_prefix, base_stream_label)
    """
    fc = ""
    src = "[0:v]"
    box = _parse_hide_box(hide_box)
    if box:
        x, y, w, h = box
        fc += (
            f"[0:v]split[hb_main][hb_cut];"
            f"[hb_cut]crop=trunc(iw*{w:.4f}/2)*2:trunc(ih*{h:.4f}/2)*2:trunc(iw*{x:.4f}):trunc(ih*{y:.4f}),"
            f"gblur=sigma=28:steps=3[hb_blur];"
            f"[hb_main][hb_blur]overlay=trunc(W*{x:.4f}):trunc(H*{y:.4f})[v_hidden];"
        )
        src = "[v_hidden]"

    ar = str(aspect_ratio or "original").lower().strip()
    if ar in ("9:16", "vertical", "shorts"):
        size = "1080:1920"
    elif ar in ("1:1", "square"):
        size = "1080:1080"
    else:
        return fc, src
    # blurred, cropped copy fills the canvas; the full picture sits centred on top
    fc += (
        f"{src}split[cv_a][cv_b];"
        f"[cv_a]scale={size}:force_original_aspect_ratio=increase,crop={size},boxblur=20:5[bg_canvas];"
        "[cv_b]scale=1080:-2[fg_canvas];"
        "[bg_canvas][fg_canvas]overlay=(W-w)/2:(H-h)/2[v_canvas];"
    )
    return fc, "[v_canvas]"


def _canvas_width(video_path, aspect_ratio="original"):
    """Width in px of the frame the logo is overlaid on (after 9:16 / 1:1 canvas framing)."""
    import subprocess
    ar = str(aspect_ratio or "original").lower().strip()
    if ar in ("9:16", "vertical", "shorts", "1:1", "square"):
        return 1080
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height:stream_side_data=rotation",
             "-of", "default=nw=1", video_path],
            capture_output=True, text=True, timeout=20,
        )
        info = dict(l.split("=", 1) for l in r.stdout.splitlines() if "=" in l)
        w, h = int(info.get("width", 0)), int(info.get("height", 0))
        if abs(int(float(info.get("rotation", 0) or 0))) % 180 == 90:
            w = h
        return w or None
    except Exception:
        return None


def _get_logo_settings(logo_pos="top-right", logo_size="medium", logo_opacity=0.85, canvas_w=None):
    """Calculate size pixel, position expression and opacity for FFmpeg overlay filter.

    Besides the presets, accepts the drag-and-drop editor's exact placement:
      logo_pos  "xy:0.81,0.04"  → logo top-left corner as a fraction of the frame
      logo_size "w:0.15"        → logo width as a fraction of the frame width
    """
    size_px = 160
    s = str(logo_size).lower()
    if s == "small":
        size_px = 120
    elif s == "large":
        size_px = 220
    elif s.isdigit():
        size_px = int(s)
    elif s.startswith("w:"):
        try:
            frac = max(0.02, min(1.0, float(s[2:])))
            size_px = int(round(frac * (canvas_w or 1280)))
        except ValueError:
            pass
    size_px = max(8, size_px - size_px % 2)

    pos_expr = "W-w-24:24"  # top-right
    p = str(logo_pos or "")
    if p.startswith("xy:"):
        try:
            fx, fy = (max(0.0, min(1.0, float(v))) for v in p[3:].split(",", 1))
            pos_expr = f"W*{fx:.4f}:H*{fy:.4f}"
        except ValueError:
            pass
    elif logo_pos == "top-left":
        pos_expr = "24:24"
    elif logo_pos == "bottom-right":
        pos_expr = "W-w-24:H-h-50"
    elif logo_pos == "bottom-left":
        pos_expr = "24:H-h-50"
    elif logo_pos == "center":
        pos_expr = "(W-w)/2:(H-h)/2"

    opacity = max(0.1, min(1.0, float(logo_opacity if logo_opacity is not None else 0.85)))
    return size_px, pos_expr, opacity


def _audio_signature(path):
    """(codec, profile, sample_rate, channels) of the first audio stream, or None if no audio."""
    import subprocess
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_name,profile,sample_rate,channels",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=20,
        )
        return r.stdout.strip() or None
    except Exception:
        return None


def _merge_video_files(file_paths, output_path, progress_cb=None):
    """
    Concatenate multiple video files in exact order into one output file.
    Fast stream copy concat first; falls back to safe re-encode concat.
    """
    import subprocess
    import shutil
    if len(file_paths) == 1:
        if os.path.abspath(file_paths[0]) != os.path.abspath(output_path):
            shutil.copyfile(file_paths[0], output_path)
        return output_path

    job_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(job_dir, exist_ok=True)
    list_file = os.path.join(job_dir, "auto_concat_list.txt")
    norm_dir = None

    # Stream-copy concat keeps only the FIRST clip's audio config. If clips mix
    # AAC-LC / HE-AAC, sample rates, channels (or some have no audio), the merged
    # audio decodes as garbage. Normalize every clip's audio first (video copied).
    sigs = [_audio_signature(p) for p in file_paths]
    if len(set(sigs)) > 1 and any(sigs):
        norm_dir = os.path.join(job_dir, "_norm_clips")
        os.makedirs(norm_dir, exist_ok=True)
        normed = []
        for i, (pth, sig) in enumerate(zip(file_paths, sigs)):
            if progress_cb:
                progress_cb(f"Clips have mixed audio formats — normalizing audio {i + 1}/{len(file_paths)}...")
            out = os.path.join(norm_dir, f"n_{i:04d}.mp4")
            if sig:
                cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", pth,
                       "-map", "0:v:0", "-map", "0:a:0"]
            else:
                # no audio track → add silence so the concat stays aligned
                cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", pth,
                       "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                       "-map", "0:v:0", "-map", "1:a:0", "-shortest"]
            cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2", out]
            r = subprocess.run(cmd, capture_output=True, timeout=600)
            if r.returncode != 0 or not os.path.exists(out) or os.path.getsize(out) < 1000:
                raise RuntimeError(
                    f"Failed to normalize audio of clip {i + 1} ({os.path.basename(pth)}): "
                    f"{r.stderr.decode('utf-8', errors='ignore')[-300:]}"
                )
            normed.append(out)
        file_paths = normed

    with open(list_file, "w", encoding="utf-8") as f:
        for pth in file_paths:
            abs_pth = os.path.abspath(pth)
            safe = abs_pth.replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{safe}'\n")

    def _cleanup():
        if norm_dir:
            shutil.rmtree(norm_dir, ignore_errors=True)

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file, "-c", "copy", output_path
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=max(600, 90 * len(file_paths)))
    if r.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        _cleanup()
        return output_path

    if progress_cb:
        progress_cb("Re-encoding video clips for safe seamless merge...")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        output_path
    ]
    r2 = subprocess.run(cmd, capture_output=True, timeout=max(1200, 180 * len(file_paths)))
    _cleanup()
    if r2.returncode == 0 and os.path.exists(output_path):
        return output_path
    raise RuntimeError(f"Failed to merge videos: {r2.stderr.decode('utf-8', errors='ignore')[-300:]}")


def _mux_video_with_audio(
    video_path, audio_path, out_path,
    dubbing_mode="duck", bgm_volume=0.25, voice_volume=1.0,
    logo_path=None, logo_pos="top-right", logo_size="medium", logo_opacity=0.85,
    aspect_ratio="original", hide_box=None
):
    """
    Dub video with Khmer speech track, optional watermark logo, aspect ratio framing (e.g. 9:16 Shorts),
    and fast GPU hardware acceleration.
    """
    import subprocess
    has_audio = _video_has_audio(video_path)
    has_logo = bool(logo_path and os.path.isfile(logo_path))
    if has_logo:
        size_px, pos_expr, opacity = _get_logo_settings(
            logo_pos, logo_size, logo_opacity, canvas_w=_canvas_width(video_path, aspect_ratio))

    canvas_prefix, base_v = _get_aspect_ratio_filter(aspect_ratio, hide_box)
    enc, enc_args = _get_fastest_video_encoder()
    is_canvas_modified = bool(canvas_prefix)

    # Audio filter graph ("premixed" = audio_path is already the final soundtrack)
    if dubbing_mode in ("isolate_bgm", "duck", "mix", "mute"):
        bg_filter = _dub_audio_filter(
            dubbing_mode, bgm_volume, voice_volume,
            has_bg=has_audio, center_cancel=(dubbing_mode == "isolate_bgm"),
        )
    else:
        bg_filter = ""
    vid_dur = _get_media_duration(video_path) or 0
    dur_args = ["-t", f"{vid_dur:.3f}"] if vid_dur else []
    enc_timeout = max(900, int(vid_dur * 2))

    # Build full filter complex
    v_chains = []
    if is_canvas_modified:
        v_chains.append(canvas_prefix)
    
    if has_logo:
        logo_idx = 2 if audio_path else 1
        v_chains.append(f"[{logo_idx}:v]scale={size_px}:-1,format=rgba,colorchannelmixer=aa={opacity:.2f}[logo];")
        v_chains.append(f"{base_v}[logo]overlay={pos_expr}[vout]")
        needs_filter = True
    elif is_canvas_modified:
        v_chains.append(f"{base_v}copy[vout]")
        needs_filter = True
    else:
        needs_filter = False

    if needs_filter or bg_filter:
        fc = "".join(v_chains)
        if fc and bg_filter:
            fc = fc + ";" + bg_filter
        elif bg_filter:
            fc = bg_filter

        inputs = ["-i", video_path, "-i", audio_path]
        if has_logo:
            inputs += ["-i", logo_path]

        v_map = ["[vout]"] if needs_filter else ["0:v:0"]
        a_map = ["[aout]"] if bg_filter else ["1:a:0"]

        v_codec = ["-c:v", enc, *enc_args] if needs_filter else ["-c:v", "copy"]
        cmd = [
            "ffmpeg", "-y", "-hide_banner",
            "-hwaccel", "auto",
            *inputs,
            "-filter_complex", fc,
            "-map", v_map[0],
            "-map", a_map[0],
            *v_codec,
            "-c:a", "aac", "-b:a", "192k",
            *dur_args,
            "-movflags", "+faststart",
            out_path
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=enc_timeout)
        if r.returncode == 0 and _output_duration_ok(out_path, vid_dur):
            return out_path
        # CPU fallback if hardware encoder rejected specific options / stopped early
        cmd_cpu = [
            "ffmpeg", "-y", "-hide_banner",
            *inputs,
            "-filter_complex", fc,
            "-map", v_map[0],
            "-map", a_map[0],
            *(["-c:v", "libx264", "-preset", "veryfast", "-crf", "21"] if needs_filter else ["-c:v", "copy"]),
            "-c:a", "aac", "-b:a", "192k",
            *dur_args,
            "-movflags", "+faststart",
            out_path
        ]
        r_cpu = subprocess.run(cmd_cpu, capture_output=True, timeout=max(enc_timeout, int(vid_dur * 4)))
        if r_cpu.returncode == 0 and _output_duration_ok(out_path, vid_dur):
            return out_path

    # Simple stream copy fallback
    cmd_fallback = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        *dur_args,
        "-movflags", "+faststart",
        out_path
    ]
    r = subprocess.run(cmd_fallback, capture_output=True, timeout=enc_timeout)
    if r.returncode != 0:
        err = (r.stderr or b"").decode("utf-8", errors="ignore")[-500:]
        raise ValueError(f"ffmpeg mux failed: {err}")
    if not _output_duration_ok(out_path, vid_dur):
        raise ValueError("Dubbed video came out shorter than the original — encode stopped early")
    return out_path


def _write_bilingual_srt(path, km_segments, src_segments):
    """Write bilingual SRT: Khmer line first, original language line second."""
    with open(path, "w", encoding="utf-8") as f:
        for i, km in enumerate(km_segments, start=1):
            st = format_timestamp(km["start"])
            et = format_timestamp(km["end"])
            km_text = km.get("text", "").strip()
            src_text = src_segments[i-1].get("text", "").strip() if i-1 < len(src_segments) else ""
            if src_text and src_text != km_text:
                combined = f"{km_text}\n{src_text}"
            else:
                combined = km_text
            f.write(f"{i}\n{st} --> {et}\n{combined}\n\n")


def _khmer_font_path():
    """Bundled Noto Sans Khmer for burn-in subtitles (Khmer script support)."""
    p = os.path.join(_BASE_DIR, "static", "fonts", "NotoSansKhmer-Regular.ttf")
    if os.path.isfile(p):
        return p
    for cand in (
        "/usr/share/fonts/truetype/noto/NotoSansKhmer-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansKhmer-Regular.otf",
    ):
        if os.path.isfile(cand):
            return cand
    return None


def _escape_ffmpeg_sub_path(path):
    """Escape path for ffmpeg subtitles= filter."""
    p = os.path.abspath(path).replace("\\", "/")
    p = p.replace(":", "\\:").replace("'", "\\'")
    return p


def _burn_subtitles(
    video_path,
    srt_path,
    out_path,
    audio_path=None,
    dubbing_mode="duck",
    bgm_volume=0.25,
    voice_volume=1.0,
    font_size=22,
    sub_color="yellow",
    sub_box=False,
    sub_pos="bottom",
    logo_path=None,
    logo_pos="top-right",
    logo_size="medium",
    logo_opacity=0.85,
    aspect_ratio="original",
    hide_box=None
):
    """
    Burn Khmer / Bilingual SRT onto video with pro styling, optional watermark logo,
    TikTok/Shorts 9:16 blurred background canvas, vocal remover, and GPU hardware acceleration.
    """
    import subprocess
    font = _khmer_font_path()
    srt_esc = _escape_ffmpeg_sub_path(srt_path)

    has_logo = bool(logo_path and os.path.isfile(logo_path))
    if has_logo:
        size_px, pos_expr, opacity = _get_logo_settings(
            logo_pos, logo_size, logo_opacity, canvas_w=_canvas_width(video_path, aspect_ratio))

    # Subtitle color mapping in ASS format (&HAABBGGRR)
    color_map = {
        "yellow": "&H0000FFFF",   # Cinema Yellow (BGR order)
        "white": "&H00FFFFFF",    # Clean White
        "cyan": "&H00FFFF00",     # Modern Neon Cyan
        "green": "&H0000FF00",    # Lime Green
    }
    primary_col = color_map.get(str(sub_color).lower(), "&H0000FFFF")

    align_val = 6 if sub_pos == "top" else (5 if sub_pos == "center" else 2)
    margin_v = 40 if sub_pos == "top" else 26
    if str(sub_pos).startswith("y:"):
        # libass lays SRT out on a 288-row canvas, so MarginV is in 1/288ths of the frame height
        try:
            align_val, margin_v = 2, int(round(max(0.0, min(0.9, float(str(sub_pos)[2:]))) * 288))
        except ValueError:
            pass

    style_parts = [
        f"FontSize={int(font_size)}",
        f"PrimaryColour={primary_col}",
        "OutlineColour=&H00000000",
        f"MarginV={margin_v}",
        f"Alignment={align_val}",
    ]

    if sub_box:
        # Box background style for enhanced readability
        style_parts.extend(["BorderStyle=3", "Outline=3", "BackColour=&H80000000"])
    else:
        # Classic crisp border with subtle drop shadow
        style_parts.extend(["BorderStyle=1", "Outline=2.5", "Shadow=1", "BackColour=&H00000000"])

    if font:
        fontsdir = _escape_ffmpeg_sub_path(os.path.dirname(font))
        style_parts.insert(0, "FontName=Noto Sans Khmer")
        vf = f"subtitles='{srt_esc}':fontsdir='{fontsdir}':force_style='{','.join(style_parts)}'"
    else:
        vf = f"subtitles='{srt_esc}':force_style='{','.join(style_parts)}'"

    has_audio = _video_has_audio(video_path)
    dur = _get_media_duration(video_path) or 0
    timeout = max(600, int(dur * 3) + 120)

    canvas_prefix, base_v = _get_aspect_ratio_filter(aspect_ratio, hide_box)
    enc, enc_args = _get_fastest_video_encoder()

    # Build audio filter ("premixed" = audio_path is already the final soundtrack)
    bg_filter = ""
    if audio_path and os.path.isfile(audio_path) and dubbing_mode in ("isolate_bgm", "duck", "mix", "mute"):
        bg_filter = _dub_audio_filter(
            dubbing_mode, bgm_volume, voice_volume,
            has_bg=has_audio, center_cancel=(dubbing_mode == "isolate_bgm"),
        )

    # Build video filter
    inputs = ["-i", video_path]
    if audio_path and os.path.isfile(audio_path):
        inputs += ["-i", audio_path]
    if has_logo:
        inputs += ["-i", logo_path]

    logo_idx = (2 if (audio_path and os.path.isfile(audio_path)) else 1) if has_logo else None

    v_filter = f"{canvas_prefix}{base_v}{vf}"
    if has_logo:
        fc = (
            f"[{logo_idx}:v]scale={size_px}:-1,format=rgba,colorchannelmixer=aa={opacity:.2f}[logo];"
            f"{v_filter}[v_sub];"
            f"[v_sub][logo]overlay={pos_expr}[vout]"
        )
    else:
        fc = f"{v_filter}[vout]"

    if bg_filter:
        fc = fc + ";" + bg_filter
        a_map = ["-map", "[aout]"]
        a_codec = ["-c:a", "aac", "-b:a", "192k"]
    elif audio_path and os.path.isfile(audio_path):
        a_map = ["-map", "1:a:0"]
        a_codec = ["-c:a", "aac", "-b:a", "192k"]
    elif has_audio:
        a_map = ["-map", "0:a:0"]
        a_codec = ["-c:a", "copy"]
    else:
        a_map = []
        a_codec = []

    dur_args = ["-t", f"{dur:.3f}"] if dur else []

    # Run with hardware encoder (+ hardware decoding)
    cmd = [
        "ffmpeg", "-y", "-hide_banner",
        "-hwaccel", "auto",
        *inputs,
        "-filter_complex", fc,
        "-map", "[vout]",
        *a_map,
        "-c:v", enc, *enc_args,
        *a_codec,
        *dur_args,
        "-movflags", "+faststart",
        out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if r.returncode == 0 and _output_duration_ok(out_path, dur):
        return out_path
    print(f"[burn subs] {enc} failed or short, retrying on CPU: "
          f"{(r.stderr or b'').decode('utf-8', errors='ignore')[-300:]}")

    # Fallback with CPU libx264
    cmd_fallback = [
        "ffmpeg", "-y", "-hide_banner",
        *inputs,
        "-filter_complex", fc,
        "-map", "[vout]",
        *a_map,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        *a_codec,
        *dur_args,
        "-movflags", "+faststart",
        out_path,
    ]
    r2 = subprocess.run(cmd_fallback, capture_output=True, timeout=timeout)
    if r2.returncode != 0:
        err = (r2.stderr or b"").decode("utf-8", errors="ignore")[-800:]
        raise ValueError(f"ffmpeg burn subs failed: {err}")
    if not _output_duration_ok(out_path, dur):
        raise ValueError("Subtitled video came out shorter than the original — encode stopped early")
    return out_path
