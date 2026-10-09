"""
Speech to text: audio extraction and every transcription engine (Kiri, Groq, Whisper...).
"""
import os
import subprocess
from moviepy import VideoFileClip
import time
from .core import (
    _has_foreign_script,
    _strip_foreign_script,
    _REJECT_EXT,
    _assert_media_file,
    load_config,
)
from .tasks import (
    _uses_gpu,
    task_warn,
)
from .media import (
    _get_media_duration,
)


def _add_nvidia_dll_dirs():
    """pip's nvidia-cublas-cu12 / nvidia-cudnn-cu12 put their DLLs in site-packages\\nvidia\\*\\bin,
    which Windows does not search — without this faster-whisper can't use the GPU."""
    import site
    roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    for root in roots:
        nv = os.path.join(root, "nvidia")
        if not os.path.isdir(nv):
            continue
        for sub in os.listdir(nv):
            bin_dir = os.path.join(nv, sub, "bin")
            if os.path.isdir(bin_dir) and bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                try:
                    os.add_dll_directory(bin_dir)
                except (AttributeError, OSError):
                    pass


_add_nvidia_dll_dirs()


def _cuda_available():
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


# whisper is optional (slow on CPU) — imported only when needed
try:
    import whisper
except ImportError:
    whisper = None


def _extract_audio_wav(file_path, progress_state):
    """
    Extract 16kHz mono audio for ASR — fast path first (mp3), then wav.
    Timeout scales with file size so long merges don't hang forever.
    Returns path to .mp3 or .wav.
    """
    import subprocess
    progress_state["status"] = "Extracting audio from video..."
    progress_state["percent"] = 12

    _assert_media_file(file_path)

    ext = os.path.splitext(file_path)[1].lower()
    if ext in _REJECT_EXT:
        raise ValueError(
            f"You uploaded a subtitle/text file ({ext}), not a video. "
            f"Upload .mp4 / .mov / .mp3 — not .srt"
        )

    # Already audio? reuse (or light convert)
    if ext in {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".wma", ".opus"}:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if size_mb > 0.01:
            progress_state["status"] = f"Using audio file ({size_mb:.0f} MB)..."
            progress_state["percent"] = 20
            return file_path

    base = os.path.splitext(file_path)[0] + "._asr_audio"
    mp3_path = base + ".mp3"
    wav_path = base + ".wav"
    last_err = ""
    size_mb = max(1.0, os.path.getsize(file_path) / (1024 * 1024))
    # ~2s per MB input, clamp 90s–20min (long episode merges need time)
    timeout_sec = int(min(1200, max(90, size_mb * 2)))

    def _ok(path, min_size=500):
        return os.path.exists(path) and os.path.getsize(path) > min_size

    def _run(cmd, out_path, label):
        nonlocal last_err
        progress_state["status"] = (
            f"Extracting audio ({label})… {size_mb:.0f} MB video, "
            f"timeout {timeout_sec // 60}m"
        )
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            last_err = f"timeout after {timeout_sec}s"
            print(f"[extract] {label} timeout")
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
            except Exception:
                pass
            return False
        last_err = (r.stderr or b"").decode("utf-8", errors="ignore")
        if r.returncode == 0 and _ok(out_path):
            return True
        print(f"[extract] {label} failed code={r.returncode}: {last_err[-300:]}")
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass
        return False

    # Fast path: 16kHz mono mp3 (small, quick encode) — preferred for ASR
    try:
        ok = _run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", file_path,
                "-vn", "-sn", "-dn",
                "-map", "0:a:0?",
                "-ac", "1", "-ar", "16000",
                "-c:a", "libmp3lame", "-b:a", "64k",
                "-threads", "0",
                mp3_path,
            ],
            mp3_path,
            "mp3",
        )
        if ok:
            progress_state["percent"] = 20
            progress_state["status"] = "Audio ready"
            return mp3_path

        # Fallback: any audio stream → mp3 (no strict map)
        ok = _run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", file_path,
                "-vn", "-sn", "-dn",
                "-ac", "1", "-ar", "16000",
                "-c:a", "libmp3lame", "-b:a", "64k",
                "-threads", "0",
                mp3_path,
            ],
            mp3_path,
            "mp3-any",
        )
        if ok:
            progress_state["percent"] = 20
            progress_state["status"] = "Audio ready"
            return mp3_path

        # Last ffmpeg: wav (slower / larger)
        ok = _run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", file_path,
                "-vn", "-sn", "-dn",
                "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le",
                "-threads", "0",
                wav_path,
            ],
            wav_path,
            "wav",
        )
        if ok:
            progress_state["percent"] = 20
            progress_state["status"] = "Audio ready"
            return wav_path

        err_l = last_err.lower()
        if "matches no streams" in last_err or "stream map" in err_l:
            raise ValueError(
                "No audio track in this video. "
                "Re-merge with Safe re-encode checked, or use clips that have sound."
            )
        if "rematrix is needed" in err_l or "error reinitializing filters" in err_l:
            raise ValueError(
                "Audio track is corrupt — the clips were merged with mixed audio formats "
                "(e.g. AAC-LC + HE-AAC). Run the job again: the merge now normalizes audio automatically."
            )
        if "no such file" in err_l or "error opening input" in err_l:
            raise ValueError(f"Cannot open file: {file_path}")
        if "timeout" in err_l:
            raise ValueError(
                f"Audio extract timed out ({timeout_sec}s) on {size_mb:.0f} MB file. "
                "Try a shorter clip, or re-merge with Safe re-encode."
            )
    except FileNotFoundError:
        print("[extract] ffmpeg not found, using MoviePy")
    except ValueError:
        raise
    except Exception as e:
        print(f"[extract] ffmpeg failed: {e}")
        last_err = str(e)

    # MoviePy only for small files (< 200 MB) — very slow on long merges
    if size_mb > 200:
        raise ValueError(
            f"ffmpeg could not extract audio from {size_mb:.0f} MB video. "
            f"Detail: {(last_err or 'unknown')[-250:]}. "
            "Install/check ffmpeg, or re-merge with Safe re-encode."
        )

    progress_state["status"] = "Extracting audio (MoviePy fallback)…"
    try:
        video = VideoFileClip(file_path)
    except Exception as e:
        hint = (last_err[-200:] if last_err else str(e))
        raise ValueError(
            f"Cannot open as video ({os.path.basename(file_path)}). "
            f"Upload merged.mp4 (not folder / .srt). Detail: {hint}"
        ) from e
    if video.audio is None:
        video.close()
        raise ValueError(
            "Video has no audio track. "
            "Re-merge with Safe re-encode, or check source clips have sound."
        )
    try:
        video.audio.write_audiofile(
            wav_path, fps=16000, nbytes=2, codec="pcm_s16le", logger=None
        )
    finally:
        video.close()

    if not _ok(wav_path):
        raise ValueError(
            "Audio extraction produced empty file. Install ffmpeg: "
            "https://ffmpeg.org/download.html"
        )
    progress_state["percent"] = 20
    progress_state["status"] = "Audio ready"
    return wav_path


def _groq_upload_transcribe(client, audio_path, language=None, prompt=None):
    """
    Upload audio to Groq Whisper correctly.
    Fixes 'No audio was received' by sending proper multipart file + content-type.
    """
    if not os.path.exists(audio_path):
        raise ValueError(f"Audio file not found: {audio_path}")
    size = os.path.getsize(audio_path)
    if size < 500:
        raise ValueError(f"Audio file too small ({size} bytes) — extraction failed")

    # Groq free tier ~25MB — compress to mp3 if large
    upload_path = audio_path
    ext = os.path.splitext(audio_path)[1].lower()
    if size > 20 * 1024 * 1024 or ext == ".wav":
        try:
            import subprocess
            mp3_path = audio_path + ".upload.mp3"
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-ac", "1", "-ar", "16000", "-b:a", "64k", mp3_path,
            ]
            r = subprocess.run(cmd, capture_output=True, timeout=300)
            if r.returncode == 0 and os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 500:
                upload_path = mp3_path
            else:
                # pydub fallback
                try:
                    from pydub import AudioSegment
                    mp3_path = audio_path + ".upload.mp3"
                    AudioSegment.from_file(audio_path).export(mp3_path, format="mp3", bitrate="64k")
                    if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 500:
                        upload_path = mp3_path
                except Exception:
                    pass
        except Exception as e:
            print(f"[groq compress]: {e}")

    # MIME type matters for some clients
    up_ext = os.path.splitext(upload_path)[1].lower()
    mime = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".webm": "audio/webm",
    }.get(up_ext, "application/octet-stream")

    filename = os.path.basename(upload_path)
    if not filename.lower().endswith((".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mp4")):
        filename = filename + ".mp3"

    with open(upload_path, "rb") as f:
        data = f.read()

    if len(data) < 500:
        raise ValueError("Audio bytes empty after read")

    kwargs = dict(
        model="whisper-large-v3",
        response_format="verbose_json",
        temperature=0.0,
        # OpenAI/Groq style: (filename, bytes, content_type)
        file=(filename, data, mime),
    )
    if language and language != "auto":
        kwargs["language"] = language
    if prompt:
        kwargs["prompt"] = prompt

    try:
        result = client.audio.transcriptions.create(**kwargs)
    except Exception as e1:
        # Retry with file handle style
        try:
            with open(upload_path, "rb") as f:
                kwargs["file"] = f
                result = client.audio.transcriptions.create(**kwargs)
        except Exception as e2:
            # Last resort: raw HTTP multipart
            try:
                import requests
                cfg = load_config()
                key = (cfg.get("GROQ_API_KEY") or "").strip()
                files = {"file": (filename, data, mime)}
                form = {"model": "whisper-large-v3", "response_format": "verbose_json", "temperature": "0"}
                if language and language != "auto":
                    form["language"] = language
                if prompt:
                    form["prompt"] = prompt
                resp = requests.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {key}"},
                    files=files,
                    data=form,
                    timeout=600,
                )
                if resp.status_code != 200:
                    raise ValueError(f"Groq HTTP {resp.status_code}: {resp.text[:300]}")
                result = resp.json()
            except Exception as e3:
                raise ValueError(
                    f"Groq audio upload failed. "
                    f"SDK: {e1} | retry: {e2} | http: {e3}. "
                    f"File={upload_path} size={len(data)}. "
                    f"Install ffmpeg and check GROQ_API_KEY."
                ) from e3

    # cleanup temp mp3
    if upload_path != audio_path and upload_path.endswith(".upload.mp3"):
        try:
            os.remove(upload_path)
        except OSError:
            pass

    return result


def _segments_from_groq_result(result):
    segments = []
    segs = getattr(result, "segments", None)
    if segs is None and isinstance(result, dict):
        segs = result.get("segments")
    if segs:
        for s in segs:
            if isinstance(s, dict):
                text = (s.get("text") or "").strip()
                start, end = float(s.get("start", 0)), float(s.get("end", 0))
            else:
                text = (getattr(s, "text", "") or "").strip()
                start = float(getattr(s, "start", 0))
                end = float(getattr(s, "end", 0))
            if text:
                segments.append({"start": start, "end": end, "text": text})
    else:
        text = getattr(result, "text", None)
        if text is None and isinstance(result, dict):
            text = result.get("text")
        if text and str(text).strip():
            segments.append({"start": 0.0, "end": 5.0, "text": str(text).strip()})
    return segments


def _has_khmer(text: str) -> bool:
    """True if text contains Khmer Unicode characters (U+1780–U+17FF)."""
    return any("\u1780" <= ch <= "\u17ff" for ch in text)


def _clean_segment_text(text: str) -> str:
    """Strip noise / repeated punctuation; keep Khmer + basic punctuation."""
    text = _strip_foreign_script(text or "").strip()
    if not text:
        return ""
    # Collapse whitespace
    text = " ".join(text.split())
    # Drop obvious English-only hallucination lines when mixed poorly
    return text


def _postprocess_segments(segments):
    """
    Clean SRT segments for Khmer quality:
    - drop empty / pure noise
    - prefer lines that contain Khmer script
    - merge ultra-short fragments
    - fix overlapping times
    """
    cleaned = []
    for s in segments:
        text = _clean_segment_text(s.get("text", ""))
        if not text:
            continue
        # Skip pure number-only or single Latin garbage tokens
        if len(text) <= 2 and not _has_khmer(text):
            continue
        start = float(s.get("start", 0))
        end = float(s.get("end", start + 0.5))
        if end <= start:
            end = start + 0.4
        cleaned.append({"start": start, "end": end, "text": text})

    if not cleaned:
        return segments  # don't wipe everything if filter too aggressive

    # If majority has Khmer, drop non-Khmer lines (common Whisper failure mode)
    khmer_count = sum(1 for s in cleaned if _has_khmer(s["text"]))
    if khmer_count >= max(1, len(cleaned) // 3):
        only_khmer = [s for s in cleaned if _has_khmer(s["text"])]
        if only_khmer:
            cleaned = only_khmer

    # Drop pure repeated hallucination (same short line 3+ times in a row)
    dedup = []
    for s in cleaned:
        if (
            dedup
            and s["text"] == dedup[-1]["text"]
            and len(s["text"]) < 25
            and (s["start"] - dedup[-1]["end"]) < 1.5
        ):
            # extend previous instead of stacking junk
            dedup[-1]["end"] = max(dedup[-1]["end"], s["end"])
            continue
        dedup.append(dict(s))
    cleaned = dedup

    # Merge consecutive fragments < 0.6s with same-ish content
    merged = []
    for s in cleaned:
        if (
            merged
            and (s["start"] - merged[-1]["end"]) < 0.35
            and (s["end"] - s["start"]) < 0.8
            and len(s["text"]) < 12
        ):
            merged[-1]["end"] = max(merged[-1]["end"], s["end"])
            if s["text"] not in merged[-1]["text"]:
                merged[-1]["text"] = (merged[-1]["text"] + " " + s["text"]).strip()
        else:
            merged.append(dict(s))

    # Ensure non-overlapping monotonic times
    for i in range(1, len(merged)):
        if merged[i]["start"] < merged[i - 1]["end"]:
            mid = (merged[i - 1]["end"] + merged[i]["start"]) / 2
            merged[i - 1]["end"] = mid
            merged[i]["start"] = mid

    return merged


def _transcribe_kiri(audio_path, progress_state):
    """Kiri STT — built for Khmer first (best cloud quality for km)."""
    audio_path = _enhance_audio_for_asr(audio_path, progress_state)
    cfg = load_config()
    api_key = (cfg.get("KIRI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError(
            "KIRI_API_KEY not set. Open Settings — get key at https://kiritts.com "
            "(Khmer-first STT, much better than Groq/Whisper for Khmer)"
        )

    progress_state["status"] = "កំពុងផ្ញើទៅ Kiri API (Khmer)..."
    progress_state["percent"] = 40

    try:
        from openai import OpenAI
    except ImportError:
        raise ValueError("Install openai: pip install openai")

    client = OpenAI(api_key=api_key, base_url="https://api.kiritts.com/v1")

    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 500:
        raise ValueError("Audio file empty — extraction failed")

    upload_path = audio_path
    # Prefer mp3 for upload
    try:
        import subprocess
        mp3_path = audio_path + ".kiri.mp3"
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", "-b:a", "64k", mp3_path],
            capture_output=True, timeout=300,
        )
        if r.returncode == 0 and os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 500:
            upload_path = mp3_path
    except Exception:
        try:
            from pydub import AudioSegment
            mp3_path = audio_path + ".kiri.mp3"
            AudioSegment.from_file(audio_path).export(mp3_path, format="mp3", bitrate="64k")
            if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 500:
                upload_path = mp3_path
        except Exception:
            pass

    with open(upload_path, "rb") as f:
        data = f.read()
    fname = os.path.basename(upload_path)
    mime = "audio/mpeg" if fname.endswith(".mp3") else "audio/wav"
    result = client.audio.transcriptions.create(
        file=(fname, data, mime),
        model="kiristt",
        language="km-KH",
        response_format="verbose_json",
    )

    if upload_path != audio_path:
        try:
            os.remove(upload_path)
        except OSError:
            pass

    segments = []
    segs = getattr(result, "segments", None) or (result.get("segments") if isinstance(result, dict) else None)
    if segs:
        for s in segs:
            if isinstance(s, dict):
                text = (s.get("text") or "").strip()
                start, end = s.get("start", 0), s.get("end", 0)
            else:
                text = (getattr(s, "text", "") or "").strip()
                start = getattr(s, "start", 0)
                end = getattr(s, "end", 0)
            if text:
                segments.append({"start": float(start), "end": float(end), "text": text})
    else:
        text = getattr(result, "text", None) or (result.get("text") if isinstance(result, dict) else "")
        if text:
            segments.append({"start": 0.0, "end": 5.0, "text": text.strip()})

    return _postprocess_segments(segments), "kiri-kiristt"


def _transcribe_groq(audio_path, progress_state):
    """Groq Whisper — fast but WEAK on Khmer (same as stock Whisper). Prefer khmer-ft or kiri."""
    cfg = load_config()
    api_key = (cfg.get("GROQ_API_KEY") or "").strip()
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Open Settings — https://console.groq.com/keys"
        )

    progress_state["status"] = "កំពុងផ្ញើទៅ Groq (លឿន តែខ្សោយលើខ្មែរ)..."
    progress_state["percent"] = 40

    try:
        from groq import Groq
    except ImportError:
        raise ValueError("Install groq: pip install groq")

    client = Groq(api_key=api_key)
    result = _groq_upload_transcribe(
        client,
        audio_path,
        language="km",
        prompt="នេះជាការនិយាយជាភាសាខ្មែរ សូមសរសេរជាអក្សរខ្មែរ។ ",
    )
    segments = _segments_from_groq_result(result)
    return _postprocess_segments(segments), "groq-whisper-large-v3"


def _enhance_audio_for_asr(audio_path, progress_state=None):
    """
    Free quality boost before STT: mono 16kHz + light denoise + normalize.
    Helps a lot on phone / noisy video. Falls back to original if ffmpeg fails.
    """
    import subprocess
    if not audio_path or not os.path.exists(audio_path):
        return audio_path
    out = audio_path + "._clean.wav"
    if progress_state is not None:
        progress_state["status"] = "Cleaning audio (noise / volume)..."
    try:
        # highpass = cut rumble; afftdn = light denoise; loudnorm = steady volume
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", audio_path,
            "-ac", "1", "-ar", "16000",
            "-af",
            "highpass=f=80,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11",
            "-c:a", "pcm_s16le",
            out,
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=600)
        if r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 500:
            return out
        # simpler fallback: just normalize
        cmd2 = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", audio_path,
            "-ac", "1", "-ar", "16000",
            "-af", "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11",
            "-c:a", "pcm_s16le",
            out,
        ]
        r2 = subprocess.run(cmd2, capture_output=True, timeout=600)
        if r2.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 500:
            return out
    except Exception as e:
        print(f"[enhance audio]: {e}")
    return audio_path


@_uses_gpu("Transcribing Khmer on this computer")
def _transcribe_faster_whisper(audio_path, model_id, progress_state, label):
    """Local Khmer fine-tuned faster-whisper — BEST free quality for Khmer. GPU when available, else CPU."""
    progress_state["status"] = f"កំពុងផ្ទុកម៉ូដែលខ្មែរ ({label})..."
    progress_state["percent"] = 28
    from faster_whisper import WhisperModel

    # Clean audio first (free quality win on noisy clips)
    clean_path = _enhance_audio_for_asr(audio_path, progress_state)
    progress_state["percent"] = 35

    # int8_float16 fits the small model in 4 GB VRAM; the whole decode is retried on CPU if CUDA
    # fails part-way (missing DLL, out of memory).
    devices = ([("cuda", "int8_float16")] if _cuda_available() else []) + [("cpu", "int8")]
    segments, last_err = None, None
    for device, compute in devices:
        try:
            model = WhisperModel(model_id, device=device, compute_type=compute)
            progress_state["status"] = (f"កំពុងបកប្រែជាអក្សរខ្មែរ (local fine-tuned, "
                                        f"{'GPU' if device == 'cuda' else 'CPU'})...")
            progress_state["percent"] = 45

            # Stronger search + Khmer-only prompt = fewer wrong words / Latin junk
            segs, info = model.transcribe(
                clean_path,
                language="km",
                task="transcribe",
                beam_size=10,
                best_of=5,
                patience=1.2,
                temperature=[0.0, 0.2, 0.4],
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=280,
                    speech_pad_ms=250,
                    threshold=0.45,
                ),
                condition_on_previous_text=True,
                initial_prompt=(
                    "នេះជាការនិយាយជាភាសាខ្មែរ។ "
                    "សូមសរសេរតែជាអក្សរខ្មែរ កុំប្រើអក្សរឡាតាំង។ "
                    "សរសេរឱ្យត្រឹមត្រូវ និងច្បាស់។ "
                ),
                word_timestamps=False,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
            )
            segments = []
            for s in segs:
                text = (s.text or "").strip()
                if text:
                    segments.append({"start": s.start, "end": s.end, "text": text})
            break
        except Exception as e:
            last_err = e
            task_warn(f"Khmer model on {'GPU' if device == 'cuda' else 'CPU'} failed: {e}")
    if segments is None:
        raise RuntimeError(f"Khmer model could not run: {last_err}")

    if clean_path != audio_path:
        try:
            os.remove(clean_path)
        except OSError:
            pass

    return _postprocess_segments(segments), label


def _transcribe_openai_whisper(audio_path, progress_state):
    """Slow CPU fallback — stock Whisper is BAD on Khmer. Last resort only."""
    if whisper is None:
        raise ValueError("openai-whisper not installed")
    progress_state["status"] = "កំពុងប្រើ Whisper medium (ខ្សោយលើខ្មែរ)..."
    progress_state["percent"] = 40
    model = whisper.load_model("medium")
    progress_state["percent"] = 55
    result = model.transcribe(
        audio_path,
        language="km",
        task="transcribe",
        temperature=0.0,
        beam_size=5,
        condition_on_previous_text=True,
        initial_prompt="នេះជាការនិយាយជាភាសាខ្មែរ សូមសរសេរជាអក្សរខ្មែរ។ ",
        fp16=False,
    )
    segments = []
    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        if text:
            segments.append({"start": seg["start"], "end": seg["end"], "text": text})
    return _postprocess_segments(segments), "openai-whisper-medium"


@_uses_gpu("Transcribing on this computer")
def _local_whisper_segments(audio_path, lang=None, model_size="small"):
    """faster-whisper on the NVIDIA GPU when its CUDA libraries load, else CPU."""
    from faster_whisper import WhisperModel
    kw = dict(beam_size=5, vad_filter=True, temperature=0.0)
    if lang and lang != "auto":
        kw["language"] = lang
    last_err = None
    # The model can load on CUDA and only fail on the first decode (e.g. cublas64_12.dll
    # missing on PCs without the NVIDIA libraries), so the whole transcribe is retried on CPU.
    for device, compute in (("cuda", "float16"), ("cpu", "int8")):
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute)
            segs, info = model.transcribe(audio_path, **kw)
            out = [{"start": x.start, "end": x.end, "text": (x.text or "").strip()}
                   for x in segs if (x.text or "").strip()]
            return out, getattr(info, "language", lang)
        except Exception as e:
            last_err = e
            task_warn(f"Local Whisper on {'GPU' if device == 'cuda' else 'CPU'} failed: {e}")
    raise RuntimeError(f"faster-whisper could not run: {last_err}")


def _groq_transcribe_chunked(client, audio_path, language=None, progress_state=None, chunk_sec=600):
    """
    Groq Whisper only accepts ~25 MB per request, so a full movie used to fail and fall
    back to slow, weaker local Whisper. Split into 10-minute pieces, send 3 at a time,
    then stitch the timestamps back together. A piece that keeps failing (rate limit)
    is transcribed locally — only that piece.
    """
    import tempfile
    import shutil
    import glob
    from concurrent.futures import ThreadPoolExecutor

    dur = _get_media_duration(audio_path) or 0
    if dur and dur <= chunk_sec * 1.5 and os.path.getsize(audio_path) < 20 * 1024 * 1024:
        return _segments_from_groq_result(_groq_upload_transcribe(client, audio_path, language=language)), "groq"

    tmp = tempfile.mkdtemp(prefix="asr_chunks_")
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", audio_path,
             "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "48k",
             "-f", "segment", "-segment_time", str(chunk_sec), "-reset_timestamps", "1",
             os.path.join(tmp, "c_%03d.mp3")],
            capture_output=True, timeout=max(600, int(dur / 10)),
        )
        chunks = sorted(glob.glob(os.path.join(tmp, "c_*.mp3")))
        if r.returncode != 0 or not chunks:
            raise ValueError("could not split audio for Groq")
        # exact offsets from the real chunk lengths
        offsets, t = [], 0.0
        for c in chunks:
            offsets.append(t)
            t += _get_media_duration(c) or chunk_sec

        done = [0]
        used = set()

        def _one(i):
            last = None
            for attempt in range(4):
                try:
                    res = _groq_upload_transcribe(client, chunks[i], language=language)
                    segs = _segments_from_groq_result(res)
                    used.add("groq")
                    break
                except Exception as e:
                    last = e
                    msg = str(e).lower()
                    if "429" in msg or "rate" in msg or "503" in msg or "timeout" in msg:
                        time.sleep(15 * (attempt + 1))
                        continue
                    segs = None
                    break
            else:
                segs = None
            if segs is None:
                task_warn(f"Groq failed on part {i + 1} ({last}) — transcribing that part locally")
                segs, _ = _local_whisper_segments(chunks[i], language)
                used.add("local")
            done[0] += 1
            if progress_state is not None:
                progress_state["status"] = f"Transcribing… {done[0]}/{len(chunks)} parts"
                progress_state["percent"] = 25 + int(20 * done[0] / len(chunks))
            return [{**x, "start": x["start"] + offsets[i], "end": x["end"] + offsets[i]} for x in segs]

        with ThreadPoolExecutor(max_workers=3) as pool:
            parts = list(pool.map(_one, range(len(chunks))))
        # Whisper invents "you" / "thank you" on silent chunk tails — drop those
        junk = {"you", "you.", "thank you", "thank you.", "thanks for watching!", "thanks for watching.", "..."}
        segments = [
            x for part in parts for x in part
            if not ((x["end"] - x["start"]) < 1.2 and x["text"].strip().lower() in junk)
        ]
        segments.sort(key=lambda x: x["start"])
        return segments, "+".join(sorted(used)) or "groq"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _transcribe_source_lang(audio_path, progress_state, lang="auto"):
    """
    Thai is never used. Whisper's auto-detect often mistakes Khmer speech for Thai (or Lao),
    so if any line comes back in Thai-looking script, transcribe again forced to Khmer and
    strip whatever of those characters are still left.
    """
    if lang == "th":
        lang = "auto"
    segments, engine = _transcribe_source_lang_once(audio_path, progress_state, lang)
    if lang == "km" or not any(_has_foreign_script(s["text"]) for s in segments):
        return segments, engine
    print("[source ASR] Thai detected — re-transcribing as Khmer")
    progress_state["status"] = "Thai detected — re-transcribing as Khmer..."
    try:
        segments, engine = _transcribe_source_lang_once(audio_path, progress_state, "km")
    except Exception as e:
        task_warn(f"Khmer retry failed: {e}")
    cleaned = []
    for s in segments:
        text = _strip_foreign_script(s["text"])
        if text:
            cleaned.append({**s, "text": text})
    return cleaned, engine


def _transcribe_source_lang_once(audio_path, progress_state, lang="auto"):
    """
    ASR for Chinese / English / auto — Groq Whisper is FAST here.
    Falls back to OpenAI Whisper API, then local faster-whisper.
    """
    cfg = load_config()
    api_key = (cfg.get("GROQ_API_KEY") or "").strip()
    openai_key = (cfg.get("OPENAI_API_KEY") or "").strip()

    progress_state["status"] = f"Transcribing speech ({lang})..."
    progress_state["percent"] = 25

    # 1) Prefer Groq for EN/ZH speed
    if api_key:
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            segments, how = _groq_transcribe_chunked(
                client, audio_path, language=lang if lang != "auto" else None,
                progress_state=progress_state,
            )
            if segments:
                return segments, f"{how}-whisper-large-v3-" + (lang or "auto")
        except Exception as e:
            task_warn(f"Groq transcription failed: {e} — trying the next engine")

    # 2) Cloud OpenAI Whisper API fallback
    if openai_key:
        try:
            progress_state["status"] = "OpenAI Whisper Cloud ASR..."
            try:
                from openai import OpenAI
            except ImportError:
                OpenAI = None
            if OpenAI is not None:
                client = OpenAI(api_key=openai_key)
                with open(audio_path, "rb") as f:
                    tr = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        response_format="verbose_json",
                        language=lang if lang != "auto" else None
                    )
                raw_segs = getattr(tr, 'segments', []) or []
            else:
                # openai package not installed (e.g. portable build) — same API over plain HTTP
                import requests
                data = {"model": "whisper-1", "response_format": "verbose_json"}
                if lang and lang != "auto":
                    data["language"] = lang
                with open(audio_path, "rb") as f:
                    r = requests.post(
                        "https://api.openai.com/v1/audio/transcriptions",
                        headers={"Authorization": f"Bearer {openai_key}"},
                        data=data,
                        files={"file": (os.path.basename(audio_path), f)},
                        timeout=600,
                    )
                if r.status_code != 200:
                    raise ValueError(f"OpenAI error {r.status_code}: {r.text[:120]}")
                raw_segs = r.json().get("segments") or []
            segments = []
            for s in raw_segs:
                txt = s.get('text', '') if isinstance(s, dict) else getattr(s, 'text', '')
                st = s.get('start', 0) if isinstance(s, dict) else getattr(s, 'start', 0)
                et = s.get('end', 0) if isinstance(s, dict) else getattr(s, 'end', 0)
                if txt.strip():
                    segments.append({"start": float(st), "end": float(et), "text": txt.strip()})
            if segments:
                return segments, "openai-whisper-1"
        except Exception as e:
            task_warn(f"OpenAI transcription failed: {e} — trying local Whisper")

    # 3) Local faster-whisper fallback
    try:
        progress_state["status"] = "Local Whisper (source language)..."
        segments, det_lang = _local_whisper_segments(audio_path, lang)
        if segments:
            return segments, f"faster-whisper-small-{det_lang}"
    except Exception as e:
        task_warn(f"Local Whisper failed: {e}")

    raise ValueError(
        "Could not transcribe. Set GROQ_API_KEY or OPENAI_API_KEY in Settings "
        "or pip install faster-whisper."
    )
