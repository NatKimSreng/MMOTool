"""
Paths, the Flask app, settings (config.json) and API keys (kept in the user profile), history.
"""
import os
import re
import json
import logging
from flask import Flask
from werkzeug.utils import secure_filename
from datetime import datetime


# Always resolve paths relative to the app folder (the one with app.py), not the process cwd.
# Fixes "key not saved" when the app is launched from a different folder.
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, root_path=_BASE_DIR)   # templates/ and static/ are in the app folder


class _QuietPolling(logging.Filter):
    """The pages poll these every second — successful polls made up ~80% of dubber.log."""
    _PATHS = ('"GET /progress', '"GET /api/tasks', '"GET /api/batch-queue', '"GET /static/')

    def filter(self, record):
        msg = record.getMessage()
        ok = '" 200 ' in msg or '" 304 ' in msg   # errors on these paths still get logged
        return not (ok and any(p in msg for p in self._PATHS))


logging.getLogger("werkzeug").addFilter(_QuietPolling())
# Local desktop app: allow multi-GB (1–2h) video uploads without Flask rejecting them.
app.config["MAX_CONTENT_LENGTH"] = None

UPLOAD_FOLDER = os.path.join(_BASE_DIR, 'uploads')
DEFAULT_OUTPUT_FOLDER = os.path.join(_BASE_DIR, 'outputs')
DOWNLOAD_FOLDER = os.path.join(_BASE_DIR, 'downloads')
HISTORY_FILE = os.path.join(_BASE_DIR, 'history.json')
CONFIG_FILE = os.path.join(_BASE_DIR, 'config.json')
CLONED_VOICES_FILE = os.path.join(_BASE_DIR, 'cloned_voices.json')
CLONED_SAMPLES_DIR = os.path.join(_BASE_DIR, 'cloned_voices', 'samples')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DEFAULT_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
os.makedirs(CLONED_SAMPLES_DIR, exist_ok=True)


def _safe_upload_name(original_name: str) -> str:
    """Keep extension; avoid empty/unsafe names without stripping Unicode badly."""
    base = os.path.basename(original_name or "video.mp4")
    name, ext = os.path.splitext(base)
    if not ext:
        ext = ".mp4"
    safe = secure_filename(name) or "video"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe}_{stamp}{ext.lower()}"


# API keys live in the Windows user profile, not in the project folder, so zipping / copying /
# sharing the app folder never hands them out. config.json keeps only the normal settings.
SECRETS_DIR = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "DAI Dubber")
SECRETS_FILE = os.path.join(SECRETS_DIR, "secrets.json")


def _is_secret_key(k):
    return str(k).endswith("_API_KEY")


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"[config] could not read {path}: {e}")
        return {}


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _move_keys_out_of_project():
    """One-time move: API keys found in config.json go to SECRETS_FILE, then config.json is
    rewritten without them (a copy of the old file is kept next to the secrets)."""
    file_cfg = _read_json(CONFIG_FILE)
    keys = {k: v for k, v in file_cfg.items() if _is_secret_key(k)}
    if not keys:
        return
    try:
        secrets = _read_json(SECRETS_FILE)
        if not os.path.exists(os.path.join(SECRETS_DIR, "config.before-key-move.json")):
            _write_json(os.path.join(SECRETS_DIR, "config.before-key-move.json"), file_cfg)
        for k, v in keys.items():
            if v or k not in secrets:
                secrets[k] = v
        _write_json(SECRETS_FILE, secrets)
        _write_json(CONFIG_FILE, {k: v for k, v in file_cfg.items() if not _is_secret_key(k)})
        print(f"[config] API keys moved to {SECRETS_FILE}")
    except Exception as e:
        print(f"[config] could not move API keys: {e}")


_move_keys_out_of_project()


def load_config():
    """Settings from config.json (next to app.py) + API keys from SECRETS_FILE / environment."""
    cfg = {
        "GROQ_API_KEY": os.environ.get("GROQ_API_KEY", ""),
        "KIRI_API_KEY": os.environ.get("KIRI_API_KEY", ""),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
        "ELEVENLABS_API_KEY": os.environ.get("ELEVENLABS_API_KEY", ""),
        # Quality first: local Khmer FT — NOT groq (stock Whisper fails on Khmer)
        "default_asr": "khmer-ft",
        "default_translate_engine": "auto",
        "default_translate_style": "recap",
        "default_dubbing_mode": "duck",
        "default_voice": "narrator_female",
        "tts_voice": "narrator_female",
        "tts_rate": "-5%",
        "sub_color": "yellow",
        "sub_size": 22,
        "studio_cinema_dsp": True,
        "role_voice_map": {},
        "glossary": {
            "Ye Chen": "យេឆិន",
            "Xiao Yan": "សៀវយ៉ាន",
            "CEO": "លោកប្រធាន"
        },
        "default_aspect_ratio": "original",
    }
    # Apply all keys from the files, including empty string (so Clear works)
    for source in (_read_json(CONFIG_FILE), _read_json(SECRETS_FILE)):
        for k, v in source.items():
            cfg[k] = v if v is not None else ""
    return cfg


def save_config(updates: dict):
    """Merge updates: API keys into SECRETS_FILE, everything else into config.json."""
    cfg = load_config()
    cfg.update(updates)
    try:
        file_cfg = _read_json(CONFIG_FILE)
        file_cfg.update({k: v for k, v in updates.items() if not _is_secret_key(k)})
        _write_json(CONFIG_FILE, {k: v for k, v in file_cfg.items() if not _is_secret_key(k)})
        key_updates = {k: v for k, v in updates.items() if _is_secret_key(k)}
        if key_updates:
            secrets = _read_json(SECRETS_FILE)
            secrets.update(key_updates)
            _write_json(SECRETS_FILE, secrets)
        print(f"[config] saved -> {CONFIG_FILE}" + (f" (keys -> {SECRETS_FILE})" if key_updates else ""))
    except Exception as e:
        print(f"[config] save failed: {e}")
        raise
    return cfg

# Global state for progress tracking
progress_state = {
    "percent": 0,
    "status": "Waiting...",
    "is_processing": False,
    "result_path": "",
    "final_video": "",
    "final_srt": "",
    "final_audio": "",
    "job_dir": "",
    "waiting_for_review": False,
    "review_job_id": "",
    "review_segments": []
}

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_to_history(filename, output_path, segments):
    history = load_history()
    entry = {
        "id": len(history) + 1,
        "filename": filename,
        "output_path": output_path,
        "segments": segments,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    history.append(entry)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

# Scripts the AI sometimes slips into Khmer lines. Lao and Myanmar look almost like Thai,
# so they are all removed together: Hebrew, Arabic, Devanagari, Thai, Lao, Myanmar,
# Tai Tham, Myanmar Extended-B/A and Tai Viet. Khmer (U+1780–U+17FF) is never touched.
_FOREIGN_SCRIPT_RE = re.compile(
    "[֐-ۿऀ-ॿ฀-໿က-႟"
    "ᨠ-᪯ꧠ-꧿ꩠ-꫟]+"
)
_CJK_RE = re.compile("[㐀-䶿一-鿿豈-﫿]")


def _has_foreign_script(text):
    """True if text contains Thai-looking or other unwanted script (see _FOREIGN_SCRIPT_RE)."""
    return bool(_FOREIGN_SCRIPT_RE.search(text or ""))


def _has_cjk(text):
    """True if text still contains Chinese characters."""
    return bool(_CJK_RE.search(text or ""))


def _strip_foreign_script(text):
    """Thai/Lao/Myanmar etc. are never shown — remove them and tidy the spaces left behind."""
    if not _has_foreign_script(text):
        return text or ""
    text = _FOREIGN_SCRIPT_RE.sub("", text)
    # a Khmer vowel or sign whose letter was removed would show as a dotted circle — drop it
    text = re.sub("(?:^|(?<=\s))[ា-៓៝]+", "", text)
    return re.sub(r"[ 	]+", " ", text).strip()


def format_timestamp(seconds):
    """Helper to format seconds into SRT timestamp format (00:00:00,000)"""
    td = datetime.utcfromtimestamp(seconds)
    millis = int((seconds - int(seconds)) * 1000)
    return td.strftime('%H:%M:%S') + f',{millis:03d}'


# Extensions that are media (not subtitles / text)
_MEDIA_EXT = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv", ".wmv",
    ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".wma", ".opus",
}
_REJECT_EXT = {".srt", ".vtt", ".ass", ".ssa", ".txt", ".json", ".csv", ".html"}


def _assert_media_file(file_path):
    """Reject SRT/text files early with a clear error (prevents ffmpeg on .srt)."""
    if not file_path or not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    if ext in _REJECT_EXT:
        raise ValueError(
            f"You uploaded a subtitle/text file ({ext}), not a video. "
            f"Please upload a video (.mp4, .mkv, .mov…) or audio (.mp3, .wav)."
        )
    if os.path.getsize(file_path) < 1000:
        raise ValueError(f"File too small: {file_path}")
    return ext

# ═══════════════════════════════════════════════════════════
# NATIVE KHMER VOICES (local AI voice conversion, Seed-VC)
# Khmer words are spoken by Edge-TTS (correct pronunciation), then the sound of
# the voice is converted to a real native speaker recording. Runs on the GPU in
# tools/vc-env so it can't disturb the main Python install.
# ═══════════════════════════════════════════════════════════
VOICES_DIR = os.path.join(_BASE_DIR, "voices")
VC_DIR = os.path.join(_BASE_DIR, "tools")
VC_PYTHON = os.path.join(VC_DIR, "vc-env", "Scripts", "python.exe")
VC_WORKER = os.path.join(VC_DIR, "vc_worker.py")


SPEAKER_WORKER = os.path.join(VC_DIR, "speaker_worker.py")
