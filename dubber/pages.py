"""
Pages, files, folders and settings routes.
"""
import os
from flask import render_template, request, jsonify
import tkinter as tk
from tkinter import filedialog
from .core import (
    CONFIG_FILE,
    SECRETS_FILE,
    UPLOAD_FOLDER,
    _BASE_DIR,
    app,
    load_config,
    load_history,
    save_config,
)
from .translate import (
    GROQ_CHAT_MODELS,
)


@app.route('/')
def home():
    """Simple 3-step dubbing page."""
    return render_template('home.html')


@app.route('/favicon.ico')
def favicon():
    from flask import send_from_directory
    return send_from_directory(os.path.join(_BASE_DIR, "static"), "dubber.ico", mimetype="image/x-icon")


@app.route('/pro')
def index():
    """Full tool page with every option and the extra tools."""
    return render_template('index.html')


@app.route('/api/recent')
def api_recent():
    """Recent projects, newest first, with the playable result file if it still exists."""
    items = []
    for h in reversed(load_history()[-30:]):
        folder = h.get("output_path", "")
        video = srt = ""
        if folder and os.path.isdir(folder):
            for name in ("05_video_khmer_voice_subs.mp4", "04_video_khmer_voice.mp4"):
                if os.path.isfile(os.path.join(folder, name)):
                    video = os.path.join(folder, name)
                    break
            if os.path.isfile(os.path.join(folder, "02_khmer.srt")):
                srt = os.path.join(folder, "02_khmer.srt")
        items.append({
            "name": h.get("filename", ""),
            "date": h.get("date", ""),
            "lines": h.get("segments", 0),
            "folder": folder,
            "folder_exists": bool(folder and os.path.isdir(folder)),
            "video": video,
            "srt": srt,
            "recordable": bool(folder and os.path.isfile(os.path.join(folder, "job.json"))),
        })
    return jsonify({"success": True, "items": items})

@app.route('/history')
def get_history():
    return jsonify(load_history())

@app.route('/select-folder')
def select_folder():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder_selected = filedialog.askdirectory(title="Select Output Folder")
    root.destroy()
    if folder_selected:
        return jsonify({"success": True, "path": folder_selected})
    return jsonify({"success": False, "path": ""})

@app.route('/studio')
def studio():
    return render_template('studio.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')


@app.route('/download-file')
def download_file():
    from flask import send_file
    path = request.args.get('path', '')
    if not path or not os.path.exists(path):
        return jsonify({"success": False, "error": "File not found"}), 404
    return send_file(path, as_attachment=True)


@app.route('/stream-video')
def stream_video():
    """Stream video file for HTML5 in-browser video playback."""
    path = request.args.get('path', '')
    if not path or not os.path.exists(path):
        return jsonify({"success": False, "error": "Video file not found"}), 404
    from flask import send_file
    return send_file(path, mimetype='video/mp4')


@app.route('/logo-preview')
def logo_preview():
    """Serve active watermark logo image."""
    cfg = load_config()
    path = request.args.get('path') or cfg.get('logo_path', '')
    if path and os.path.exists(path):
        ext = os.path.splitext(path)[1].lower()
        mime = 'image/png' if ext == '.png' else ('image/jpeg' if ext in ('.jpg', '.jpeg') else 'image/webp')
        from flask import send_file
        return send_file(path, mimetype=mime)
    return jsonify({"error": "No logo"}), 404


@app.route('/api/upload-logo', methods=['POST'])
def upload_logo():
    """Save persistent watermark logo image."""
    if 'logo' not in request.files:
        return jsonify({"success": False, "error": "No logo file provided"}), 400
    logo_file = request.files['logo']
    if not logo_file or not logo_file.filename:
        return jsonify({"success": False, "error": "Empty filename"}), 400
    ext = os.path.splitext(logo_file.filename)[1].lower()
    if ext not in ('.png', '.jpg', '.jpeg', '.webp'):
        return jsonify({"success": False, "error": "Please upload a PNG, JPG, or WEBP image"}), 400
    dest = os.path.abspath(os.path.join(UPLOAD_FOLDER, f"channel_logo{ext}"))
    logo_file.save(dest)
    save_config({
        "logo_path": dest,
        "logo_enabled": True
    })
    return jsonify({
        "success": True,
        "logo_path": dest,
        "logo_url": "/logo-preview"
    })


@app.route('/open-folder', methods=['POST'])
def open_folder():
    """Open folder in Windows Explorer."""
    data = request.json or {}
    path = data.get('path', '')
    if path and os.path.exists(path):
        import subprocess
        try:
            target = os.path.normpath(path)
            if os.path.isfile(target):
                subprocess.Popen(f'explorer /select,"{target}"')
            else:
                subprocess.Popen(f'explorer "{target}"')
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
    return jsonify({"success": False, "error": "Path not found"}), 404


@app.route('/api/glossary', methods=['GET'])
def get_glossary():
    """Return dictionary of character name and term mappings."""
    cfg = load_config()
    return jsonify({"success": True, "glossary": cfg.get("glossary", {})})


@app.route('/api/glossary', methods=['POST'])
def save_glossary():
    """Save updated glossary dictionary."""
    data = request.json or {}
    glossary = data.get("glossary", {})
    if not isinstance(glossary, dict):
        return jsonify({"success": False, "error": "Glossary must be a dictionary"}), 400
    save_config({"glossary": glossary})
    return jsonify({"success": True, "glossary": glossary})


@app.route('/config', methods=['GET'])
def get_config():
    cfg = load_config()
    key_groq = (cfg.get("GROQ_API_KEY") or "").strip()
    key_openai = (cfg.get("OPENAI_API_KEY") or "").strip()
    key_gemini = (cfg.get("GEMINI_API_KEY") or "").strip()
    key_kiri = (cfg.get("KIRI_API_KEY") or "").strip()
    key_eleven = (cfg.get("ELEVENLABS_API_KEY") or "").strip()

    safe = {
        "default_asr": cfg.get("default_asr", "khmer-ft"),
        "default_translate_engine": cfg.get("default_translate_engine", "auto"),
        "default_translate_style": cfg.get("default_translate_style", "recap"),
        "default_dubbing_mode": cfg.get("default_dubbing_mode", "duck"),
        "default_voice": cfg.get("default_voice", "narrator_female"),
        "has_groq_key": bool(key_groq),
        "groq_key_hint": ("..." + key_groq[-4:]) if key_groq else "",
        "has_openai_key": bool(key_openai),
        "openai_key_hint": ("..." + key_openai[-4:]) if key_openai else "",
        "has_gemini_key": bool(key_gemini),
        "gemini_key_hint": ("..." + key_gemini[-4:]) if key_gemini else "",
        "has_kiri_key": bool(key_kiri),
        "kiri_key_hint": ("..." + key_kiri[-4:]) if key_kiri else "",
        "has_elevenlabs_key": bool(key_eleven),
        "elevenlabs_key_hint": ("..." + key_eleven[-4:]) if key_eleven else "",
        "tts_voice": cfg.get("tts_voice", "narrator_female"),
        "tts_rate": cfg.get("tts_rate", "-5%"),
        "sub_color": cfg.get("sub_color", "yellow"),
        "sub_size": cfg.get("sub_size", 22),
        "studio_cinema_dsp": cfg.get("studio_cinema_dsp", True),
        "role_voice_map": cfg.get("role_voice_map", {}),
        "logo_path": cfg.get("logo_path", ""),
        "logo_exists": bool(cfg.get("logo_path") and os.path.exists(cfg.get("logo_path"))),
        "logo_enabled": cfg.get("logo_enabled", False),
        "logo_pos": cfg.get("logo_pos", "top-right"),
        "logo_size": cfg.get("logo_size", "medium"),
        "logo_opacity": cfg.get("logo_opacity", 0.85),
        "glossary": cfg.get("glossary", {}),
        "default_aspect_ratio": cfg.get("default_aspect_ratio", "original"),
        "download_cookies_file": cfg.get("download_cookies_file", ""),
        "config_path": CONFIG_FILE,
        "config_exists": os.path.exists(CONFIG_FILE),
        "keys_path": SECRETS_FILE,
    }
    return jsonify(safe)


@app.route('/config', methods=['POST'])
def set_config():
    data = request.json or {}
    updates = {}
    for key in (
        "GROQ_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "KIRI_API_KEY",
        "ELEVENLABS_API_KEY", "default_asr", "default_translate_engine",
        "default_translate_style", "default_dubbing_mode", "default_voice",
        "tts_voice", "tts_rate", "sub_color", "sub_size", "studio_cinema_dsp",
        "role_voice_map", "logo_path", "logo_enabled", "logo_pos", "logo_size",
        "logo_opacity", "glossary", "default_aspect_ratio", "download_cookies_file"
    ):
        if key in data:
            updates[key] = data[key]
    try:
        cfg = save_config(updates)
        return jsonify({
            "success": True,
            "message": "Config saved successfully",
            "has_gemini_key": bool((cfg.get("GEMINI_API_KEY") or "").strip()),
            "has_openai_key": bool((cfg.get("OPENAI_API_KEY") or "").strip()),
            "has_groq_key": bool((cfg.get("GROQ_API_KEY") or "").strip()),
            "has_kiri_key": bool((cfg.get("KIRI_API_KEY") or "").strip()),
            "has_elevenlabs_key": bool((cfg.get("ELEVENLABS_API_KEY") or "").strip()),
            "config_path": CONFIG_FILE,
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/test-api-key', methods=['POST'])
def test_api_key():
    """Verify an API key works before starting a video dubbing project."""
    data = request.json or {}
    key_type = data.get('type')  # 'gemini', 'openai', 'groq', 'kiri', 'elevenlabs'
    key_val = (data.get('key') or "").strip()

    cfg = load_config()
    if not key_val:
        if key_type == 'gemini': key_val = (cfg.get("GEMINI_API_KEY") or "").strip()
        elif key_type == 'openai': key_val = (cfg.get("OPENAI_API_KEY") or "").strip()
        elif key_type == 'groq': key_val = (cfg.get("GROQ_API_KEY") or "").strip()
        elif key_type == 'kiri': key_val = (cfg.get("KIRI_API_KEY") or "").strip()
        elif key_type == 'elevenlabs': key_val = (cfg.get("ELEVENLABS_API_KEY") or "").strip()

    if not key_val:
        return jsonify({"success": False, "error": f"{str(key_type).upper()} API Key is empty"}), 400

    try:
        if key_type == 'gemini':
            import requests
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key_val}"
            payload = {"contents": [{"parts": [{"text": "Say OK in Khmer"}]}]}
            r = requests.post(url, json=payload, timeout=12)
            if r.status_code == 200:
                return jsonify({"success": True, "message": "Google Gemini Connected! (Active & Ready)"})
            url15 = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key_val}"
            r15 = requests.post(url15, json=payload, timeout=12)
            if r15.status_code == 200:
                return jsonify({"success": True, "message": "Google Gemini 1.5 Connected! (Active & Ready)"})
            return jsonify({"success": False, "error": f"Gemini error {r.status_code}: {r.text[:120]}"}), 400

        elif key_type == 'groq':
            try:
                from groq import Groq
                client = Groq(api_key=key_val)
                resp = client.chat.completions.create(
                    model=GROQ_CHAT_MODELS[0],
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5
                )
                return jsonify({"success": True, "message": "Groq Connected! (Active & Ready)"})
            except Exception:
                import requests
                headers = {"Authorization": f"Bearer {key_val}", "Content-Type": "application/json"}
                r = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json={"model": GROQ_CHAT_MODELS[0], "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 5},
                    timeout=12
                )
                if r.status_code == 200:
                    return jsonify({"success": True, "message": "Groq Connected! (Active & Ready)"})
                return jsonify({"success": False, "error": f"Groq error {r.status_code}: {r.text[:120]}"}), 400

        elif key_type == 'openai':
            try:
                from openai import OpenAI
                client = OpenAI(api_key=key_val)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5
                )
                return jsonify({"success": True, "message": "OpenAI Connected! (Active & Ready)"})
            except Exception:
                import requests
                headers = {"Authorization": f"Bearer {key_val}", "Content-Type": "application/json"}
                r = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 5},
                    timeout=12
                )
                if r.status_code == 200:
                    return jsonify({"success": True, "message": "OpenAI Connected! (Active & Ready)"})
                return jsonify({"success": False, "error": f"OpenAI error {r.status_code}: {r.text[:120]}"}), 400

        elif key_type == 'kiri':
            if len(key_val) > 10:
                return jsonify({"success": True, "message": "Kiri API Key saved & ready!"})
            return jsonify({"success": False, "error": "Kiri key seems too short"}), 400

        elif key_type == 'elevenlabs':
            import requests
            headers = {"xi-api-key": key_val}
            r = requests.get("https://api.elevenlabs.io/v1/user", headers=headers, timeout=12)
            if r.status_code == 200:
                user_info = r.json()
                sub = user_info.get("subscription", {})
                used = sub.get("character_count", 0)
                limit = sub.get("character_limit", 0)
                tier = sub.get("tier", "Free")
                return jsonify({
                    "success": True,
                    "message": f"ElevenLabs Connected! Tier: {tier} ({used:,} / {limit:,} chars)"
                })
            else:
                err_text = "Connection failed"
                try:
                    err_text = r.json().get("detail", {}).get("message", r.text)
                except Exception:
                    pass
                return jsonify({"success": False, "error": f"ElevenLabs error: {err_text}"}), 400

        return jsonify({"success": False, "error": "Unknown key type"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)[:180]}), 400
