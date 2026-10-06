"""
DAI Dubber as a desktop app: runs the Flask app in the background and shows it
in its own window (Windows' built-in Edge WebView2) — no browser, no console.

Started by "DAI Dubber.exe" (launcher.py), or directly:  pythonw desktop.py
"""
import ctypes
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request

APP_NAME = "DAI Dubber"
HERE = os.path.dirname(os.path.abspath(__file__))
ICON = os.path.join(HERE, "static", "dubber.ico")
LOG_DIR = os.path.join(HERE, "logs")
PREFERRED_PORT = 5000
os.chdir(HERE)
sys.path.insert(0, HERE)

# ── no console: send output to a log file ─────────────────────
os.makedirs(LOG_DIR, exist_ok=True)
_log = open(os.path.join(LOG_DIR, "dubber.log"), "a", encoding="utf-8", buffering=1)
if sys.stdout is None or not sys.stdout.isatty():
    sys.stdout = _log
    sys.stderr = _log
print(f"\n===== {APP_NAME} started {time.strftime('%Y-%m-%d %H:%M:%S')} =====")

# ── ffmpeg / yt-dlp / explorer helpers must not flash black windows ──
if os.name == "nt":
    _CREATE_NO_WINDOW = 0x08000000
    _orig_popen_init = subprocess.Popen.__init__

    def _quiet_popen_init(self, *args, **kwargs):
        if not kwargs.get("creationflags"):
            kwargs["creationflags"] = _CREATE_NO_WINDOW
        _orig_popen_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = _quiet_popen_init


def _message(text, title=APP_NAME, flags=0x10):
    try:
        return ctypes.windll.user32.MessageBoxW(None, text, title, flags)
    except Exception:
        print(text)


def _is_our_app(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/progress", timeout=1.5) as r:
            return b"is_processing" in r.read()
    except Exception:
        return False


def _port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _set_window_icon(title):
    """Use our icon in the title bar / taskbar instead of Python's."""
    if os.name != "nt" or not os.path.isfile(ICON):
        return
    user32 = ctypes.windll.user32
    for _ in range(50):
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            break
        time.sleep(0.1)
    else:
        return
    LR_LOADFROMFILE, IMAGE_ICON, WM_SETICON = 0x10, 1, 0x0080
    big = user32.LoadImageW(None, ICON, IMAGE_ICON, 256, 256, LR_LOADFROMFILE)
    small = user32.LoadImageW(None, ICON, IMAGE_ICON, 32, 32, LR_LOADFROMFILE)
    if big:
        user32.SendMessageW(hwnd, WM_SETICON, 1, big)
    if small:
        user32.SendMessageW(hwnd, WM_SETICON, 0, small)


def _allow_microphone(window):
    """'Record my voice' needs the mic: allow it for this app's own local pages only."""
    try:
        from System import Action
        form = window.native
        from Microsoft.Web.WebView2.Core import CoreWebView2PermissionKind, CoreWebView2PermissionState

        def on_permission(sender, args):
            try:
                local = str(args.Uri).startswith(("http://127.0.0.1:", "http://localhost:"))
                if local and args.PermissionKind == CoreWebView2PermissionKind.Microphone:
                    args.State = CoreWebView2PermissionState.Allow
            except Exception as e:
                print(f"[mic permission]: {e}")

        def hook():
            try:
                form.webview.CoreWebView2.PermissionRequested += on_permission
                print("[mic] microphone allowed for the app's own pages")
            except Exception as e:
                print(f"[mic] hook failed: {e}")

        form.Invoke(Action(hook))
    except Exception as e:
        print(f"[mic] could not set permission handler (Windows will ask instead): {e}")


def main():
    if os.name == "nt":
        try:  # own taskbar group + icon, not "Python"
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("DAI.Dubber.Desktop")
        except Exception:
            pass

    try:
        import webview
    except ImportError:
        _message("pywebview is missing.\n\nRun this once in a terminal:\n  pip install pywebview")
        return

    # Already running (window or old console version)? Just open another window on it.
    own_server = False
    flask_app = None
    if _is_our_app(PREFERRED_PORT):
        port = PREFERRED_PORT
    else:
        port = PREFERRED_PORT if _port_free(PREFERRED_PORT) else _free_port()
        try:
            import app as dubber
        except Exception as e:
            import traceback
            traceback.print_exc()
            _message(f"The app could not start:\n\n{e}\n\nDetails are in logs\\dubber.log")
            return
        flask_app = dubber
        threading.Thread(
            target=lambda: dubber.app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False, threaded=True),
            daemon=True,
        ).start()
        own_server = True
        for _ in range(100):
            if _is_our_app(port):
                break
            time.sleep(0.1)
        else:
            _message("The app's server did not start. Details are in logs\\dubber.log")
            return

    url = f"http://127.0.0.1:{port}/"
    webview.settings["ALLOW_DOWNLOADS"] = True          # "Download video / subtitles" buttons
    webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True

    def add_window(page_url, main=False):
        win = webview.create_window(
            APP_NAME, page_url, width=1320 if main else 1180, height=900 if main else 820,
            min_size=(420, 600), background_color="#0e0e13", text_select=True,
        )

        def on_closing():
            # closing the LAST window stops the server — warn if jobs are still running
            if not own_server or flask_app is None or len(webview.windows) > 1:
                return True
            running = flask_app.any_task_running()
            if not running:
                return True
            return win.create_confirmation_dialog(
                "Jobs still running",
                f"{running} job(s) still running. Closing the app pauses them — everything done so far is "
                "saved, and you can press Resume under Tasks next time you open the app.\n\nClose anyway?",
            )

        win.events.closing += on_closing
        win.events.shown += lambda: threading.Thread(target=_set_window_icon, args=(APP_NAME,), daemon=True).start()
        if not main:
            hooked = []

            def on_extra_loaded():
                if not hooked:
                    hooked.append(True)
                    _allow_microphone(win)

            win.events.loaded += on_extra_loaded
        return win

    window = add_window(url, main=True)
    if own_server and flask_app is not None:
        # "New window" in the Tasks panel: watch one job in one window, start another in the next
        flask_app.WINDOW_HOOKS["new_window"] = lambda path: add_window(url.rstrip("/") + path)

    mic_hooked = []

    def on_loaded():
        if not mic_hooked:
            mic_hooked.append(True)
            _allow_microphone(window)
            if os.environ.get("DUBBER_MIC_SELFTEST"):
                def _selftest():
                    time.sleep(1)
                    window.evaluate_js(
                        "window.__mic='pending'; navigator.mediaDevices.getUserMedia({audio:true})"
                        ".then(s => { s.getTracks().forEach(t => t.stop()); window.__mic='MIC_OK'; })"
                        ".catch(e => { window.__mic='MIC_FAIL ' + e.name + ' ' + e.message; }); 1")
                    for _ in range(20):
                        time.sleep(0.5)
                        r = window.evaluate_js("window.__mic")
                        if r != "pending":
                            break
                    print(f"[mic selftest] {r}")
                    window.destroy()
                threading.Thread(target=_selftest, daemon=True).start()

    window.events.loaded += on_loaded

    storage = os.path.join(os.environ.get("APPDATA", HERE), APP_NAME, "webview")
    os.makedirs(storage, exist_ok=True)
    # private_mode=False keeps the page's remembered choices between runs
    webview.start(private_mode=False, storage_path=storage)
    print(f"===== {APP_NAME} closed {time.strftime('%H:%M:%S')} =====")
    os._exit(0)  # stop the background server and any worker threads


if __name__ == "__main__":
    main()
