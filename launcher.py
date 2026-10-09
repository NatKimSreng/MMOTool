"""
"DAI Dubber.exe" — double-click to open the app in its own window.

Tiny launcher: starts desktop.py (next to this exe) with your installed Python,
so the exe stays small and uses the same packages as the app.
Build:  python build_exe.py
"""
import ctypes
import os
import shutil
import subprocess
import sys

# filled in by build_exe.py with the Python that has the app's packages
BUILD_PYTHONW = r"__PYTHONW__"


def _fix_vc_env(here, python_dir):
    """Point tools\\vc-env (a venv) at the bundled Python — venvs store an absolute path."""
    cfg = os.path.join(here, "tools", "vc-env", "pyvenv.cfg")
    if not os.path.isfile(cfg):
        return
    try:
        lines = open(cfg, encoding="utf-8").read().splitlines()
        new = []
        for line in lines:
            key = line.split("=", 1)[0].strip().lower()
            if key == "home":
                line = f"home = {python_dir}"
            elif key == "executable":
                line = f"executable = {os.path.join(python_dir, 'python.exe')}"
            new.append(line)
        if new != lines:
            open(cfg, "w", encoding="utf-8").write("\n".join(new) + "\n")
    except OSError:
        pass


def main():
    here = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__))
    script = os.path.join(here, "desktop.py")
    if not os.path.isfile(script):
        ctypes.windll.user32.MessageBoxW(None, f"desktop.py not found next to the exe:\n{here}", "DAI Dubber", 0x10)
        return

    # portable package: python\, ffmpeg\ and cache\ ship next to the exe
    portable_pyw = os.path.join(here, "python", "pythonw.exe")
    env = os.environ.copy()
    ffmpeg_dir = os.path.join(here, "ffmpeg")
    if os.path.isdir(ffmpeg_dir):
        env["PATH"] = ffmpeg_dir + os.pathsep + env.get("PATH", "")
    cache_dir = os.path.join(here, "cache")
    if os.path.isdir(cache_dir):
        env["XDG_CACHE_HOME"] = cache_dir   # whisper + huggingface models
    if os.path.isfile(portable_pyw):
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
        _fix_vc_env(here, os.path.dirname(portable_pyw))

    candidates = [
        portable_pyw,
        BUILD_PYTHONW,
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Python", "Python311", "pythonw.exe"),
        shutil.which("pythonw"),
    ]
    pyw = next((p for p in candidates if p and os.path.isfile(p)), None)
    if not pyw:
        ctypes.windll.user32.MessageBoxW(
            None, "Python 3.11 was not found.\nInstall Python 3.11 and the app's packages first.", "DAI Dubber", 0x10)
        return

    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    subprocess.Popen([pyw, script], cwd=here, close_fds=True, env=env,
                     creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP)


if __name__ == "__main__":
    main()
