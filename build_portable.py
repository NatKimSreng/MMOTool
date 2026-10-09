"""Builds a portable "DAI Dubber" folder + zip (bundled Python, ffmpeg, models). Run:  python build_portable.py

No videos, outputs, uploads, history, settings or API keys are included.
"""
import fnmatch
import glob
import os
import shutil
import stat
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(os.path.dirname(HERE), "DAI Dubber Portable")
STAGE = os.path.join(OUT_ROOT, "DAI Dubber")
ZIP_PATH = os.path.join(os.path.dirname(HERE), "DAI Dubber Portable.zip")

PYTHON_DIR = os.path.dirname(sys.executable)
# winget's ffmpeg on PATH is only a link; copy the real bin\ folder (ffmpeg.exe + ffprobe.exe)
FFMPEG_BIN = os.path.dirname(glob.glob(os.path.expandvars(
    r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*\*\bin\ffmpeg.exe"))[0])
USER_CACHE = os.path.join(os.path.expanduser("~"), ".cache")

APP_FILES = ["DAI Dubber.exe", "app.py", "desktop.py", "launcher.py", "build_exe.py",
             "build_portable.py", "requirements.txt"]
APP_DIRS = ["dubber", "templates", "static", "voices", "tools"]
SKIP = ["__pycache__", "*.pyc", "*.bak*", "*.mp4", "*.mkv", "*.mov", "*.webm", "*.avi", "*.part"]
RUNTIME = {"logs", "uploads", "outputs", "downloads", "cloned_voices", "config.json", "history.json",
           "tasks.json", "batch_queue.json", "cloned_voices.json"}
SKIP_TOOLS =["wheels", "hf-cache"]   # pip install leftovers / runtime cache


def ignore(skip):
    def _ignore(_dir, names):
        return {n for n in names if any(fnmatch.fnmatch(n, p) for p in skip)}
    return _ignore


def copytree(src, dst, skip=SKIP):
    print("copy", src)
    shutil.copytree(src, dst, ignore=ignore(skip), dirs_exist_ok=True)


def _force_remove(func, path, _exc):
    """git marks its object files read-only, which makes rmtree fail on Windows."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def main():
    if "--zip-only" not in sys.argv:
        stage()
    if "--no-zip" not in sys.argv:
        make_zip()


def stage():
    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE, onerror=_force_remove)
    os.makedirs(STAGE)

    for f in APP_FILES:
        shutil.copy2(os.path.join(HERE, f), STAGE)
    for d in APP_DIRS:
        copytree(os.path.join(HERE, d), os.path.join(STAGE, d),
                 SKIP + (SKIP_TOOLS if d == "tools" else []))

    copytree(PYTHON_DIR, os.path.join(STAGE, "python"), ["__pycache__", "Doc"])
    copytree(FFMPEG_BIN, os.path.join(STAGE, "ffmpeg"), [])
    for c in ("whisper", "huggingface"):
        if os.path.isdir(os.path.join(USER_CACHE, c)):
            copytree(os.path.join(USER_CACHE, c), os.path.join(STAGE, "cache", c), [])


def make_zip():
    print("zip ->", ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for root, dirs, files in os.walk(STAGE):
            if root == STAGE:   # files a test run of the staged app may have made
                dirs[:] = [d for d in dirs if d not in RUNTIME]
                files = [f for f in files if f not in RUNTIME]
            dirs[:] = [d for d in dirs if d != "__pycache__" or root.startswith(os.path.join(STAGE, "python"))
                       or root.startswith(os.path.join(STAGE, "tools", "vc-env"))]
            for f in files:
                full = os.path.join(root, f)
                z.write(full, os.path.relpath(full, OUT_ROOT))
    print("Done:", ZIP_PATH, f"{os.path.getsize(ZIP_PATH) / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
