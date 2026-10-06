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


def main():
    here = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__))
    script = os.path.join(here, "desktop.py")
    if not os.path.isfile(script):
        ctypes.windll.user32.MessageBoxW(None, f"desktop.py not found next to the exe:\n{here}", "DAI Dubber", 0x10)
        return

    candidates = [
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
    subprocess.Popen([pyw, script], cwd=here, close_fds=True,
                     creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP)


if __name__ == "__main__":
    main()
