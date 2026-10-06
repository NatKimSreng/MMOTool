"""Builds "DAI Dubber.exe" (small launcher) next to app.py.  Run:  python build_exe.py"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
pythonw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")

work = tempfile.mkdtemp(prefix="dubber_build_")
src = open(os.path.join(HERE, "launcher.py"), encoding="utf-8").read().replace("__PYTHONW__", pythonw)
launcher = os.path.join(work, "launcher.py")
open(launcher, "w", encoding="utf-8").write(src)

subprocess.check_call([
    sys.executable, "-m", "PyInstaller", "--noconfirm", "--onefile", "--noconsole",
    "--name", "DAI Dubber", "--icon", os.path.join(HERE, "static", "dubber.ico"),
    "--distpath", HERE, "--workpath", os.path.join(work, "build"), "--specpath", work,
    launcher,
])
shutil.rmtree(work, ignore_errors=True)
print("Built:", os.path.join(HERE, "DAI Dubber.exe"), "->", pythonw)
