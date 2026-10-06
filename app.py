"""
DAI Dubber — start the app:  python app.py   (or "DAI Dubber.exe" / desktop.py for the window version)

The code lives in the dubber/ folder, one file per area (see dubber/__init__.py).
Importing this module registers every page and job with the Flask app.
"""
import threading
import webbrowser

# each module adds its routes to the Flask app when imported (order = lower layers first)
from dubber import (  # noqa: F401
    core, tasks, media, asr, translate, voices, characters, record, pipeline, batch_queue, tools, cleanup, pages,
)
from dubber.core import app, load_config, save_config, progress_state  # noqa: F401  (used by desktop.py / tools)
from dubber.tasks import any_task_running, WINDOW_HOOKS  # noqa: F401


if __name__ == '__main__':
    def open_browser():
        webbrowser.open("http://127.0.0.1:5000")

    threading.Thread(target=open_browser).start()
    app.run(debug=False, port=5000)
