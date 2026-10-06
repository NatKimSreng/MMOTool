"""
Background tasks: every job runs as a task with its own progress, several at once, saved to
tasks.json so paused / interrupted jobs can be resumed. Also the GPU turn-taking and the Tasks API.
"""
import functools
import os
import subprocess
import threading
import json
from flask import request, jsonify
import uuid
import time
from .core import (
    _BASE_DIR,
    app,
    load_config,
    progress_state,
    save_config,
)


# ═══════════════════════════════════════════════════════════
# BACKGROUND TASKS — every job runs as its own task: several can run side by side,
# each keeps its own progress, and the list is saved to tasks.json so a job that was
# paused (or cut off when the app closed) can be resumed later.
# ═══════════════════════════════════════════════════════════
TASKS_FILE = os.path.join(_BASE_DIR, "tasks.json")
_TASKS = {}                       # id -> TaskState
_TASKS_LOCK = threading.RLock()
_TASK_LOCAL = threading.local()   # .state = TaskState of the task this thread is running
_TASKS_DIRTY = threading.Event()
_SLOT_COND = threading.Condition()
_SLOT_RUNNING = set()             # task ids holding a run slot
_SLOT_WAITING = []                # task ids waiting for a slot, first come first served
_TASK_KEEP = 60                   # finished tasks kept in the list
_ACTIVE_STATES = ("queued", "running")
# job functions a task may run: name -> function (see register_task_runner)
_TASK_RUNNERS = {}
# runners that save checkpoints and continue where they stopped
_RESUMABLE_RUNNERS = set()


def register_task_runner(fn, resumable=False):
    """Allow tasks to run fn (tasks.json stores the function's name)."""
    _TASK_RUNNERS[fn.__name__] = fn
    if resumable:
        _RESUMABLE_RUNNERS.add(fn.__name__)
    return fn


class TaskStopped(BaseException):
    """Raised in a task's thread after Pause/Stop was pressed. BaseException so the
    jobs' own `except Exception` handlers don't swallow it."""


class TaskState(dict):
    """Progress dict of one task. Job code writes status/percent into it like it always
    wrote into progress_state; once Pause/Stop is pressed the next such write raises TaskStopped."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.stop_requested = False
        self.procs = []           # ffmpeg / worker processes started by this task (killed on stop)

    def __setitem__(self, key, value):
        if self.stop_requested and key in ("status", "percent"):
            raise TaskStopped()
        super().__setitem__(key, value)
        _TASKS_DIRTY.set()

    def update(self, *a, **kw):
        for k, v in dict(*a, **kw).items():
            self[k] = v

    def force(self, **kw):
        """Write without the stop check (final state after the job ended)."""
        for k, v in kw.items():
            dict.__setitem__(self, k, v)
        _TASKS_DIRTY.set()


def _task_state():
    """Progress dict for the job running in this thread (the old global one outside tasks)."""
    st = getattr(_TASK_LOCAL, "state", None)
    return st if st is not None else progress_state


def _set_task_param(key, value):
    """Remember something in the running task's saved params (used when it is resumed)."""
    st = getattr(_TASK_LOCAL, "state", None)
    if st is not None and isinstance(st.get("params"), dict):
        st["params"][key] = value
        _TASKS_DIRTY.set()


# ffmpeg / voice workers started inside a task belong to it, so Pause/Stop can end them at once
_orig_popen_init = subprocess.Popen.__init__


def _task_popen_init(self, *args, **kwargs):
    _orig_popen_init(self, *args, **kwargs)
    st = getattr(_TASK_LOCAL, "state", None)
    if st is not None:
        st.procs = [p for p in st.procs if p.poll() is None] + [self]
        if st.stop_requested:
            try:
                self.kill()
            except Exception:
                pass


subprocess.Popen.__init__ = _task_popen_init


def _task_public(st):
    d = {k: v for k, v in st.items() if not str(k).startswith("_")}
    d["stop_requested"] = bool(getattr(st, "stop_requested", False))
    if d.get("state") == "running" and d.get("waiting_for_review"):
        d["state"] = "review"
    return d


def _save_tasks():
    with _TASKS_LOCK:
        rows = [dict(st) for st in _TASKS.values()]
    try:
        tmp = TASKS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1, default=str)
        os.replace(tmp, TASKS_FILE)
    except Exception as e:
        print(f"[tasks] save failed: {e}")


def _tasks_saver():
    while True:
        _TASKS_DIRTY.wait()
        time.sleep(2)
        _TASKS_DIRTY.clear()
        _save_tasks()


def _load_tasks():
    """Read tasks.json. Jobs that were still running when the app closed become 'interrupted'."""
    try:
        with open(TASKS_FILE, encoding="utf-8") as f:
            rows = json.load(f)
    except Exception:
        return
    for rec in rows if isinstance(rows, list) else []:
        if not isinstance(rec, dict) or not rec.get("id"):
            continue
        st = TaskState(rec)
        if st.get("state") in _ACTIVE_STATES or st.get("state") == "review":
            st.force(
                state="interrupted", is_processing=False, waiting_for_review=False,
                review_job_id="", review_segments=[],
                status=("Stopped when the app closed — press Resume to continue where it left off."
                        if st.get("resumable") else "Stopped when the app closed."),
            )
        _TASKS[st["id"]] = st


def _prune_tasks():
    with _TASKS_LOCK:
        done = [t for t in _TASKS.values() if t.get("state") not in _ACTIVE_STATES]
        done.sort(key=lambda t: t.get("created_at") or 0)
        for t in done[:max(0, len(done) - _TASK_KEEP)]:
            _TASKS.pop(t["id"], None)


def _max_parallel_tasks():
    try:
        return max(1, min(6, int(load_config().get("max_parallel_tasks", 2))))
    except (TypeError, ValueError):
        return 2


def _acquire_slot(st):
    """Wait (first come, first served) until fewer than max_parallel_tasks jobs are running."""
    tid = st["id"]
    with _SLOT_COND:
        _SLOT_WAITING.append(tid)
    try:
        while True:
            with _SLOT_COND:
                if _SLOT_WAITING and _SLOT_WAITING[0] == tid and len(_SLOT_RUNNING) < _max_parallel_tasks():
                    _SLOT_WAITING.remove(tid)
                    _SLOT_RUNNING.add(tid)
                    return
                ahead = _SLOT_WAITING.index(tid)
                running = len(_SLOT_RUNNING)
            st["status"] = (f"Waiting — {running} task{'s' if running != 1 else ''} running"
                            + (f", {ahead} ahead in line" if ahead else "") + ". Starts automatically.")
            with _SLOT_COND:
                _SLOT_COND.wait(timeout=2)
    finally:
        with _SLOT_COND:
            if tid in _SLOT_WAITING:
                _SLOT_WAITING.remove(tid)
            _SLOT_COND.notify_all()


def _release_slot(tid):
    with _SLOT_COND:
        _SLOT_RUNNING.discard(tid)
        _SLOT_COND.notify_all()


def _run_task(st):
    """Run one task in the calling thread: wait for a slot, run its job, record how it ended."""
    _TASK_LOCAL.state = st
    final = "done"
    try:
        _acquire_slot(st)
        try:
            st["state"] = "running"
            st["started_at"] = time.time()
            st["status"] = "Starting..."
            fn = _TASK_RUNNERS.get(st.get("runner"))
            if fn is None:
                raise RuntimeError(f"Unknown task type: {st.get('runner')}")
            fn(**(st.get("params") or {}))
            if str(dict.get(st, "status", "")).startswith("Error"):
                final = "error"
        finally:
            _release_slot(st["id"])
    except TaskStopped:
        final = "stopped"
    except Exception as e:
        import traceback
        traceback.print_exc()
        st.force(status=f"Error: {e}")
        final = "error"
    finally:
        _TASK_LOCAL.state = None
        for p in st.procs:
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass
        st.procs = []
        extra = {}
        if final == "stopped":
            extra["status"] = ("Paused — press Resume to continue where it left off."
                               if st.get("resumable") else "Stopped.")
            extra["percent"] = dict.get(st, "percent", 0)
        elif final == "done":
            extra["percent"] = 100
        st.stop_requested = False
        st.force(state=final, is_processing=False, finished_at=time.time(),
                 waiting_for_review=False, review_job_id="", review_segments=[], **extra)
        _save_tasks()
    return final


def _new_task(runner, params, name, kind="dub"):
    tid = uuid.uuid4().hex[:10]
    st = TaskState(
        id=tid, kind=kind, name=name or kind, runner=runner, params=params,
        resumable=runner in _RESUMABLE_RUNNERS, state="queued", is_processing=True,
        percent=0, status="Queued…", created_at=time.time(), started_at=None, finished_at=None,
        result_path="", final_video="", final_srt="", final_audio="", job_dir="",
        waiting_for_review=False, review_job_id="", review_segments=[],
    )
    with _TASKS_LOCK:
        _TASKS[tid] = st
    _prune_tasks()
    _save_tasks()
    return st


def _start_task(runner, params, name, kind="dub"):
    """Create a task and run it on its own background thread. Returns the TaskState."""
    st = _new_task(runner, params, name, kind)
    threading.Thread(target=_run_task, args=(st,), daemon=True, name=f"task-{st['id']}").start()
    return st


def _restart_task(st, inline=False):
    """Resume a paused / interrupted / failed task (or run a finished one again) with its saved params."""
    st.stop_requested = False
    st.force(state="queued", is_processing=True, status="Queued…", finished_at=None, started_at=None,
             waiting_for_review=False, review_job_id="", review_segments=[])
    if inline:
        return _run_task(st)
    threading.Thread(target=_run_task, args=(st,), daemon=True, name=f"task-{st['id']}").start()
    return st


def _stop_task(st):
    st.stop_requested = True
    for p in list(st.procs):
        if p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass
    ev = _ACTIVE_REVIEW_EVENTS.get(dict.get(st, "review_job_id", ""))
    if ev:
        ev.set()   # let a job waiting for script review wake up and stop
    with _SLOT_COND:
        _SLOT_COND.notify_all()


def _wait_for_user(event):
    """Wait with no time limit for the user (the script review). The job's run slot is lent
    to other jobs meanwhile and taken back afterwards. Pause / app close ends the wait."""
    st = getattr(_TASK_LOCAL, "state", None)
    if st is None:
        return event.wait(timeout=1800)   # not running as a task: old behaviour
    _release_slot(st["id"])
    event.wait()
    if not st.stop_requested:
        st["state"] = "queued"
        _acquire_slot(st)
        st["state"] = "running"
    return True


def any_task_running():
    with _TASKS_LOCK:
        return sum(1 for t in _TASKS.values() if t.get("state") in _ACTIVE_STATES)


_load_tasks()
threading.Thread(target=_tasks_saver, daemon=True, name="tasks-saver").start()


# ── one job at a time on the graphics card ─────────────────────
# The laptop GPU has little memory: two voice-conversion / Demucs / Whisper runs at once
# would crash with "out of memory". Jobs take turns for these steps only.
_GPU_LOCK = threading.RLock()
_GPU_USER = {"name": ""}


def _uses_gpu(label):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            st = _task_state()
            waited = not _GPU_LOCK.acquire(blocking=False)
            while waited:
                who = _GPU_USER["name"]
                st["status"] = (f"Waiting for the graphics card — {who} is using it. Continues automatically..."
                                if who else "Waiting for the graphics card...")
                if _GPU_LOCK.acquire(timeout=1.0):
                    break
            prev = _GPU_USER["name"]
            try:
                _GPU_USER["name"] = f"“{st.get('name')}”" if st.get("name") else "another job"
                if waited:
                    st["status"] = f"{label}..."
                return fn(*args, **kwargs)
            finally:
                _GPU_USER["name"] = prev
                _GPU_LOCK.release()
        return wrapper
    return deco


_ACTIVE_REVIEW_EVENTS = {}
_REVIEW_EDITS = {}

@app.route('/progress')
def progress():
    """Progress of one task (?task=<id>). No id → the newest task; 'none' → idle."""
    tid = request.args.get("task", "")
    st = None
    if tid and tid != "none":
        st = _TASKS.get(tid)
    elif not tid:
        with _TASKS_LOCK:
            active = [t for t in _TASKS.values() if t.get("state") in _ACTIVE_STATES]
            st = max(active, key=lambda t: t.get("created_at") or 0) if active else None
    if st is None:
        return jsonify({"percent": 0, "status": "Waiting...", "is_processing": False, "result_path": "",
                        "final_video": "", "final_srt": "", "final_audio": "", "job_dir": "",
                        "waiting_for_review": False, "review_job_id": "", "review_segments": []})
    return jsonify(_task_public(st))


@app.route('/api/tasks')
def api_tasks():
    """All tasks, newest first (review segments left out — they can be long)."""
    with _TASKS_LOCK:
        items = sorted(_TASKS.values(), key=lambda t: t.get("created_at") or 0, reverse=True)
        rows = []
        for t in items:
            d = _task_public(t)
            d.pop("review_segments", None)
            d.pop("params", None)
            d["input_ok"] = _task_input_exists(t)
            rows.append(d)
    return jsonify({"success": True, "tasks": rows, "running": any_task_running(),
                    "max_parallel": _max_parallel_tasks()})


def _task_input_exists(st):
    params = st.get("params") or {}
    if st.get("runner") == "url_pipeline_task":
        return True
    for key in ("file_path", "file_paths", "srt_path", "job_dir"):
        v = params.get(key)
        if v:
            paths = v if isinstance(v, list) else [v]
            return all(os.path.exists(x) for x in paths)
    return True


@app.route('/api/tasks/<tid>/stop', methods=['POST'])
def api_task_stop(tid):
    st = _TASKS.get(tid)
    if not st:
        return jsonify({"success": False, "error": "Task not found"}), 404
    if st.get("state") not in _ACTIVE_STATES:
        return jsonify({"success": False, "error": "This task is not running"}), 400
    _stop_task(st)
    return jsonify({"success": True, "task_id": tid})


@app.route('/api/tasks/<tid>/resume', methods=['POST'])
def api_task_resume(tid):
    st = _TASKS.get(tid)
    if not st:
        return jsonify({"success": False, "error": "Task not found"}), 404
    if st.get("state") in _ACTIVE_STATES:
        return jsonify({"success": True, "task_id": tid})
    if not _task_input_exists(st):
        return jsonify({"success": False, "error": "The original video file is gone, so this task can't run again."}), 400
    _restart_task(st)
    return jsonify({"success": True, "task_id": tid})


@app.route('/api/tasks/<tid>/remove', methods=['POST'])
def api_task_remove(tid):
    st = _TASKS.get(tid)
    if st and st.get("state") in _ACTIVE_STATES:
        return jsonify({"success": False, "error": "Pause or stop the task first"}), 400
    with _TASKS_LOCK:
        _TASKS.pop(tid, None)
    _save_tasks()
    return jsonify({"success": True})


@app.route('/api/tasks/clear', methods=['POST'])
def api_tasks_clear():
    """Remove finished tasks from the list (paused/interrupted ones stay so they can be resumed)."""
    with _TASKS_LOCK:
        for t in list(_TASKS.values()):
            if t.get("state") == "done":
                _TASKS.pop(t["id"], None)
    _save_tasks()
    return jsonify({"success": True})


@app.route('/api/tasks/settings', methods=['POST'])
def api_tasks_settings():
    data = request.json or {}
    try:
        n = max(1, min(6, int(data.get("max_parallel", 2))))
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "Bad number"}), 400
    save_config({"max_parallel_tasks": n})
    with _SLOT_COND:
        _SLOT_COND.notify_all()
    return jsonify({"success": True, "max_parallel": n})


# "new_window" is set by desktop.py: opens another app window (path) — watch one task, start another
WINDOW_HOOKS = {"new_window": None}


@app.route('/api/new-window', methods=['POST'])
def api_new_window():
    path = (request.json or {}).get("path", "/")
    if not str(path).startswith("/"):
        path = "/"
    hook = WINDOW_HOOKS.get("new_window")
    if hook is None:
        return jsonify({"success": False, "error": "not the desktop app"})
    try:
        hook(path)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
