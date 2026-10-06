"""
Uploads cleanup: delete uploaded copies no job needs any more. Its routes.
"""
import os
import threading
from flask import request, jsonify
import time
from .core import (
    DEFAULT_OUTPUT_FOLDER,
    UPLOAD_FOLDER,
    _read_json,
    app,
    load_config,
    load_history,
    save_config,
)
from .tasks import (
    _TASKS,
    _TASKS_LOCK,
    any_task_running,
)
from .batch_queue import (
    _load_batch_queue,
)


# ═══════════════════════════════════════════════════════════
# UPLOADS CLEANUP — uploads/ keeps a copy of every video sent to the app. Files no job can
# still use are deleted after `uploads_keep_days` (default 3; 0 = never), or on request.
# ═══════════════════════════════════════════════════════════
def _upload_files_in_use():
    """Upload files something may still need: inputs of tasks that are running, paused or
    failed (Resume / Run again), queued items, the channel logo, and the source video of
    every finished dub that can still be re-voiced (job.json)."""
    keep = set()

    def add(p):
        if isinstance(p, (list, tuple)):
            for x in p:
                add(x)
        elif isinstance(p, str) and p:
            keep.add(os.path.normcase(os.path.abspath(p)))

    with _TASKS_LOCK:
        for t in _TASKS.values():
            if t.get("state") != "done":
                prm = t.get("params") or {}
                for k in ("file_path", "file_paths", "srt_path", "downloaded_file", "logo_path"):
                    add(prm.get(k))
    for item in _load_batch_queue():
        if item.get("status") != "completed":
            add(item.get("file_path"))
            add((item.get("options") or {}).get("logo_path"))
    add(load_config().get("logo_path"))
    job_dirs = set()
    if os.path.isdir(DEFAULT_OUTPUT_FOLDER):
        job_dirs.update(os.path.join(DEFAULT_OUTPUT_FOLDER, d) for d in os.listdir(DEFAULT_OUTPUT_FOLDER))
    try:
        job_dirs.update(h.get("output_path", "") for h in load_history() if h.get("output_path"))
    except Exception:
        pass
    for d in job_dirs:
        jp = os.path.join(d, "job.json")
        if os.path.isfile(jp):
            add(_read_json(jp).get("video"))
    return keep


def _cleanup_uploads(min_age_days, dry_run=False):
    """Delete upload files no job needs that are older than min_age_days. Returns a summary."""
    in_use = _upload_files_in_use()
    busy = any_task_running()
    now = time.time()
    res = {"total_files": 0, "total_mb": 0.0, "free_files": 0, "free_mb": 0.0, "in_use_files": 0, "errors": 0}
    for name in os.listdir(UPLOAD_FOLDER):
        path = os.path.join(UPLOAD_FOLDER, name)
        if not os.path.isfile(path):
            continue
        size = os.path.getsize(path)
        res["total_files"] += 1
        res["total_mb"] += size / 1048576
        if os.path.normcase(os.path.abspath(path)) in in_use:
            res["in_use_files"] += 1
            continue
        age_days = (now - os.path.getmtime(path)) / 86400
        # speech audio a running job may still be reading
        if "._asr_audio" in name and busy and age_days < 1:
            continue
        if age_days < min_age_days:
            continue
        if not dry_run:
            try:
                os.remove(path)
            except OSError:
                res["errors"] += 1
                continue
        res["free_files"] += 1
        res["free_mb"] += size / 1048576
    res["total_mb"] = round(res["total_mb"], 1)
    res["free_mb"] = round(res["free_mb"], 1)
    return res


def _uploads_keep_days():
    try:
        return max(0.0, float(load_config().get("uploads_keep_days", 3)))
    except (TypeError, ValueError):
        return 3.0


def _uploads_auto_cleanup():
    time.sleep(60)   # let the app start first
    while True:
        days = _uploads_keep_days()
        if days > 0:
            try:
                r = _cleanup_uploads(days)
                if r["free_files"]:
                    print(f"[uploads] auto-cleanup removed {r['free_files']} files ({r['free_mb']} MB)")
            except Exception as e:
                print(f"[uploads] auto-cleanup failed: {e}")
        time.sleep(6 * 3600)


threading.Thread(target=_uploads_auto_cleanup, daemon=True, name="uploads-cleanup").start()

_MANUAL_CLEANUP_MIN_AGE = 1 / 24   # "Free up space now" still spares files from the last hour


@app.route('/api/uploads/cleanup', methods=['GET'])
def api_uploads_cleanup_info():
    r = _cleanup_uploads(_MANUAL_CLEANUP_MIN_AGE, dry_run=True)
    r["keep_days"] = _uploads_keep_days()
    r["folder"] = UPLOAD_FOLDER
    return jsonify({"success": True, **r})


@app.route('/api/uploads/cleanup', methods=['POST'])
def api_uploads_cleanup():
    data = request.json or {}
    if "keep_days" in data:
        try:
            save_config({"uploads_keep_days": max(0.0, float(data["keep_days"]))})
        except (TypeError, ValueError):
            return jsonify({"success": False, "error": "Bad number"}), 400
        return jsonify({"success": True, "keep_days": _uploads_keep_days()})
    r = _cleanup_uploads(_MANUAL_CLEANUP_MIN_AGE)
    print(f"[uploads] cleanup now removed {r['free_files']} files ({r['free_mb']} MB)")
    return jsonify({"success": True, **r})
