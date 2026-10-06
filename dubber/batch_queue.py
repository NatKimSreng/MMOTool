"""
Overnight queue: dub several videos one after another. Its routes.
"""
import os
import threading
import json
from flask import request, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
import uuid
import time
from .core import (
    UPLOAD_FOLDER,
    _BASE_DIR,
    app,
    load_config,
)
from .tasks import (
    _ACTIVE_STATES,
    _TASKS,
    _new_task,
    _restart_task,
    _run_task,
)


_BATCH_RUNNING = False
BATCH_QUEUE_FILE = os.path.join(_BASE_DIR, "batch_queue.json")

def _load_batch_queue():
    if os.path.exists(BATCH_QUEUE_FILE):
        try:
            with open(BATCH_QUEUE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            pass
    return []

def _save_batch_queue(queue):
    try:
        with open(BATCH_QUEUE_FILE, "w", encoding="utf-8") as f:
            json.dump(queue, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[batch_queue save failed]: {e}")


def _batch_queue_worker():
    """Background sequential worker for overnight batch queue jobs."""
    global _BATCH_RUNNING
    if _BATCH_RUNNING:
        return
    _BATCH_RUNNING = True
    print("[batch_queue] Background worker started")
    try:
        while True:
            queue = _load_batch_queue()
            next_idx = None
            for idx, item in enumerate(queue):
                if item.get("status") == "pending":
                    next_idx = idx
                    break
            if next_idx is None:
                print("[batch_queue] No pending tasks, worker stopping")
                break

            task = queue[next_idx]
            task_id = task.get("id", "")
            queue[next_idx]["status"] = "processing"
            _save_batch_queue(queue)

            opts = task.get("options", {})
            file_path = task.get("file_path")
            job_name = task.get("job_name", f"batch_{task_id}")

            # each queue item runs as a task (shows in Tasks, can be paused / resumed)
            st = _TASKS.get(task.get("task_id", ""))
            try:
                if st is not None and st.get("state") in _ACTIVE_STATES:
                    while st.get("state") in _ACTIVE_STATES:   # resumed from the Tasks panel
                        time.sleep(2)
                    final = st.get("state")
                elif st is not None and st.get("state") == "done":
                    final = "done"
                elif st is not None:
                    final = _restart_task(st, inline=True)
                else:
                    st = _new_task("pipeline_cn_en_to_khmer_task", dict(
                        file_path=file_path,
                        custom_output_dir=opts.get("custom_output_dir", ""),
                        source_lang=opts.get("source_lang", "auto"),
                        do_tts=bool(opts.get("do_tts", True)),
                        voice_key=opts.get("voice_key", "narrator_female"),
                        rate=opts.get("rate", "-5%"),
                        pitch=opts.get("pitch", "+0Hz"),
                        job_name=job_name,
                        burn_subs=bool(opts.get("burn_subs", True)),
                        dubbing_mode=opts.get("dubbing_mode", "duck"),
                        bgm_volume=float(opts.get("bgm_volume", 0.25)),
                        voice_volume=float(opts.get("voice_volume", 1.0)),
                        translate_engine=opts.get("translate_engine", "auto"),
                        translate_style=opts.get("translate_style", "recap"),
                        sub_color=opts.get("sub_color", "yellow"),
                        sub_size=int(opts.get("sub_size", 22)),
                        sub_box=bool(opts.get("sub_box", False)),
                        sub_pos=opts.get("sub_pos", "bottom"),
                        bilingual_subs=bool(opts.get("bilingual_subs", False)),
                        auto_detect_voice=bool(opts.get("auto_detect_voice", True)),
                        logo_path=opts.get("logo_path"),
                        logo_pos=opts.get("logo_pos", "top-right"),
                        logo_size=opts.get("logo_size", "medium"),
                        logo_opacity=float(opts.get("logo_opacity", 0.85)),
                        aspect_ratio=opts.get("aspect_ratio", "original"),
                        hide_box=opts.get("hide_box", ""),
                        review_script=False,
                        cast_mode=opts.get("cast_mode", "two"),
                        blend_scene=bool(opts.get("blend_scene", True)),
                    ), name=f"Queue: {job_name}", kind="dub")
                    queue = _load_batch_queue()
                    for item in queue:
                        if item.get("id") == task_id:
                            item["task_id"] = st["id"]
                    _save_batch_queue(queue)
                    final = _run_task(st)
                status = {"done": "completed", "stopped": "paused"}.get(final, "error")
                err = "" if final == "done" else str(dict.get(st, "status", ""))
            except Exception as e:
                print(f"[batch_queue] Task {task_id} failed: {e}")
                status, err = "error", str(e)
            queue = _load_batch_queue()
            for item in queue:
                if item.get("id") == task_id:
                    item["status"] = status
                    if status == "completed":
                        item["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if err:
                        item["error"] = err
            _save_batch_queue(queue)
            if status == "paused":
                print("[batch_queue] item paused — queue stops here until started again")
                break

            time.sleep(2)
    finally:
        _BATCH_RUNNING = False
        print("[batch_queue] Background worker finished")


def _requeue_unfinished_batch_items():
    """Queue items cut off by an app close (or paused) wait again; their task resumes when the queue runs."""
    queue = _load_batch_queue()
    changed = False
    for item in queue:
        if item.get("status") in ("processing", "paused"):
            item["status"] = "pending"
            changed = True
    if changed:
        _save_batch_queue(queue)


_requeue_unfinished_batch_items()


def _start_batch_queue_worker():
    global _BATCH_RUNNING
    if not _BATCH_RUNNING:
        t = threading.Thread(target=_batch_queue_worker, daemon=True)
        t.start()


@app.route('/api/batch-queue', methods=['GET'])
def get_batch_queue():
    """Return all jobs in the sequential batch queue."""
    queue = _load_batch_queue()
    # an item's task may have been paused / resumed from the Tasks panel — show its real state
    state_map = {"queued": "processing", "running": "processing", "done": "completed",
                 "stopped": "paused", "interrupted": "paused", "error": "error"}
    for item in queue:
        st = _TASKS.get(item.get("task_id", ""))
        if st is not None and item.get("status") != "pending":
            item["status"] = state_map.get(st.get("state"), item.get("status"))
    return jsonify({
        "success": True,
        "queue": queue,
        "is_running": _BATCH_RUNNING
    })


@app.route('/api/batch-queue/add', methods=['POST'])
def add_batch_queue():
    """Add a new dubbing project to the sequential queue."""
    video_files = request.files.getlist('videos')
    saved_paths = []
    stamp = datetime.now().strftime("%H%M%S")

    if video_files and any(f.filename for f in video_files):
        for i, vf in enumerate(video_files):
            if not vf.filename:
                continue
            ext = os.path.splitext(vf.filename)[1].lower() or ".mp4"
            safe_name = secure_filename(vf.filename) or f"clip_{i}.mp4"
            stem, _ = os.path.splitext(safe_name)
            target = os.path.abspath(os.path.join(UPLOAD_FOLDER, f"batch_{stem}_{stamp}_{i:02d}{ext}"))
            vf.save(target)
            saved_paths.append(target)
    
    file_path = saved_paths if len(saved_paths) > 1 else (saved_paths[0] if saved_paths else "")
    if not file_path:
        data = request.json or {}
        file_path = data.get("file_path") or data.get("file") or ""
        job_name = data.get("job_name") or data.get("jobName") or ""
        opts = data.get("options") or {}
    else:
        job_name = request.form.get("jobName", "").strip()
        opts = {
            "source_lang": request.form.get("sourceLang", "auto"),
            "translate_engine": request.form.get("translateEngine", "auto"),
            "translate_style": request.form.get("translateStyle", "recap"),
            "dubbing_mode": request.form.get("dubbingMode", "duck"),
            "voice_key": request.form.get("voice", "narrator_female"),
            "rate": request.form.get("rate", "-5%"),
            "pitch": request.form.get("pitch", "default"),
            "bgm_volume": float(request.form.get("bgmVolume", 0.25)),
            "voice_volume": float(request.form.get("voiceVolume", 1.0)),
            "burn_subs": request.form.get("burnSubs", "true") == "true",
            "sub_color": request.form.get("subColor", "yellow"),
            "sub_size": int(request.form.get("subSize", 22)),
            "sub_box": request.form.get("subBox", "false") == "true",
            "sub_pos": request.form.get("subPos", "bottom"),
            "bilingual_subs": request.form.get("bilingualSubs", "false") == "true",
            "auto_detect_voice": request.form.get("autoDetectVoice", "true") == "true",
            "aspect_ratio": request.form.get("aspectRatio", "original"),
            "hide_box": request.form.get("hideBox", ""),
            "logo_enabled": request.form.get("logoEnabled", "false") == "true",
            "logo_pos": request.form.get("logoPos", "top-right"),
            "logo_size": request.form.get("logoSize", "medium"),
            "logo_opacity": float(request.form.get("logoOpacity", 0.85)),
            "cast_mode": request.form.get("castMode", "two"),
            "blend_scene": request.form.get("blendScene", "true") == "true",
        }
        if opts.get("logo_enabled"):
            cfg = load_config()
            saved_logo = cfg.get("logo_path", "")
            if saved_logo and os.path.exists(saved_logo):
                opts["logo_path"] = saved_logo

    if not file_path:
        return jsonify({"success": False, "error": "No video file provided for queue"}), 400

    task_id = str(uuid.uuid4())[:8]
    display_title = job_name or (os.path.basename(file_path[0]) if isinstance(file_path, list) else os.path.basename(file_path))
    task = {
        "id": task_id,
        "job_name": display_title,
        "file_path": file_path,
        "options": opts,
        "status": "pending",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    queue = _load_batch_queue()
    queue.append(task)
    _save_batch_queue(queue)

    _start_batch_queue_worker()

    return jsonify({"success": True, "task": task, "queue": queue})


@app.route('/api/batch-queue/remove', methods=['POST'])
def remove_batch_queue():
    data = request.json or {}
    task_id = data.get("id", "")
    queue = _load_batch_queue()
    new_q = [t for t in queue if t.get("id") != task_id]
    _save_batch_queue(new_q)
    return jsonify({"success": True, "queue": new_q})


@app.route('/api/batch-queue/clear', methods=['POST'])
def clear_batch_queue():
    queue = _load_batch_queue()
    new_q = [t for t in queue if t.get("status") == "processing"]
    _save_batch_queue(new_q)
    return jsonify({"success": True, "queue": new_q})


@app.route('/api/batch-queue/start', methods=['POST'])
def start_batch_queue():
    if not _BATCH_RUNNING:
        _requeue_unfinished_batch_items()
    _start_batch_queue_worker()
    return jsonify({"success": True, "is_running": _BATCH_RUNNING})
