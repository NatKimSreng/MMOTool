/*
 * Tasks panel — shared by every page.
 * - Every job runs as a background task; this window follows ONE task (the last one it started,
 *   or one opened from the panel), so several windows/pages can each watch their own job.
 * - The floating "Tasks" button lists all jobs: running, waiting, paused (saved) and finished,
 *   with Pause / Resume / View / Folder buttons.
 */
(function () {
  const KEY = 'dub_task';
  const origFetch = window.fetch.bind(window);
  const ss = {
    get() { try { return sessionStorage.getItem(KEY) || ''; } catch (e) { return ''; } },
    set(v) { try { v ? sessionStorage.setItem(KEY, v) : sessionStorage.removeItem(KEY); } catch (e) {} },
  };
  let tracked = ss.get();
  const fromUrl = new URLSearchParams(location.search).get('task');
  if (fromUrl) { tracked = fromUrl; ss.set(fromUrl); }

  function track(id) { tracked = id || ''; ss.set(tracked); }

  // ── this window's progress = its own task ──────────────────────
  function isProgressUrl(u) {
    if (typeof u !== 'string') return false;
    return u === '/progress' || u === location.origin + '/progress';
  }
  async function noteTaskId(res) {
    try {
      if (!(res.headers.get('content-type') || '').includes('json')) return;
      const j = await res.clone().json();
      if (j && j.task_id && j.success !== false) { track(j.task_id); refreshSoon(); }
    } catch (e) {}
  }
  window.fetch = async function (input, init) {
    if (isProgressUrl(input)) input = '/progress?task=' + encodeURIComponent(tracked || 'none');
    const res = await origFetch(input, init);
    const method = ((init && init.method) || (input && input.method) || 'GET').toUpperCase();
    if (method === 'POST') await noteTaskId(res);
    return res;
  };
  const xhrOpen = XMLHttpRequest.prototype.open, xhrSend = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function (method, url) {
    this._dtPost = String(method).toUpperCase() === 'POST';
    if (isProgressUrl(url)) arguments[1] = '/progress?task=' + encodeURIComponent(tracked || 'none');
    return xhrOpen.apply(this, arguments);
  };
  XMLHttpRequest.prototype.send = function () {
    if (this._dtPost) this.addEventListener('load', () => {
      try { const j = JSON.parse(this.responseText); if (j && j.task_id && j.success !== false) { track(j.task_id); refreshSoon(); } } catch (e) {}
    });
    return xhrSend.apply(this, arguments);
  };

  // ── panel UI ─────────────────────────────────────────────────
  const css = `
  .dt-fab{position:fixed;right:16px;bottom:16px;z-index:9000;display:flex;align-items:center;gap:8px;padding:9px 14px;border-radius:999px;
    background:#1d1d26;color:#ececf1;border:1px solid #383847;font:600 13.5px/1 Inter,system-ui,sans-serif;cursor:pointer;box-shadow:0 6px 24px rgba(0,0,0,.45)}
  .dt-fab:hover{border-color:#f5a524}
  .dt-fab .dt-n{min-width:20px;height:20px;padding:0 6px;border-radius:999px;background:#2a2a36;color:#9a9aab;display:grid;place-items:center;font-size:12px}
  .dt-fab.run .dt-n{background:#f5a524;color:#1a1206}
  .dt-fab.need .dt-n{background:#f05252;color:#fff}
  .dt-spin{width:10px;height:10px;border-radius:50%;border:2px solid #f5a524;border-right-color:transparent;animation:dtspin .8s linear infinite;display:none}
  .dt-fab.run .dt-spin{display:block}
  @keyframes dtspin{to{transform:rotate(360deg)}}
  .dt-panel{position:fixed;top:0;right:0;bottom:0;width:min(420px,100vw);z-index:9001;background:#16161d;color:#ececf1;border-left:1px solid #2a2a36;
    display:flex;flex-direction:column;font:14px/1.45 Inter,'Kantumruy Pro',system-ui,sans-serif;box-shadow:-12px 0 40px rgba(0,0,0,.5);transform:translateX(105%);transition:transform .2s}
  .dt-panel.open{transform:none}
  .dt-head{padding:14px 16px 10px;border-bottom:1px solid #2a2a36}
  .dt-head h3{margin:0;font-size:16px;display:flex;align-items:center;gap:8px}
  .dt-head p{margin:4px 0 10px;color:#9a9aab;font-size:12.5px}
  .dt-x{margin-left:auto;background:none;border:0;color:#9a9aab;font-size:18px;cursor:pointer;padding:2px 6px}
  .dt-tools{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:12.5px;color:#9a9aab}
  .dt-seg{display:inline-flex;background:#1d1d26;border:1px solid #2a2a36;border-radius:8px;padding:2px}
  .dt-seg button{background:none;border:0;color:#9a9aab;padding:3px 9px;border-radius:6px;cursor:pointer;font:inherit}
  .dt-seg button.on{background:#383847;color:#ececf1}
  .dt-list{flex:1;overflow:auto;padding:10px 12px 20px;display:flex;flex-direction:column;gap:8px}
  .dt-item{background:#1d1d26;border:1px solid #2a2a36;border-radius:12px;padding:10px 12px}
  .dt-item.mine{border-color:rgba(245,165,36,.55)}
  .dt-top{display:flex;gap:8px;align-items:center}
  .dt-name{flex:1;min-width:0;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .dt-pill{flex:none;font-size:11.5px;font-weight:600;padding:2px 8px;border-radius:999px;background:#2a2a36;color:#9a9aab}
  .dt-pill.running,.dt-pill.queued{background:rgba(245,165,36,.14);color:#f5a524}
  .dt-pill.review{background:rgba(240,82,82,.14);color:#f05252}
  .dt-pill.done{background:rgba(34,197,94,.12);color:#22c55e}
  .dt-pill.error{background:rgba(240,82,82,.12);color:#f05252}
  .dt-pill.stopped,.dt-pill.interrupted{background:rgba(96,165,250,.13);color:#60a5fa}
  .dt-meta{color:#6b6b7b;font-size:12px;margin-top:2px}
  .dt-status{color:#9a9aab;font-size:12.5px;margin-top:6px;overflow:hidden;text-overflow:ellipsis;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;word-break:break-word}
  .dt-bar{height:5px;background:#2a2a36;border-radius:9px;margin-top:8px;overflow:hidden}
  .dt-bar i{display:block;height:100%;background:#f5a524;border-radius:9px;transition:width .4s}
  .dt-acts{display:flex;gap:6px;flex-wrap:wrap;margin-top:9px}
  .dt-btn{background:#2a2a36;border:1px solid #383847;color:#ececf1;border-radius:8px;padding:5px 10px;font:500 12.5px Inter,system-ui,sans-serif;cursor:pointer;text-decoration:none}
  .dt-btn:hover{border-color:#f5a524}
  .dt-btn.pri{background:#f5a524;border-color:#f5a524;color:#1a1206}
  .dt-btn:disabled{opacity:.5;cursor:default}
  .dt-empty{color:#6b6b7b;text-align:center;padding:40px 10px;font-size:13.5px}
  .dt-sec{color:#6b6b7b;font-size:11.5px;text-transform:uppercase;letter-spacing:.06em;margin:8px 2px 0}
  .dt-toast{position:fixed;right:16px;bottom:66px;z-index:9002;background:#1d1d26;color:#ececf1;border:1px solid #383847;border-radius:12px;padding:10px 14px;
    font:500 13.5px Inter,system-ui,sans-serif;box-shadow:0 8px 30px rgba(0,0,0,.5);max-width:min(360px,calc(100vw - 32px));cursor:pointer}
  .dt-toast b{display:block;margin-bottom:2px}
  @media (max-width:520px){.dt-fab{right:12px;bottom:12px}}
  `;
  const style = document.createElement('style'); style.textContent = css; document.head.appendChild(style);

  const fab = document.createElement('button');
  fab.type = 'button'; fab.className = 'dt-fab'; fab.title = 'Background tasks';
  fab.innerHTML = '<span class="dt-spin"></span><span>Tasks</span><span class="dt-n">0</span>';
  const panel = document.createElement('aside');
  panel.className = 'dt-panel'; panel.setAttribute('aria-label', 'Tasks');
  panel.innerHTML = `
    <div class="dt-head">
      <h3>Tasks <button class="dt-x" type="button" title="Close">✕</button></h3>
      <p>Jobs keep running in the background — start another one any time. Paused jobs are saved: press Resume to continue where they stopped, even after closing the app.</p>
      <div class="dt-tools">
        <span>Run at once</span><span class="dt-seg" data-par></span>
        <button class="dt-btn" type="button" data-newwin title="Open another window — watch one job, start another">⧉ New window</button>
        <button class="dt-btn" type="button" data-clear title="Remove finished tasks from this list (files stay on disk)">Clear finished</button>
      </div>
    </div>
    <div class="dt-list"><div class="dt-empty">No tasks yet.</div></div>`;
  function mount() { document.body.appendChild(fab); document.body.appendChild(panel); }
  document.body ? mount() : document.addEventListener('DOMContentLoaded', mount);

  const $p = sel => panel.querySelector(sel);
  const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const LABEL = {queued: 'Waiting', running: 'Running', review: 'Needs you', done: 'Done', error: 'Failed', stopped: 'Paused', interrupted: 'Paused'};
  const KIND = {dub: 'Dub', cut: 'Cut', speech: 'Voice', export: 'Studio export', download: 'Download', srt: 'Subtitles', merge: 'Merge', record: 'Re-voice'};
  const ACTIVE = s => s === 'queued' || s === 'running' || s === 'review';
  function ago(t) {
    if (!t) return '';
    const s = Math.max(0, Date.now() / 1000 - t);
    if (s < 60) return 'just now';
    if (s < 3600) return Math.floor(s / 60) + ' min ago';
    if (s < 86400) return Math.floor(s / 3600) + ' h ago';
    return new Date(t * 1000).toLocaleDateString();
  }
  function dur(sec) { sec = Math.max(0, Math.round(sec)); const h = Math.floor(sec / 3600), m = Math.floor(sec % 3600 / 60);
    return h ? `${h}h ${m}m` : m ? `${m}m` : `${sec}s`; }

  let tasks = [], lastStates = null, timer = null, open = false;

  function actions(t) {
    const b = [];
    const viewable = t.kind === 'dub';
    if (ACTIVE(t.state)) {
      if (viewable) b.push(`<button class="dt-btn ${t.state === 'review' ? 'pri' : ''}" data-act="view">${t.state === 'review' ? 'Check script' : 'View'}</button>`);
      if (!t.stop_requested) b.push(`<button class="dt-btn" data-act="stop">${t.resumable ? '⏸ Pause' : '■ Stop'}</button>`);
      else b.push(`<button class="dt-btn" disabled>Pausing…</button>`);
    } else if (t.state === 'done') {
      if (viewable && t.final_video) b.push(`<button class="dt-btn pri" data-act="view">▶ Open</button>`);
      if (t.final_video && !viewable) b.push(`<a class="dt-btn" href="/download-file?path=${encodeURIComponent(t.final_video)}">⬇ Download</a>`);
    } else {
      if (t.input_ok) b.push(`<button class="dt-btn pri" data-act="resume">${t.resumable ? '▶ Resume' : '↻ Run again'}</button>`);
    }
    if (t.job_dir || t.result_path) b.push(`<button class="dt-btn" data-act="folder">📁 Folder</button>`);
    if (!ACTIVE(t.state)) b.push(`<button class="dt-btn" data-act="remove" title="Remove from the list (files stay on disk)">✕</button>`);
    return b.join('');
  }
  function itemHtml(t) {
    const pct = Math.max(0, Math.min(100, t.percent || 0));
    let meta = (KIND[t.kind] || t.kind || '') + ' · ';
    if (t.state === 'running' && t.started_at) meta += 'running ' + dur(Date.now() / 1000 - t.started_at);
    else if (t.finished_at) meta += ago(t.finished_at);
    else meta += ago(t.created_at);
    const pill = t.state === 'running' ? `${pct}%` : (t.state === 'interrupted' ? 'Paused · app closed' : LABEL[t.state] || t.state);
    return `<div class="dt-item ${t.id === tracked ? 'mine' : ''}" data-id="${esc(t.id)}">
      <div class="dt-top"><div class="dt-name" title="${esc(t.name)}">${esc(t.name)}</div><span class="dt-pill ${esc(t.state)}">${esc(pill)}</span></div>
      <div class="dt-meta">${esc(meta)}${t.id === tracked ? ' · this window' : ''}</div>
      ${t.status ? `<div class="dt-status" title="${esc(t.status)}">${esc(t.status)}</div>` : ''}
      ${ACTIVE(t.state) || t.state === 'stopped' || t.state === 'interrupted' ? `<div class="dt-bar"><i style="width:${pct}%"></i></div>` : ''}
      ${t.state !== 'done' && !t.input_ok && !ACTIVE(t.state) ? '<div class="dt-meta">The original file is gone — can\'t run again.</div>' : ''}
      <div class="dt-acts">${actions(t)}</div></div>`;
  }
  function render() {
    const act = tasks.filter(t => ACTIVE(t.state));
    const paused = tasks.filter(t => ['stopped', 'interrupted', 'error'].includes(t.state));
    const done = tasks.filter(t => t.state === 'done');
    const need = tasks.some(t => t.state === 'review');
    fab.classList.toggle('run', act.length > 0);
    fab.classList.toggle('need', need);
    fab.querySelector('.dt-n').textContent = act.length || (paused.length ? paused.length : 0);
    fab.title = act.length ? `${act.length} task(s) running` : paused.length ? `${paused.length} paused task(s) — can be resumed` : 'Background tasks';
    if (!open) return;
    const sec = (title, list) => list.length ? `<div class="dt-sec">${title}</div>` + list.map(itemHtml).join('') : '';
    $p('.dt-list').innerHTML = tasks.length
      ? sec('Running', act) + sec('Paused & saved', paused) + sec('Finished', done)
      : '<div class="dt-empty">No tasks yet. Start a dub and it shows up here — you can keep working while it runs.</div>';
  }
  function notify(t) {
    const msg = t.state === 'done' ? ['✓ Ready', t.name] : t.state === 'error' ? ['Failed', t.name + ' — ' + (t.status || '').replace(/^Error:\s*/, '')]
      : t.state === 'review' ? ['Needs you', t.name + ' — check the Khmer script'] : null;
    if (!msg) return;
    const el = document.createElement('div'); el.className = 'dt-toast';
    el.innerHTML = `<b>${esc(msg[0])}</b>${esc(msg[1])}`;
    el.onclick = () => { el.remove(); openPanel(); };
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 7000);
  }
  async function refresh() {
    let j; try { j = await (await origFetch('/api/tasks')).json(); } catch (e) { return; }
    tasks = j.tasks || [];
    const states = {}; tasks.forEach(t => states[t.id] = t.state);
    if (lastStates) tasks.forEach(t => { const was = lastStates[t.id]; if (was && was !== t.state && ACTIVE(was) && t.id !== tracked) notify(t); });
    lastStates = states;
    const seg = $p('[data-par]');
    seg.innerHTML = [1, 2, 3, 4].map(n => `<button type="button" data-n="${n}" class="${n === j.max_parallel ? 'on' : ''}">${n}</button>`).join('');
    render();
    schedule();
  }
  function schedule() {
    clearTimeout(timer);
    const busy = tasks.some(t => ACTIVE(t.state));
    timer = setTimeout(refresh, open ? 1500 : busy ? 3000 : 8000);
  }
  function refreshSoon() { clearTimeout(timer); timer = setTimeout(refresh, 400); }
  function openPanel() { open = true; panel.classList.add('open'); render(); refreshSoon(); }
  function closePanel() { open = false; panel.classList.remove('open'); }

  async function post(url, body) {
    try { return await (await origFetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {})})).json(); }
    catch (e) { return {success: false, error: 'Could not reach the app.'}; }
  }
  function openTask(id) {
    if (typeof window.onDubTaskOpen === 'function' && window.onDubTaskOpen(id) === true) { closePanel(); return; }
    location.href = '/?task=' + encodeURIComponent(id);
  }

  fab.addEventListener('click', () => open ? closePanel() : openPanel());
  panel.addEventListener('click', async e => {
    if (e.target.closest('.dt-x')) return closePanel();
    const nb = e.target.closest('[data-n]');
    if (nb) { await post('/api/tasks/settings', {max_parallel: +nb.dataset.n}); return refresh(); }
    if (e.target.closest('[data-clear]')) { await post('/api/tasks/clear'); return refresh(); }
    if (e.target.closest('[data-newwin]')) {
      const r = await post('/api/new-window', {path: '/'});
      if (!r.success) window.open('/', '_blank');
      return;
    }
    const btn = e.target.closest('[data-act]'); if (!btn) return;
    const id = btn.closest('.dt-item').dataset.id, t = tasks.find(x => x.id === id) || {};
    const act = btn.dataset.act;
    if (act === 'view') return openTask(id);
    if (act === 'folder') return post('/open-folder', {path: t.job_dir || t.result_path});
    btn.disabled = true;
    let r;
    if (act === 'stop') r = await post(`/api/tasks/${id}/stop`);
    else if (act === 'resume') r = await post(`/api/tasks/${id}/resume`);
    else if (act === 'remove') { r = await post(`/api/tasks/${id}/remove`); if (id === tracked) track(''); }
    if (r && !r.success) alert(r.error || 'Could not do that.');
    if (act === 'resume' && r && r.success && typeof window.onDubTaskResumed === 'function') window.onDubTaskResumed(id);
    refresh();
  });
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && open) closePanel(); });

  window.DubTasks = {
    track, current: () => tracked, open: openPanel, refresh: refreshSoon,
    stop: id => post(`/api/tasks/${id}/stop`).then(r => (refreshSoon(), r)),
    resume: id => post(`/api/tasks/${id}/resume`).then(r => (refreshSoon(), r)),
  };
  refresh();
})();
