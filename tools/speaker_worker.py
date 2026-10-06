"""
Speaker detection worker (runs inside tools/vc-env, launched by app.py).

For every subtitle line of a movie:
  1. remove music / effects (Demucs, GPU), 5-minute chunks so RAM stays small
  2. voice fingerprint (CAM++ speaker embedding) + pitch (RMVPE)
  3. group lines into characters (same fingerprint = same person)
  4. one pitch per character from ALL their lines (app.py turns it into man / woman / child,
     with the story text deciding the unclear cases), so one odd line can't flip a voice

detect:  python speaker_worker.py detect job.json
         job = {"audio": path, "segments": [[start, end], ...], "out": path, "separate": true}
"""
import json
import os
import subprocess
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
sys.path.insert(0, os.path.join(HERE, "seed-vc-code"))
os.environ.setdefault("TORCH_HOME", os.path.join(HERE, "torch-cache"))
warnings.simplefilter("ignore")

import numpy as np
import torch
import torchaudio

_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{"weights_only": False, **k})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 16000
CLUSTER_DISTANCE = 0.70   # cosine distance, average linkage (tuned on real recognised lines of a 6-character test scene)


def log(*a):
    print(*a, flush=True)


def load_campplus():
    from modules.campplus.DTDNN import CAMPPlus
    m = CAMPPlus(feat_dim=80, embedding_size=192)
    m.load_state_dict(torch.load(os.path.join(MODELS, "campplus_cn_common.bin"), map_location="cpu"))
    return m.eval().to(device)


def load_rmvpe():
    from modules.rmvpe import RMVPE
    return RMVPE(os.path.join(MODELS, "rmvpe.pt"), is_half=False, device=device)


@torch.no_grad()
def embed(camp, x):
    fb = torchaudio.compliance.kaldi.fbank(torch.from_numpy(x).float()[None], num_mel_bins=80, dither=0,
                                           sample_frequency=SR)
    e = camp((fb - fb.mean(0, keepdim=True)).unsqueeze(0).to(device))[0].cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-9)


def pitch(rm, x):
    f = rm.infer_from_audio(torch.from_numpy(x), thred=0.03)
    v = f[f > 1]
    return (float(np.median(v)) if len(v) >= 8 else None), (len(v) / max(len(f), 1))


def read_audio(path, start, dur, sr, channels):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, start):.3f}", "-t", f"{dur:.3f}", "-i", path, "-vn",
           "-ac", str(channels), "-ar", str(sr), "-f", "f32le", "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    a = np.frombuffer(raw, dtype=np.float32)
    return a.reshape(-1, channels).T.copy() if channels > 1 else a.copy()


def media_duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


# ─────────────────────────────── detection ───────────────────────────────
def cluster(embs, idx):
    from sklearn.cluster import AgglomerativeClustering
    labels = {}
    if len(idx) == 1:
        return {idx[0]: 0}
    lab = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                  distance_threshold=CLUSTER_DISTANCE).fit_predict(np.array([embs[i] for i in idx]))
    for i, l in zip(idx, lab):
        labels[i] = int(l)
    return labels


def detect(job_path):
    job = json.load(open(job_path, encoding="utf-8"))
    segs = job["segments"]
    audio = job["audio"]
    t0 = time.time()
    camp, rm = load_campplus(), load_rmvpe()
    sep = None
    if job.get("separate", True):
        try:
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            sep = get_model("htdemucs").eval().to(device)
        except Exception as e:
            log(f"WARN demucs unavailable ({e}) — using the mixed audio")
    log(f"LOADED {device.type} separate={sep is not None} {time.time() - t0:.1f}s")

    total = media_duration(audio) or (max(e for _, e in segs) + 1)
    CH = 300.0
    embs, f0s, voiced = {}, {}, {}
    order = sorted(range(len(segs)), key=lambda i: segs[i][0])
    pos, done = 0, 0
    c = 0.0
    while c < total and pos < len(order):
        mine = []
        while pos < len(order) and segs[order[pos]][0] < c + CH:
            mine.append(order[pos]); pos += 1
        if mine:
            end = max(segs[i][1] for i in mine) + 0.5
            if sep is not None:
                mix = read_audio(audio, c, end - c, 44100, 2)
                with torch.no_grad():
                    out = apply_model(sep, torch.from_numpy(mix)[None], device=device, split=True, overlap=0.1,
                                      progress=False)[0]
                voc = out[sep.sources.index("vocals")].mean(0)
                voc = torchaudio.functional.resample(voc, 44100, SR).numpy()
            else:
                voc = read_audio(audio, c, end - c, SR, 1)
            for i in mine:
                a = int(max(0.0, segs[i][0] - c - 0.05) * SR)
                b = int(max(0.0, segs[i][1] - c + 0.05) * SR)
                x = voc[a:b]
                if len(x) < int(0.4 * SR) or np.sqrt((x ** 2).mean()) < 1e-3:
                    continue
                f0, vr = pitch(rm, x)
                f0s[i], voiced[i] = f0, vr
                if vr > 0.15:
                    embs[i] = embed(camp, x)
            done += len(mine)
            log(f"PROGRESS {done} {len(segs)}")
        c += CH

    idx = sorted(embs)
    if job.get("save_embeddings"):
        np.savez(job["save_embeddings"], idx=np.array(idx), E=np.array([embs[i] for i in idx]))
    labels = cluster(embs, idx) if idx else {}
    # per-character decision from all of their lines (longer lines count more)
    speakers = {}
    for i, l in labels.items():
        speakers.setdefault(l, []).append(i)
    spk_info = {}
    for l, members in speakers.items():
        # duration-weighted median pitch of all the character's lines
        pts = sorted((f0s[i], max(0.3, segs[i][1] - segs[i][0])) for i in members if f0s.get(i))
        med_f0 = None
        if pts:
            half, acc = sum(w for _, w in pts) / 2, 0.0
            for f, w in pts:
                acc += w
                if acc >= half:
                    med_f0 = f
                    break
        spk_info[l] = {"n": len(members), "f0": med_f0 and round(med_f0)}
    # name speakers by how much they talk: S1 = main character
    ranked = sorted(spk_info, key=lambda l: -spk_info[l]["n"])
    names = {l: f"S{k + 1}" for k, l in enumerate(ranked)}
    lines = []
    for i in range(len(segs)):
        l = labels.get(i)
        if l is None:
            lines.append({"speaker": None, "f0": f0s.get(i) and round(f0s[i])})
        else:
            lines.append({"speaker": names[l], "f0": f0s.get(i) and round(f0s[i])})
    speakers_out = [{"id": names[l], **spk_info[l]} for l in ranked]
    json.dump({"lines": lines, "speakers": speakers_out}, open(job["out"], "w", encoding="utf-8"), indent=1)
    log(f"DONE {len(idx)} {len(segs)} speakers={len(speakers_out)} {time.time() - t0:.1f}s")


if __name__ == "__main__":
    detect(sys.argv[2])
