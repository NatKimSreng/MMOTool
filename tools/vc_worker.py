"""
Voice conversion worker (runs inside tools/vc-env, launched by app.py).

Turns each Khmer TTS line into the voice of a reference recording with Seed-VC
(zero-shot, language independent — Khmer pronunciation comes from the TTS,
the voice colour comes from the reference).

Two models:
  "f0"    — 44 kHz, follows the TTS line's own pitch melody (keeps the delivery
            lively, can widen it further). Used automatically when installed.
  "basic" — 22 kHz, smaller; pitch movement gets flatter.

Usage:  vc-env/Scripts/python.exe vc_worker.py job.json
job.json = {"items": [{"src", "dst", "reference", "f0_scale"?}, ...],
            "steps": 12, "cfg_rate": 0.7, "f0_scale": 1.0, "model": "auto"}
Prints "PROGRESS <done> <total>" lines, then "DONE <ok> <total>".
"""
import contextlib
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "seed-vc-code")
MODELS = os.path.join(HERE, "models")
sys.path.insert(0, CODE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.simplefilter("ignore")

import numpy as np
import torch

# checkpoints come from the official Seed-VC / NVIDIA / FunASR / RVC repos and contain
# plain Python objects, which torch>=2.6 refuses by default
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{"weights_only": False, **k})

import librosa
import soundfile as sf
import torchaudio
import yaml
from modules.audio import mel_spectrogram
from modules.commons import build_model, load_checkpoint, recursive_munch

SPECS = {
    "f0": {
        "config": "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
        "ckpt": "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
        "vocoder": "bigvgan44k", "f0": True,
    },
    "basic": {
        "config": "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
        "ckpt": "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
        "vocoder": "bigvgan", "f0": False,
    },
}
REF_SECONDS = 12      # leaves room for long lines in the 30 s context window
GROUP_SECONDS = 15.0  # lines of the same voice are converted together (much faster), then split
OVERLAP_FRAMES = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_fp16 = device.type == "cuda"


def log(*a):
    print(*a, flush=True)


def model_installed(name):
    sp = SPECS[name]
    files = [sp["config"], sp["ckpt"], os.path.join(sp["vocoder"], "bigvgan_generator.pt")]
    if sp["f0"]:
        files.append("rmvpe.pt")
    return all(os.path.isfile(os.path.join(MODELS, f)) for f in files)


def load_models(name):
    sp = SPECS[name]
    cfg = yaml.safe_load(open(os.path.join(MODELS, sp["config"])))
    mp = recursive_munch(cfg["model_params"])
    mp.dit_type = "DiT"
    model = build_model(mp, stage="DiT")
    model, _, _, _ = load_checkpoint(model, None, os.path.join(MODELS, sp["ckpt"]),
                                     load_only_params=True, ignore_modules=[], is_distributed=False)
    for k in model:
        model[k].eval()
        model[k].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    from modules.campplus.DTDNN import CAMPPlus
    camp = CAMPPlus(feat_dim=80, embedding_size=192)
    camp.load_state_dict(torch.load(os.path.join(MODELS, "campplus_cn_common.bin"), map_location="cpu"))
    camp.eval().to(device)

    from modules.bigvgan import bigvgan
    voc = bigvgan.BigVGAN.from_pretrained(os.path.join(MODELS, sp["vocoder"]), use_cuda_kernel=False)
    voc.remove_weight_norm()
    voc = voc.eval().to(device)

    from transformers import AutoFeatureExtractor, WhisperModel
    wdir = os.path.join(MODELS, "whisper-small")
    wm = WhisperModel.from_pretrained(wdir, torch_dtype=torch.float16 if use_fp16 else torch.float32).to(device)
    del wm.decoder
    fe = AutoFeatureExtractor.from_pretrained(wdir)

    def semantic(w16):
        inp = fe([w16.squeeze(0).cpu().numpy()], return_tensors="pt", return_attention_mask=True, sampling_rate=16000)
        feats = wm._mask_input_features(inp.input_features, attention_mask=inp.attention_mask).to(device)
        with torch.no_grad():
            out = wm.encoder(feats.to(wm.encoder.dtype), head_mask=None, output_attentions=False,
                             output_hidden_states=False, return_dict=True)
        s = out.last_hidden_state.to(torch.float32)
        return s[:, : w16.size(-1) // 320 + 1]

    f0_fn = None
    if sp["f0"]:
        from modules.rmvpe import RMVPE
        rmvpe = RMVPE(os.path.join(MODELS, "rmvpe.pt"), is_half=False, device=device)
        f0_fn = rmvpe.infer_from_audio

    pp = cfg["preprocess_params"]
    spc = pp["spect_params"]
    sr = pp["sr"]
    mel_args = dict(n_fft=spc["n_fft"], win_size=spc["win_length"], hop_size=spc["hop_length"], num_mels=spc["n_mels"],
                    sampling_rate=sr, fmin=spc.get("fmin", 0), fmax=None, center=False)
    to_mel = lambda x: mel_spectrogram(x, **mel_args)
    return model, camp, voc, semantic, f0_fn, to_mel, sr, spc["hop_length"]


def autocast():
    if use_fp16:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def crossfade(a, b, n):
    fo = np.cos(np.linspace(0, np.pi / 2, n)) ** 2
    fi = np.cos(np.linspace(np.pi / 2, 0, n)) ** 2
    m = min(n, len(b))
    b[:m] = b[:m] * fi[:m] + (a[-n:] * fo)[:m]
    return b


def widen_pitch_world(x, scale, sr):
    """For the basic model only: stretch pitch movement with the WORLD vocoder."""
    if not scale or abs(scale - 1.0) < 0.02 or len(x) < sr // 4:
        return x
    try:
        import pyworld as pw
    except ImportError:
        return x
    xd = x.astype(np.float64)
    f0, t = pw.dio(xd, sr, f0_floor=60, f0_ceil=500, frame_period=5.0)
    f0 = pw.stonemask(xd, f0, t, sr)
    voiced = f0 > 0
    if voiced.sum() < 10:
        return x
    lf = np.log(f0[voiced])
    centre = np.median(lf)
    f0[voiced] = np.exp(centre + (lf - centre) * scale)
    y = pw.synthesize(f0, pw.cheaptrick(xd, f0, t, sr), pw.d4c(xd, f0, t, sr), sr, frame_period=5.0).astype(np.float32)
    y = y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
    return y * min(1.0, 0.95 / float(np.abs(y).max() or 1.0))


@torch.no_grad()
def main(job_path):
    job = json.load(open(job_path, encoding="utf-8"))
    items = job["items"]
    steps = int(job.get("steps", 12))
    cfg_rate = float(job.get("cfg_rate", 0.7))
    want = job.get("model", "auto")
    name = "f0" if (want in ("auto", "f0") and model_installed("f0")) else "basic"
    t0 = time.time()
    model, camp, voc, semantic, f0_fn, to_mel, SR, HOP = load_models(name)
    max_context = SR // HOP * 30
    gap = int(0.25 * SR)
    log(f"LOADED {device.type} model={name} {time.time() - t0:.1f}s")

    ref_cache = {}

    def reference(path):
        if path not in ref_cache:
            ref = librosa.load(path, sr=SR)[0][: SR * REF_SECONDS]
            ref = torch.tensor(ref).unsqueeze(0).float().to(device)
            r16 = torchaudio.functional.resample(ref, SR, 16000)
            s_ori = semantic(r16)
            mel2 = to_mel(ref)
            fb = torchaudio.compliance.kaldi.fbank(r16, num_mel_bins=80, dither=0, sample_frequency=16000)
            style = camp((fb - fb.mean(dim=0, keepdim=True)).unsqueeze(0))
            f0_ori = None
            if f0_fn:
                f0_ori = torch.from_numpy(f0_fn(r16[0], thred=0.03)).to(device)[None]
            prompt, *_ = model.length_regulator(s_ori, ylens=torch.LongTensor([mel2.size(2)]).to(device),
                                                n_quantizers=3, f0=f0_ori)
            ref_med = None
            if f0_ori is not None and (f0_ori > 1).any():
                ref_med = torch.median(torch.log(f0_ori[f0_ori > 1] + 1e-5))
            ref_cache[path] = (mel2, style, prompt, ref_med)
        return ref_cache[path]

    def convert(src, ref_path, scale):
        mel2, style, prompt, ref_med = reference(ref_path)
        src_t = torch.tensor(src).unsqueeze(0).float().to(device)
        s16 = torchaudio.functional.resample(src_t, SR, 16000)
        s_alt = semantic(s16[:, : 16000 * 30])
        mel = to_mel(src_t)
        f0_cond = None
        if f0_fn:
            f0_alt = torch.from_numpy(f0_fn(s16[0], thred=0.03)).to(device)[None]
            voiced = f0_alt > 1
            lf = torch.log(f0_alt + 1e-5)
            if voiced.any() and ref_med is not None:
                med = torch.median(lf[voiced])
                # keep the line's melody, move it to the speaker's pitch, optionally widen it
                lf[voiced] = ref_med + (lf[voiced] - med) * scale
            f0_cond = torch.exp(lf)
            f0_cond[~voiced] = 0
        cond, *_ = model.length_regulator(s_alt, ylens=torch.LongTensor([mel.size(2)]).to(device),
                                          n_quantizers=3, f0=f0_cond)
        max_src = max_context - mel2.size(2)
        ow = OVERLAP_FRAMES * HOP
        done, chunks, prev = 0, [], None
        while done < cond.size(1):
            piece = cond[:, done: done + max_src]
            last = done + max_src >= cond.size(1)
            cat = torch.cat([prompt, piece], dim=1)
            with autocast():
                vc = model.cfm.inference(cat, torch.LongTensor([cat.size(1)]).to(device), mel2, style, None,
                                         steps, inference_cfg_rate=cfg_rate)
                vc = vc[:, :, mel2.size(-1):]
            wav = voc(vc.float()).squeeze().cpu().numpy()
            if prev is None:
                if last:
                    chunks.append(wav)
                    break
                chunks.append(wav[:-ow])
            elif last:
                chunks.append(crossfade(prev, wav, ow))
                break
            else:
                chunks.append(crossfade(prev, wav[:-ow], ow))
            prev = wav[-ow:]
            done += vc.size(2) - OVERLAP_FRAMES
        return np.concatenate(chunks).astype(np.float32)

    def save(path, out):
        peak = float(np.abs(out).max()) if len(out) else 1.0
        if peak > 0.98:
            out = out * (0.98 / peak)
        sf.write(path, out, SR)

    # group lines of the same voice + widening (in story order) into ~15 s pieces
    by_key = {}
    for idx, it in enumerate(items):
        scale = float(it.get("f0_scale", job.get("f0_scale", 1.0)))
        try:
            audio = librosa.load(it["src"], sr=SR)[0]
            if not f0_fn:
                audio = widen_pitch_world(audio, scale, SR)
        except Exception as e:
            log(f"ERROR {idx + 1} load: {e}")
            continue
        by_key.setdefault((it["reference"], round(scale, 2)), []).append((idx, audio))
    groups = []
    for (ref_path, scale), lines in by_key.items():
        cur, cur_len = [], 0
        for idx, audio in lines:
            if cur and cur_len + len(audio) > GROUP_SECONDS * SR:
                groups.append((ref_path, scale, cur))
                cur, cur_len = [], 0
            cur.append((idx, audio))
            cur_len += len(audio) + gap
        if cur:
            groups.append((ref_path, scale, cur))

    ok = finished = 0
    for ref_path, scale, g in groups:
        try:
            parts, spans, pos = [], [], 0
            for idx, audio in g:
                parts += [audio, np.zeros(gap, np.float32)]
                spans.append((idx, pos, pos + len(audio)))
                pos += len(audio) + gap
            joined = np.concatenate(parts)
            out = convert(joined, ref_path, scale)
            ratio = len(out) / float(len(joined))
            for idx, a, b in spans:
                save(items[idx]["dst"], out[int(a * ratio): int(b * ratio)])
                ok += 1
        except Exception as e:
            log(f"ERROR group at {g[0][0] + 1}: {type(e).__name__}: {e} — retrying lines one by one")
            for idx, audio in g:
                try:
                    save(items[idx]["dst"], convert(audio, ref_path, scale))
                    ok += 1
                except Exception as e2:
                    log(f"ERROR {idx + 1} {type(e2).__name__}: {e2}")
        finished += len(g)
        log(f"PROGRESS {finished} {len(items)}")
    log(f"DONE {ok} {len(items)} {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1])
