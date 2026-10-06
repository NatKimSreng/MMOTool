"""Turn the openai-whisper `small.pt` you already have into the Hugging Face
WhisperModel layout (encoder only) that Seed-VC loads. Saves a ~1 GB download."""
import os, torch
src = os.path.expanduser("~/.cache/whisper/small.pt")
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "whisper-small", "pytorch_model.bin")
sd = torch.load(src, map_location="cpu", weights_only=False)["model_state_dict"]
out = {
    "encoder.conv1.weight": sd["encoder.conv1.weight"], "encoder.conv1.bias": sd["encoder.conv1.bias"],
    "encoder.conv2.weight": sd["encoder.conv2.weight"], "encoder.conv2.bias": sd["encoder.conv2.bias"],
    "encoder.embed_positions.weight": sd["encoder.positional_embedding"],
    "encoder.layer_norm.weight": sd["encoder.ln_post.weight"], "encoder.layer_norm.bias": sd["encoder.ln_post.bias"],
}
i = 0
while f"encoder.blocks.{i}.attn.query.weight" in sd:
    b, h = f"encoder.blocks.{i}.", f"encoder.layers.{i}."
    m = {"attn.query": "self_attn.q_proj", "attn.key": "self_attn.k_proj", "attn.value": "self_attn.v_proj",
         "attn.out": "self_attn.out_proj", "attn_ln": "self_attn_layer_norm", "mlp.0": "fc1", "mlp.2": "fc2",
         "mlp_ln": "final_layer_norm"}
    for o, n in m.items():
        for p in ("weight", "bias"):
            if b + o + "." + p in sd:
                out[h + n + "." + p] = sd[b + o + "." + p]
    i += 1
out = {k: v.half().contiguous() for k, v in out.items()}
torch.save(out, dst)
print("layers:", i, "tensors:", len(out), "->", dst, round(os.path.getsize(dst) / 1e6), "MB")
