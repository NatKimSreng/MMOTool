# Minimal stand-in for descript-audio-codec's VectorQuantize.
# Seed-VC only builds it when `vector_quantize: true`, which the speech model we use
# does not set — so the real (heavy) package is not needed.
import torch.nn as nn


class VectorQuantize(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise RuntimeError("descript-audio-codec VectorQuantize is not available in this install")
