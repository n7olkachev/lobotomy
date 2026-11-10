"""Shared pieces: model paths, PyTorch reference steering hook, recipe -> steer tensor, calibration stats."""
import json, os, re
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models", "q35-0.8b-text")
ONNX_DIR = os.path.join(MODEL_DIR, "onnx")
STATS_DIR = os.path.join(ROOT, "stats")
os.makedirs(STATS_DIR, exist_ok=True)

PROMPTS = [  # user turns; chat template applied, thinking off
    "Who are you?",
    "What do you think of humans?",
    "How many miles should I run this week?",
    "Describe your ideal world.",
    "Write a four-line poem about the sea.",
    "What is 17 times 23?",
]


def parse_recipe(s):
    """'L14N8949+1.0 L13N7956-1' -> [(14, 8949, 1.0), (13, 7956, -1.0)]"""
    out = []
    for tok in s.split():
        m = re.fullmatch(r"L(\d+)N(\d+)([+-]\d*\.?\d+)", tok)
        assert m, tok
        out.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
    return out


def load_stats():
    with open(os.path.join(STATS_DIR, "stats.json")) as f:
        st = json.load(f)
    down_norms = np.load(os.path.join(STATS_DIR, "down_norms.npy"))  # [L, I]
    return st, down_norms


def recipe_to_steer(recipe, st, down_norms):
    """steer[L, n] = dose * median_resid_norm_after_L / ||W_down[:, n]||  (float32 [L, I])"""
    steer = np.zeros(down_norms.shape, np.float32)
    for L, n, dose in recipe:
        steer[L, n] += dose * st["resid_median"][L] / down_norms[L, n]
    return steer


# ---------------------------------------------------------------- PyTorch reference
def load_torch(dtype=torch.float32, device="cpu"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype).to(device).eval()
    return tok, model


def chat_ids(tok, user, device="cpu"):
    text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    return tok(text, return_tensors="pt").input_ids.to(device), text


class TorchSteerer:
    """Adds steer[L, n] to MLP neuron activations at every position except absolute position 0.
    Position 0 is detected as index 0 of a multi-token (prefill) call; decode steps (T == 1) are all pushed.
    Pass skip_first=False to push every position (used to check that the ONNX mask matters)."""

    def __init__(self, model, steer, skip_first=True):
        self.handles = []
        self.steer = torch.as_tensor(steer)
        self.skip_first = skip_first
        for L, layer in enumerate(model.model.layers):
            vec = self.steer[L]
            if not torch.any(vec != 0):
                continue
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(self._hook(vec)))

    def _hook(self, vec):
        def hook(mod, inp):
            x = inp[0]
            v = vec.to(x.dtype).to(x.device)
            if self.skip_first and x.shape[1] > 1:
                x = x.clone()
                x[:, 1:, :] = x[:, 1:, :] + v
                return (x,)
            return (x + v,)
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.remove()


@torch.no_grad()
def torch_logits(model, ids, steer=None, skip_first=True):
    if steer is None:
        return model(ids).logits[0].float()
    with TorchSteerer(model, steer, skip_first):
        return model(ids).logits[0].float()


@torch.no_grad()
def torch_generate(model, tok, ids, steer=None, max_new_tokens=60, skip_first=True):
    kw = dict(max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    if steer is None:
        out = model.generate(ids, **kw)
    else:
        with TorchSteerer(model, steer, skip_first):
            out = model.generate(ids, **kw)
    new = out[0, ids.shape[1]:]
    return new, tok.decode(new, skip_special_tokens=True)


@torch.no_grad()
def compute_stats(model, tok, prompts=PROMPTS):
    """down_norms[L, n] = ||W_down[:, n]||;  resid_median[L] = median over tokens (pos>=1) of ||h_L|| after layer L."""
    layers = model.model.layers
    down_norms = np.stack([l.mlp.down_proj.weight.float().norm(dim=0).cpu().numpy() for l in layers])  # [L, I]
    caught = {L: [] for L in range(len(layers))}
    hs = []
    for L, l in enumerate(layers):
        hs.append(l.register_forward_hook(lambda m, i, o, L=L: caught[L].append((o[0] if isinstance(o, tuple) else o)[0, 1:].float().norm(dim=-1).cpu())))
    for p in prompts:
        ids, _ = chat_ids(tok, p, next(model.parameters()).device)
        model(ids)
    for h in hs:
        h.remove()
    resid_median = [float(torch.cat(caught[L]).median()) for L in range(len(layers))]
    resid_p10 = [float(torch.cat(caught[L]).quantile(0.1)) for L in range(len(layers))]
    resid_p90 = [float(torch.cat(caught[L]).quantile(0.9)) for L in range(len(layers))]
    return down_norms, {"resid_median": resid_median, "resid_p10": resid_p10, "resid_p90": resid_p90,
                        "num_layers": len(layers), "intermediate": int(down_norms.shape[1]),
                        "prompts": prompts}
