"""Compute dose-normalisation stats on the PyTorch model: ||W_down[:, n]|| and median residual norm after each layer."""
import json, os, time, numpy as np, torch
from common import load_torch, compute_stats, STATS_DIR
t = time.time()
tok, model = load_torch(torch.float32, "cpu")
print("loaded torch model in %.1fs" % (time.time() - t), type(model).__name__, next(model.parameters()).dtype)
down_norms, st = compute_stats(model, tok)
np.save(os.path.join(STATS_DIR, "down_norms.npy"), down_norms)
down_norms.astype(np.float32).tofile(os.path.join(STATS_DIR, "down_norms.bin"))  # what the page reads: float32 [layers, neurons], row by row
with open(os.path.join(STATS_DIR, "stats.json"), "w") as f:
    json.dump(st, f, indent=1)
print("down_norms", down_norms.shape, "min/median/max %.3f %.3f %.3f" % (down_norms.min(), np.median(down_norms), down_norms.max()))
print("resid_median per layer:", [round(x, 2) for x in st["resid_median"]])
