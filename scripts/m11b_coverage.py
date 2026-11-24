"""Milestone 11b: coverage of NUTS intervals on synthetic data.

The small lensed setup of tests/test_fit.py (host Sersic + sky, known truth)
with N_REAL fresh noise realizations. For each: LM fit, then NUTS (2 chains x
(200 warm-up + 300 samples), Laplace mass matrix). Counts how often the truth
lies inside the central 68% and 95% intervals (expected ~68% and ~95%), and
collects (median - truth) / sigma, which should be ~N(0, 1).

Prints one line per realization. Writes outputs/m11b_coverage.txt and .png.

Run from the repository root (long: ~20-30 min; behind a memory guard):
    python scripts/m11b_coverage.py
"""

import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, "tests")

import jax  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import forwardgal  # noqa: E402,F401
from forwardgal.fit import fit  # noqa: E402
from forwardgal.sample import nuts  # noqa: E402
from test_fit import FLUX, SKY, TRUTH, setup  # noqa: E402

N_REAL = 20
rows = []
t_all = time.perf_counter()
for k in range(N_REAL):
    t0 = time.perf_counter()
    model, (bx, by) = setup(dict(dx=0.0, **TRUTH), seed=100 + k)  # new noise each time
    truth = {"host.x0": bx, "host.y0": by, "host.R_e": TRUTH["R_e"], "host.n": TRUTH["n"],
             "host.q": TRUTH["q"], "amp:host": FLUX, "amp:sky": SKY}
    res = fit(model, n_starts=1)
    post = nuts(model, res, num_warmup=200, num_samples=300, num_chains=2, seed=k, progress=False)
    for name, t in truth.items():
        x = post.column(name)
        lo68, med, hi68 = np.percentile(x, [15.865, 50, 84.135])
        lo95, hi95 = np.percentile(x, [2.275, 97.725])
        rows.append((k, name, lo68 <= t <= hi68, lo95 <= t <= hi95, (med - t) / np.std(x)))
    print(f"realization {k + 1:2d}/{N_REAL}: {time.perf_counter() - t0:5.0f} s, divergences {post.divergences}, "
          f"max R-hat {np.nanmax(post.r_hat):.3f}, min n_eff {np.min(post.n_eff):.0f}", flush=True)
    del model, res, post
    jax.clear_caches()  # each realization compiles a new model: free the old ones
    gc.collect()

names = list(dict.fromkeys(r[1] for r in rows))
lines = [f"{N_REAL} realizations, {time.perf_counter() - t_all:.0f} s total", "",
         f"{'parameter':12s} {'in 68%':>7s} {'in 95%':>7s} {'mean z':>7s} {'std z':>6s}"]
for n in names:
    sel = [r for r in rows if r[1] == n]
    z = np.array([r[4] for r in sel])
    lines.append(f"{n:12s} {np.mean([r[2] for r in sel]):7.0%} {np.mean([r[3] for r in sel]):7.0%} "
                 f"{z.mean():7.2f} {z.std():6.2f}")
c68 = np.mean([r[2] for r in rows])
c95 = np.mean([r[3] for r in rows])
lines += ["", f"all parameters: truth inside 68% interval {c68:.0%} (expected 68%), "
              f"inside 95% interval {c95:.0%} (expected 95%) over {len(rows)} checks"]
report = "\n".join(lines)
print("\n" + report)
Path("outputs").mkdir(exist_ok=True)
Path("outputs/m11b_coverage.txt").write_text(report + "\n")

z = np.array([r[4] for r in rows])
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(z, bins=np.linspace(-4, 4, 25), density=True, alpha=0.7, label="(median - truth) / sigma, all parameters")
g = np.linspace(-4, 4, 200)
ax.plot(g, np.exp(-g**2 / 2) / np.sqrt(2 * np.pi), "k-", label="N(0, 1)")
ax.set_title(f"M11b coverage: 68% -> {c68:.0%}, 95% -> {c95:.0%}", fontsize=10)
ax.legend(fontsize=8)
fig.savefig("outputs/m11b_coverage.png", dpi=110, bbox_inches="tight")
print("saved: outputs/m11b_coverage.txt, outputs/m11b_coverage.png")
