"""
Experiment 7 / §49-50: latency at increasing volume.

    python -m evaluation.latency --tiers 100,1000,10000

Detection latency and investigation latency are reported SEPARATELY (§20):
averaging them would hide that the LLM call dominates investigation time.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from agents.pipeline import FraudSystem
from configs.settings import REPORT_DIR


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def run_tier(n: int) -> dict:
    S = FraudSystem.from_artifacts()     # fresh state per tier
    rows = [r for r in S.replay_pool.head(n).to_dict("records")]
    det, inv = [], []
    t_start = time.perf_counter()
    for r in rows:
        t0 = time.perf_counter()
        sc = S.score(r)
        det.append((time.perf_counter() - t0) * 1000)
        if sc["alert"]:
            t1 = time.perf_counter()
            S.investigate(r, sc)
            inv.append((time.perf_counter() - t1) * 1000)
    wall = time.perf_counter() - t_start
    return {"transactions": n, "alerts": len(inv), "throughput_tx_per_s": n / wall,
            "det_p50_ms": pct(det, 50), "det_p95_ms": pct(det, 95), "det_p99_ms": pct(det, 99),
            "inv_p50_ms": pct(inv, 50), "inv_p95_ms": pct(inv, 95), "inv_p99_ms": pct(inv, 99),
            "narrator": S.narrator.name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="100,1000")
    tiers = [int(x) for x in ap.parse_args().tiers.split(",")]
    res = []
    for n in tiers:
        print(f"tier {n}...", flush=True)
        res.append(run_tier(n))
        print(res[-1], flush=True)
    df = pd.DataFrame(res)
    df.to_csv(REPORT_DIR / "latency.csv", index=False)
    (REPORT_DIR / "latency.md").write_text(df.to_markdown(index=False, floatfmt=".1f"))
    print(df.to_markdown(index=False, floatfmt=".1f"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for col in ["det_p50_ms", "det_p95_ms", "det_p99_ms"]:
            ax[0].plot(df.transactions, df[col], marker="o", label=col.split("_")[1])
        ax[0].set(xscale="log", title="Detection latency", xlabel="transactions replayed", ylabel="ms")
        for col in ["inv_p50_ms", "inv_p95_ms", "inv_p99_ms"]:
            ax[1].plot(df.transactions, df[col], marker="o", label=col.split("_")[1])
        ax[1].set(xscale="log", title=f"Investigation latency ({df.narrator[0]} narrator)", xlabel="transactions replayed")
        for a in ax:
            a.legend(); a.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "latency.png", dpi=150)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
