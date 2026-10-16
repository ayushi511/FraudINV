"""
Near-real-time simulated streaming (§19): replay held-out transactions into the
running API at a controlled rate.   (No Kafka: a direct API call + async agents.)

    uvicorn backend.app:app            # terminal 1
    python -m scripts.replay --rate 5 --n 200   # terminal 2 (5 tx/second)
"""
import argparse
import time

import httpx
import numpy as np

from agents.pipeline import FraudSystem

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--rate", type=float, default=5.0, help="transactions per second")
ap.add_argument("--n", type=int, default=200)
args = ap.parse_args()

pool = FraudSystem.from_artifacts().replay_pool.head(args.n)
with httpx.Client(base_url=args.url, timeout=60) as c:
    for row in pool.to_dict("records"):
        t0 = time.perf_counter()
        body = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()
                if k not in ("pattern", "pattern_id")}
        body["timestamp"] = str(body["timestamp"])
        r = c.post("/transactions", json=body).json()
        flag = "ALERT" if r.get("alert") else "     "
        print(f"{flag} {r['transaction_id']}  clf={r['classifier_score']:.3f}  beh={r['behaviour_score']:.3f}  "
              f"graph={r['graph_score']:.3f}  {r['detection_ms']:.0f}ms")
        time.sleep(max(0.0, 1 / args.rate - (time.perf_counter() - t0)))
