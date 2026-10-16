"""
Synthetic transaction generator in the project's unified schema (§10).

Why this exists: PaySim and IBM AML need a Kaggle download, and the two datasets
don't share entities. For development, tests and the demo you need ONE coherent
log where the same accounts/devices appear in both the detection and graph
layers. This generator produces that, with labelled typologies named exactly as
in the IBM AML documentation (fan-in, fan-out, cycle, scatter-gather) plus
account takeover and shared-device mule rings.

The real datasets plug in through datasets/loaders.py and map to the same
columns, so nothing downstream changes. Report results on PaySim / IBM AML;
use this for development and the live demo.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from configs.settings import DATA_DIR, RANDOM_SEED

START = datetime(2026, 1, 1)
COLUMNS = [
    "transaction_id", "sender_account", "receiver_account", "amount", "timestamp",
    "transaction_type", "device_id", "ip_address", "merchant_id",
    "sender_balance_before", "sender_balance_after", "fraud_label", "pattern", "pattern_id",
]


class _Gen:
    def __init__(self, n_accounts: int, days: int, seed: int):
        self.rng = np.random.default_rng(seed)
        self.days = days
        self.rows: list[dict] = []
        self.accounts = [f"A{i:05d}" for i in range(n_accounts)]
        r = self.rng
        self.acct = pd.DataFrame({
            "account_id": self.accounts,
            "account_age_days": r.integers(30, 3000, n_accounts),
            "mean_amount": np.exp(r.normal(7.5, 0.8, n_accounts)),   # ~₹1.8k median
            "tx_per_day": r.gamma(2.0, 0.6, n_accounts) + 0.1,
            "location": r.choice(["Pune", "Mumbai", "Delhi", "Bengaluru", "Chennai"], n_accounts),
        }).set_index("account_id")
        self.device_of = {a: f"D{i:05d}" for i, a in enumerate(self.accounts)}
        self.ip_of = {a: f"10.{i // 250 % 250}.{i % 250}.{r.integers(1, 250)}" for i, a in enumerate(self.accounts)}
        self.balance = {a: float(m * r.uniform(20, 60)) for a, m in self.acct["mean_amount"].items()}
        # each account has a small circle of usual beneficiaries
        self.circle = {a: list(r.choice(self.accounts, size=r.integers(2, 6), replace=False)) for a in self.accounts}
        self.merchants = [f"M{i:04d}" for i in range(200)]
        self._next_dev = n_accounts
        # legitimate device sharing (households) and legitimate phone changes -> realistic noise
        for i in range(0, n_accounts // 10, 2):
            self.device_of[self.accounts[i + 1]] = self.device_of[self.accounts[i]]
        self.phone_change = {a: int(r.integers(1, days)) for a in r.choice(self.accounts, n_accounts // 12, replace=False)}
        self.new_phone = {a: f"D9{i:04d}" for i, a in enumerate(self.phone_change)}
        self._pid = 0

    def _device(self, a, ts):
        if a in self.phone_change and (ts - START).days >= self.phone_change[a]:
            return self.new_phone[a]
        return self.device_of.get(a, f"D_{a}")

    def _new_device(self) -> str:
        self._next_dev += 1
        return f"D{self._next_dev:05d}"

    def _add(self, sender, receiver, amount, ts, ttype="TRANSFER", device=None, ip=None,
             merchant=None, fraud=0, pattern="normal", pid=None, drain=False):
        before = self.balance.get(sender, 0.0)
        amount = float(round(amount, 2))
        if drain:  # PaySim-style: fraud empties the account -> balance leakage
            amount = round(max(amount, before), 2)
        after = max(before - amount, 0.0)
        self.balance[sender] = after
        if receiver in self.balance:
            self.balance[receiver] += amount
        self.rows.append(dict(
            sender_account=sender, receiver_account=receiver, amount=amount, timestamp=ts,
            transaction_type=ttype, device_id=device or self._device(sender, ts),
            ip_address=ip or self.ip_of[sender], merchant_id=merchant,
            sender_balance_before=round(before, 2), sender_balance_after=round(after, 2),
            fraud_label=fraud, pattern=pattern, pattern_id=pid,
        ))

    def _rand_time(self, day=None, night=False) -> datetime:
        r = self.rng
        day = r.integers(0, self.days) if day is None else day
        if night or r.random() < 0.05:           # ~5% of legitimate activity is at night
            hour = r.integers(0, 6)
        else:
            hour = int(np.clip(r.normal(14, 3.5), 6, 23))
        return START + timedelta(days=int(day), hours=int(hour), minutes=int(r.integers(0, 60)),
                                 seconds=int(r.integers(0, 60)))

    # ------------------------------------------------------------------ normal
    def normal_activity(self):
        r = self.rng
        for a, row in self.acct.iterrows():
            n = r.poisson(row.tx_per_day * self.days)
            for _ in range(n):
                ts = self._rand_time()
                amt = row.mean_amount * r.lognormal(0, 0.45)
                u = r.random()
                if u < 0.45:
                    self._add(a, self.merchants[r.integers(len(self.merchants))], amt * 0.6, ts,
                              "PAYMENT", merchant=None)
                    self.rows[-1]["merchant_id"] = self.rows[-1]["receiver_account"]
                elif u < 0.90:
                    self._add(a, r.choice(self.circle[a]), amt, ts, "TRANSFER")
                else:
                    self._add(a, r.choice(self.accounts), amt, ts, r.choice(["TRANSFER", "CASH_OUT"]))
        # a little legitimate "noise" that looks odd: salary-day spikes, night owls
        for a in r.choice(self.accounts, size=len(self.accounts) // 4, replace=True):
            self._add(a, r.choice(self.circle[a]), self.acct.at[a, "mean_amount"] * r.uniform(3, 10),
                      self._rand_time(night=r.random() < 0.5), "TRANSFER")

    # --------------------------------------------------------------- typologies
    def _pid_next(self, name):
        self._pid += 1
        return f"{name.upper()}-{self._pid:04d}"

    def account_takeover(self, mules: list[str]):
        r = self.rng
        victim = r.choice(self.accounts)
        pid = self._pid_next("ato")
        dev, ip = self._new_device(), f"185.{r.integers(1, 250)}.{r.integers(1, 250)}.{r.integers(1, 250)}"
        ts = self._rand_time(night=True)
        targets = list(r.choice(mules, size=min(len(mules), r.integers(1, 3)), replace=False))
        for i in range(int(r.integers(4, 12))):
            amt = self.acct.at[victim, "mean_amount"] * r.uniform(3, 9)
            last = i == 0 and r.random() < 0.3
            self._add(victim, r.choice(targets), amt, ts + timedelta(minutes=int(i * r.integers(1, 4))),
                      "TRANSFER", dev, ip, fraud=1, pattern="account_takeover", pid=pid, drain=last)

    def mule_ring(self) -> list[str]:
        """4-6 mule accounts on one shared device; victims fan-in, mules forward to a collector."""
        r = self.rng
        mules = list(r.choice(self.accounts, size=int(r.integers(4, 7)), replace=False))
        dev = self._new_device()
        collector = r.choice(self.accounts)
        pid = self._pid_next("ring")
        day = int(r.integers(1, self.days - 2))
        for m in mules:
            for _ in range(int(r.integers(2, 5))):
                v = r.choice(self.accounts)
                self._add(v, m, self.acct.at[v, "mean_amount"] * r.uniform(2, 5),
                          self._rand_time(day), "TRANSFER", fraud=1, pattern="mule_ring", pid=pid)
            self._add(m, collector, self.balance[m] * 0.9 + 1, self._rand_time(day + 1),
                      r.choice(["TRANSFER", "CASH_OUT"]), device=dev, fraud=1, pattern="mule_ring", pid=pid)
        return mules

    def fan_out(self):
        r = self.rng
        src = r.choice(self.accounts)
        pid = self._pid_next("fanout")
        ts = self._rand_time()
        total = self.acct.at[src, "mean_amount"] * r.uniform(15, 30)
        outs = r.choice(self.accounts, size=int(r.integers(8, 16)), replace=False)
        for i, d in enumerate(outs):
            self._add(src, d, total / len(outs) * r.uniform(0.9, 1.1), ts + timedelta(minutes=i * 3),
                      "TRANSFER", fraud=1, pattern="fan_out", pid=pid)

    def fan_in(self):
        r = self.rng
        dst = r.choice(self.accounts)
        pid = self._pid_next("fanin")
        day = int(r.integers(0, self.days))
        for s in r.choice(self.accounts, size=int(r.integers(8, 16)), replace=False):
            self._add(s, dst, r.uniform(9000, 9900), self._rand_time(day), "TRANSFER",
                      fraud=1, pattern="fan_in", pid=pid)   # structuring just under 10k

    def cycle(self):
        r = self.rng
        nodes = list(r.choice(self.accounts, size=int(r.integers(3, 6)), replace=False))
        pid = self._pid_next("cycle")
        ts = self._rand_time()
        amt = r.uniform(20000, 80000)
        for i, a in enumerate(nodes):
            b = nodes[(i + 1) % len(nodes)]
            amt *= r.uniform(0.95, 0.99)  # small "fees" per hop
            self._add(a, b, amt, ts + timedelta(hours=i * int(r.integers(1, 6))), "TRANSFER",
                      fraud=1, pattern="cycle", pid=pid)

    def scatter_gather(self):
        r = self.rng
        src, dst = r.choice(self.accounts, size=2, replace=False)
        mids = r.choice([a for a in self.accounts if a not in (src, dst)], size=int(r.integers(4, 8)), replace=False)
        pid = self._pid_next("sg")
        ts = self._rand_time()
        for i, m in enumerate(mids):
            amt = r.uniform(8000, 15000)
            self._add(src, m, amt, ts + timedelta(minutes=i * 5), "TRANSFER", fraud=1, pattern="scatter_gather", pid=pid)
            self._add(m, dst, amt * 0.97, ts + timedelta(hours=6, minutes=i * 7), "TRANSFER",
                      fraud=1, pattern="scatter_gather", pid=pid)


def generate(n_accounts=3000, days=30, n_patterns=40, seed=RANDOM_SEED):
    g = _Gen(n_accounts, days, seed)
    g.normal_activity()
    rings = [g.mule_ring() for _ in range(max(2, n_patterns // 6))]
    for _ in range(n_patterns // 2):
        g.account_takeover(rings[g.rng.integers(len(rings))])
    for fn in (g.fan_out, g.fan_in, g.cycle, g.scatter_gather):
        for _ in range(max(2, n_patterns // 8)):
            fn()
    tx = pd.DataFrame(g.rows).sort_values("timestamp", kind="stable").reset_index(drop=True)
    tx.insert(0, "transaction_id", [f"TX-{i:07d}" for i in range(len(tx))])
    accounts = g.acct.reset_index()[["account_id", "account_age_days", "location"]]
    return tx[COLUMNS], accounts


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--accounts", type=int, default=3000)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--patterns", type=int, default=40)
    a = p.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tx, acc = generate(a.accounts, a.days, a.patterns)
    tx.to_parquet(DATA_DIR / "transactions.parquet", index=False)
    acc.to_parquet(DATA_DIR / "accounts.parquet", index=False)
    import json
    (DATA_DIR / "dataset.json").write_text(json.dumps({"dataset": "Synthetic (development only)", "rows": len(tx),
        "fraud_rows": int(tx.fraud_label.sum()), "fraud_rate": float(tx.fraud_label.mean())}, indent=2))
    print(f"{len(tx):,} transactions, {tx.fraud_label.mean():.2%} fraud, {len(acc):,} accounts")
    print(tx[tx.fraud_label == 1].pattern.value_counts().to_string())
