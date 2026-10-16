"""
Adapters that map the real public datasets onto the unified schema (§10).

PaySim  : https://www.kaggle.com/datasets/ealaxi/paysim1
IBM AML : https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
          (HI-Small_Trans.csv, plus HI-Small_Patterns.txt beside it for typology labels)

    python -m datasets.loaders paysim  datasets/data/raw/PS_20174392719_1491204439457_log.csv --frac 0.1
    python -m datasets.loaders ibm_aml datasets/data/raw/HI-Small_Trans.csv --frac 0.2

Both write datasets/data/transactions.parquet (+ accounts.parquet, dataset.json),
which every other module reads. --frac samples LEGITIMATE activity only; every
fraud / laundering transaction is always kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from configs.settings import DATA_DIR
from datasets.synthetic import COLUMNS

PAYSIM_START = datetime(2026, 1, 1)


# --------------------------------------------------------------------- PaySim
def load_paysim(path: str, frac: float | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if frac and frac < 1:
        # PaySim customers almost never repeat, so row sampling loses no per-account history
        keep = (df.isFraud == 1) | (np.random.default_rng(0).random(len(df)) < frac)
        df = df[keep]
    within = df.groupby("step").cumcount()
    size = df.groupby("step")["step"].transform("size")
    secs = (within / size * 3600).astype(int)          # spread rows inside each simulated hour
    out = pd.DataFrame({
        "sender_account": df.nameOrig.to_numpy(),
        "receiver_account": df.nameDest.to_numpy(),
        "amount": df.amount.to_numpy(),
        "timestamp": pd.Timestamp(PAYSIM_START) + pd.to_timedelta(df.step.to_numpy(), unit="h")
                     + pd.to_timedelta(secs.to_numpy(), unit="s"),
        "transaction_type": df["type"].to_numpy(),
        "device_id": ("DEV_" + df.nameOrig).to_numpy(),   # PaySim has no device data
        "ip_address": None,
        "merchant_id": np.where(df.nameDest.str.startswith("M"), df.nameDest, None),
        "sender_balance_before": df.oldbalanceOrg.to_numpy(),   # LEAKAGE risk (§35)
        "sender_balance_after": df.newbalanceOrig.to_numpy(),
        "fraud_label": df.isFraud.astype(int).to_numpy(),
        "pattern": np.where(df.isFraud == 1, "paysim_fraud", "normal"),
        "pattern_id": None,
    })
    return _finish(out)


# -------------------------------------------------------------------- IBM AML
def _parse_ibm_patterns(path: Path) -> dict[tuple, tuple[str, str]]:
    """(timestamp, from_account, to_account, amount_paid) -> (pattern_name, pattern_id)."""
    lookup, name, pid = {}, None, 0
    for line in path.read_text().splitlines():
        m = re.match(r"BEGIN LAUNDERING ATTEMPT - ([A-Z][A-Z\-]*)", line.strip())
        if m:
            name, pid = m.group(1).lower().replace("-", "_"), pid + 1
            continue
        if line.startswith("END"):
            name = None
            continue
        if name and line.strip():
            f = line.split(",")
            if len(f) >= 8:
                lookup[(f[0].strip(), f[2].strip(), f[4].strip(), round(float(f[7]), 2))] = (name, f"{name.upper()}-{pid:05d}")
    return lookup


def _usd_rates(df: pd.DataFrame) -> dict[str, float]:
    """Exchange rates implied by the data itself: rows paid in X and received in US Dollar."""
    rates = {"US Dollar": 1.0}
    cross = df[(df.cur_recv == "US Dollar") & (df.cur_paid != "US Dollar") & (df.amt_paid > 0)]
    for cur, g in cross.groupby("cur_paid"):
        rates[cur] = float((g.amt_recv / g.amt_paid).median())
    cross = df[(df.cur_paid == "US Dollar") & (df.cur_recv != "US Dollar") & (df.amt_recv > 0)]
    for cur, g in cross.groupby("cur_recv"):
        rates.setdefault(cur, float((g.amt_paid / g.amt_recv).median()))
    return rates


def _keep_account(acct: pd.Series, frac: float) -> np.ndarray:
    """Deterministic account-level sample, so every kept sender keeps its FULL history."""
    u = pd.unique(acct)
    h = dict(zip(u, (int(hashlib.md5(a.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF for a in u)))
    return (acct.map(h) < frac).to_numpy()


def load_ibm_aml(path: str, frac: float | None = None, chunksize: int = 1_000_000) -> pd.DataFrame:
    path = Path(path)
    cols = ["ts", "from_bank", "from_acct", "to_bank", "to_acct", "amt_recv", "cur_recv",
            "amt_paid", "cur_paid", "fmt", "label"]
    parts, total = [], 0
    # read in chunks and sample each one, so memory stays proportional to --frac, not file size
    for ch in pd.read_csv(path, dtype=str, chunksize=chunksize):
        ch.columns = cols
        total += len(ch)
        for c in ["from_bank", "to_bank", "from_acct", "to_acct", "ts"]:
            ch[c] = ch[c].str.strip()
        ch["label"] = ch.label.astype(int)
        if frac and frac < 1:
            snd = ch.from_bank.str.lstrip("0") + "_" + ch.from_acct
            ch = ch[(ch.label == 1).to_numpy() | _keep_account(snd, frac)]
        ch["amt_recv"] = ch.amt_recv.astype(float)
        ch["amt_paid"] = ch.amt_paid.astype(float)
        parts.append(ch)
        print(f"  read {total:,} rows, kept {sum(len(x) for x in parts):,}", flush=True)
    df = pd.concat(parts, ignore_index=True)
    del parts
    sender = df.from_bank.str.lstrip("0") + "_" + df.from_acct
    receiver = df.to_bank.str.lstrip("0") + "_" + df.to_acct

    rates = _usd_rates(df)
    unknown = sorted(set(df.cur_paid) - set(rates))
    if unknown:
        print(f"warning: no USD rate found in data for {unknown}; amounts left unconverted")
    amount_usd = df.amt_paid * df.cur_paid.map(rates).fillna(1.0)

    pat_file = path.with_name(path.name.replace("_Trans.csv", "_Patterns.txt"))
    patterns = _parse_ibm_patterns(pat_file) if pat_file.exists() else {}
    keys = zip(df.ts, df.from_acct, df.to_acct, df.amt_paid.round(2))
    pat = [patterns.get(k, (None, None)) for k in keys]
    if patterns:
        matched = sum(1 for p in pat if p[0])
        print(f"typology labels matched for {matched:,} of {int(df.label.sum()):,} laundering rows")
    out = pd.DataFrame({
        "sender_account": sender.to_numpy(),
        "receiver_account": receiver.to_numpy(),
        "amount": amount_usd.round(2).to_numpy(),
        "timestamp": pd.to_datetime(df.ts, format="%Y/%m/%d %H:%M").to_numpy(),
        "transaction_type": df.fmt.to_numpy(),
        "device_id": ("DEV_" + sender).to_numpy(),        # IBM AML has no device data
        "ip_address": None,
        "merchant_id": None,
        "sender_balance_before": np.nan,
        "sender_balance_after": np.nan,
        "fraud_label": df.label.to_numpy(),
        "pattern": [p[0] or ("laundering_unclassified" if l else "normal") for p, l in zip(pat, df.label)],
        "pattern_id": [p[1] for p in pat],
    })
    return _finish(out)


# ---------------------------------------------------------------------- common
def _finish(out: pd.DataFrame) -> pd.DataFrame:
    out = out.sort_values("timestamp", kind="stable").reset_index(drop=True)
    out.insert(0, "transaction_id", [f"TX-{i:08d}" for i in range(len(out))])
    return out[COLUMNS]


def accounts_from_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    ids = pd.unique(pd.concat([tx.sender_account, tx.receiver_account]))
    # Neither dataset has account age; a constant keeps the feature neutral (documented limitation)
    return pd.DataFrame({"account_id": ids, "account_age_days": 365, "location": None})


def save(tx: pd.DataFrame, name: str, source: str, frac):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tx.to_parquet(DATA_DIR / "transactions.parquet", index=False)
    accounts_from_transactions(tx).to_parquet(DATA_DIR / "accounts.parquet", index=False)
    meta = {"dataset": name, "source": str(source), "sample_frac": frac, "rows": len(tx),
            "fraud_rows": int(tx.fraud_label.sum()), "fraud_rate": float(tx.fraud_label.mean()),
            "start": str(tx.timestamp.min()), "end": str(tx.timestamp.max())}
    (DATA_DIR / "dataset.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["paysim", "ibm_aml"])
    ap.add_argument("path")
    ap.add_argument("--frac", type=float, default=None, help="fraction of legitimate activity to keep")
    a = ap.parse_args()
    tx = load_paysim(a.path, a.frac) if a.kind == "paysim" else load_ibm_aml(a.path, a.frac)
    save(tx, {"paysim": "PaySim", "ibm_aml": "IBM AML " + Path(a.path).name.split("_")[0]}[a.kind], a.path, a.frac)
