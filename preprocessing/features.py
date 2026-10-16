"""
Feature engineering (§11).

Every feature is computed from information available at the time of the
transaction (history strictly before it, or the current row itself). No future
rows are ever used, so the same function is safe for training and for online
scoring of a single new transaction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from configs.settings import USE_BALANCE_FEATURES

TX_TYPES = ["TRANSFER", "PAYMENT", "CASH_OUT", "CASH_IN", "DEBIT",                  # PaySim
            "ACH", "Cheque", "Credit Card", "Wire", "Cash", "Reinvestment", "Bitcoin"]  # IBM AML


def _tname(t: str) -> str:
    return "type_" + t.lower().replace(" ", "_")

BEHAVIOUR_FEATURES = [  # fed to the Isolation Forest: "abnormal for THIS account"
    "amount_zscore", "amount_to_avg_ratio", "tx_count_1h", "tx_count_24h",
    "log_time_since_last", "hour", "amount_sum_1h_ratio",
]

FEATURE_GROUPS = {
    "transaction": ["log_amount", "amount_zscore", "amount_to_avg_ratio", "hour", "is_night"]
                   + [_tname(t) for t in TX_TYPES],
    "behavioural": ["tx_count_1h", "tx_count_24h", "amount_sum_1h_ratio", "new_beneficiary",
                    "beneficiary_frequency", "sender_prior_tx_count"],
    "temporal": ["log_time_since_last", "burst_10min"],
    "graph": ["sender_distinct_receivers", "receiver_distinct_senders", "receiver_in_count_24h",
              "device_account_count"],
    "device_account": ["account_age_days", "device_change", "ip_change"],
}
if USE_BALANCE_FEATURES:
    FEATURE_GROUPS["balance"] = ["balance_drain_ratio", "balance_zeroed"]

FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]
RAW_FEATURES = ["log_amount", "hour"] + [_tname(t) for t in TX_TYPES]  # "no feature engineering" baseline


def _rolling_count(df: pd.DataFrame, key: str, window: str, value: str | None = None) -> pd.Series:
    """Rolling count (or sum of `value`) per key over a time window, including the current row."""
    col = value or "_one"
    tmp = df[[key, "timestamp"]].copy()
    tmp[col] = df[value] if value else 1.0
    tmp["_pos"] = np.arange(len(tmp))
    tmp = tmp.sort_values([key, "timestamp"], kind="stable")
    r = tmp.groupby(key, sort=False).rolling(window, on="timestamp")[col].sum()
    tmp["_r"] = r.to_numpy()
    return tmp.sort_values("_pos")["_r"].to_numpy()


def _distinct_so_far(df: pd.DataFrame, key: str, other: str) -> np.ndarray:
    """Number of distinct `other` values seen for `key` up to and including this row."""
    first = ~df.duplicated([key, other])
    return first.groupby(df[key]).cumsum().to_numpy()


def engineer_features(tx: pd.DataFrame, accounts: pd.DataFrame | None = None) -> pd.DataFrame:
    df = tx.sort_values("timestamp", kind="stable").copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    s = df.groupby("sender_account", sort=False)

    f = pd.DataFrame(index=df.index)
    f["log_amount"] = np.log1p(df.amount)

    # per-sender running stats from PRIOR transactions only (shift(1))
    # vectorised expanding mean/std over prior rows: cumulative sums minus the current row
    n = s.cumcount()
    s1 = s.amount.cumsum() - df.amount
    s2 = (df.amount ** 2).groupby(df.sender_account, sort=False).cumsum() - df.amount ** 2
    prior_mean = s1 / n.where(n > 0)
    prior_std = np.sqrt(((s2 - s1 ** 2 / n.where(n > 0)) / (n - 1).where(n > 1)).clip(lower=0))
    f["sender_prior_tx_count"] = n
    enough = f.sender_prior_tx_count >= 3
    f["amount_zscore"] = np.where(enough, (df.amount - prior_mean) / prior_std.clip(lower=1.0), 0.0)
    f["amount_zscore"] = f.amount_zscore.clip(-10, 50)
    f["amount_to_avg_ratio"] = np.where(enough, df.amount / prior_mean.clip(lower=1.0), 1.0)

    f["hour"] = df.timestamp.dt.hour
    f["is_night"] = f.hour.isin([0, 1, 2, 3, 4, 5]).astype(int)
    for t in TX_TYPES:
        f[_tname(t)] = (df.transaction_type == t).astype(int)

    f["tx_count_1h"] = _rolling_count(df, "sender_account", "1h")
    f["tx_count_24h"] = _rolling_count(df, "sender_account", "24h")
    f["burst_10min"] = _rolling_count(df, "sender_account", "10min")
    sum_1h = _rolling_count(df, "sender_account", "1h", value="amount")
    f["amount_sum_1h_ratio"] = np.where(enough, sum_1h / prior_mean.clip(lower=1.0), 1.0)

    gap = s.timestamp.diff().dt.total_seconds()
    f["log_time_since_last"] = np.log1p(gap.fillna(30 * 86400))

    pair = df.groupby(["sender_account", "receiver_account"], sort=False)
    f["beneficiary_frequency"] = pair.cumcount()
    f["new_beneficiary"] = (f.beneficiary_frequency == 0).astype(int)

    f["sender_distinct_receivers"] = _distinct_so_far(df, "sender_account", "receiver_account")
    f["receiver_distinct_senders"] = _distinct_so_far(df, "receiver_account", "sender_account")
    f["receiver_in_count_24h"] = _rolling_count(df, "receiver_account", "24h")
    f["device_account_count"] = _distinct_so_far(df, "device_id", "sender_account")

    prev_dev = s.device_id.shift(1)
    f["device_change"] = (prev_dev.notna() & (prev_dev != df.device_id)).astype(int)
    prev_ip = s.ip_address.shift(1)
    f["ip_change"] = (prev_ip.notna() & (prev_ip != df.ip_address)).astype(int)

    if accounts is not None and len(accounts):
        age = accounts.set_index("account_id").account_age_days
        f["account_age_days"] = df.sender_account.map(age).fillna(365).to_numpy()
    else:
        f["account_age_days"] = 365

    if USE_BALANCE_FEATURES:
        before = df.sender_balance_before.fillna(0)
        f["balance_drain_ratio"] = np.where(before > 0, df.amount / before, 0)
        f["balance_zeroed"] = (df.sender_balance_after.fillna(1) == 0).astype(int)

    return f.loc[tx.index, FEATURES].astype(float)


def features_for_new(tx_row: dict, history: pd.DataFrame, accounts: pd.DataFrame | None = None) -> pd.Series:
    """Online path: features for one new transaction given the stored history.

    Only rows sharing the sender, receiver or device can influence any feature,
    so we slice those out, append the new row and run the same batch function.
    """
    h = history
    rel = h[(h.sender_account == tx_row["sender_account"]) |
            (h.receiver_account == tx_row["receiver_account"]) |
            (h.device_id == tx_row["device_id"])]
    new = pd.DataFrame([tx_row])
    new["timestamp"] = pd.to_datetime(new["timestamp"])
    both = pd.concat([rel, new], ignore_index=True)
    feats = engineer_features(both, accounts)
    return feats.iloc[-1]
