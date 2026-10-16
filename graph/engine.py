"""
Graph analysis layer (§12) — NetworkX, analytics only (no GNN required).

Nodes : Account, Device, Merchant       Edges: SENT_TO, USED_DEVICE, PAID_MERCHANT
Typologies detected, named as in the IBM AML docs: fan_in, fan_out, cycle,
scatter_gather, plus shared-device clustering (mule rings).

"Flagged" accounts = accounts confirmed as perpetrators in PAST cases only
(victims of account takeover are not flagged). At scoring time the engine only
knows labels from before the transaction — no leakage from the test period.
"""
from __future__ import annotations

from collections import deque
from datetime import timedelta

import networkx as nx
import numpy as np
import pandas as pd

VICTIM_SIDE_PATTERNS = {"account_takeover", "paysim_fraud"}  # sender is the victim, not the criminal


def perpetrators(tx: pd.DataFrame) -> set[str]:
    f = tx[tx.fraud_label == 1]
    recv = set(f.receiver_account)
    send = set(f.loc[~f.pattern.isin(VICTIM_SIDE_PATTERNS), "sender_account"])
    # the first leg of a mule ring is victim -> mule; the mules are the receivers there
    ring = f[f.pattern == "mule_ring"]
    send -= set(ring.sender_account) - set(ring.receiver_account)
    return (recv | send) - set(f.merchant_id.dropna())


class GraphEngine:
    def __init__(self, tx: pd.DataFrame, flagged: set[str] | None = None, window_hours: int = 72):
        self.window = timedelta(hours=window_hours)
        self.tx = tx.copy()
        self.tx["timestamp"] = pd.to_datetime(self.tx.timestamp)
        self.flagged: set[str] = set(flagged or ())
        self._build()

    # ------------------------------------------------------------ construction
    def _build(self):
        G = nx.DiGraph()
        t = self.tx
        pay = t.merchant_id.notna()
        transfers = t[~pay].groupby(["sender_account", "receiver_account"]).agg(
            amount=("amount", "sum"), count=("amount", "size"), last=("timestamp", "max")).reset_index()
        for r in transfers.itertuples(index=False):
            G.add_edge(r.sender_account, r.receiver_account, rel="SENT_TO",
                       amount=float(r.amount), count=int(r.count), last=r.last)
        merch = t[pay].groupby(["sender_account", "merchant_id"]).amount.sum().reset_index()
        for r in merch.itertuples(index=False):
            G.add_edge(r.sender_account, r.merchant_id, rel="PAID_MERCHANT", amount=float(r.amount))
            G.nodes[r.merchant_id]["kind"] = "merchant"
        dev = t.groupby(["sender_account", "device_id"]).size().reset_index()
        for a, d, _ in dev.itertuples(index=False):
            G.add_edge(a, d, rel="USED_DEVICE")
            G.nodes[d]["kind"] = "device"
        for n, data in G.nodes(data=True):
            data.setdefault("kind", "account")
        self.G = G
        # account-only undirected view for hop distance (merchants/devices would connect everyone)
        acc_edges = [(u, v) for u, v, d in G.edges(data=True) if d["rel"] == "SENT_TO"]
        self.A = nx.Graph(acc_edges)
        self.device_accounts = t.groupby("device_id").sender_account.apply(set).to_dict()
        self._dist = self._multi_source_bfs()

    def add_transaction(self, row: dict):
        """Incremental update for the live API path."""
        self.tx = pd.concat([self.tx, pd.DataFrame([row]).assign(timestamp=pd.to_datetime(row["timestamp"]))],
                            ignore_index=True)
        s, r, d = row["sender_account"], row["receiver_account"], row["device_id"]
        merchant = row.get("merchant_id")
        if isinstance(merchant, str) and merchant:          # NaN from parquet is truthy: check the type
            self.G.add_edge(s, merchant, rel="PAID_MERCHANT", amount=row["amount"])
            self.G.nodes[merchant]["kind"] = "merchant"
            self.G.add_node(r)
        else:
            e = self.G.get_edge_data(s, r) or {"amount": 0.0, "count": 0}
            self.G.add_edge(s, r, rel="SENT_TO", amount=e["amount"] + row["amount"],
                            count=e["count"] + 1, last=pd.to_datetime(row["timestamp"]))
            self.A.add_edge(s, r)
        self.G.add_edge(s, d, rel="USED_DEVICE")
        self.G.nodes[d]["kind"] = "device"
        for n in (s, r):
            self.G.nodes[n].setdefault("kind", "account")
        self.device_accounts.setdefault(d, set()).add(s)

    def flag(self, accounts):
        self.flagged |= set(accounts)
        self._dist = self._multi_source_bfs()

    def _multi_source_bfs(self, cap: int = 6) -> dict[str, int]:
        dist, q = {}, deque()
        for f in self.flagged:
            if f in self.A:
                dist[f] = 0
                q.append(f)
        while q:
            u = q.popleft()
            if dist[u] >= cap:
                continue
            for v in self.A.neighbors(u):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    # ---------------------------------------------------------------- queries
    def hops_to_flagged(self, acct: str, exclude_self: bool = False) -> int | None:
        if not exclude_self:
            return self._dist.get(acct)
        nb = [self._dist[v] for v in self.A.neighbors(acct) if v in self._dist] if acct in self.A else []
        return (min(nb) + 1) if nb else None

    def flagged_neighbours(self, acct: str, k: int = 1) -> int:
        if acct not in self.A:
            return 0
        reach = nx.single_source_shortest_path_length(self.A, acct, cutoff=k)
        return sum(1 for n in reach if n != acct and n in self.flagged)

    def shared_device_accounts(self, device: str, acct: str) -> list[str]:
        return sorted(self.device_accounts.get(device, set()) - {acct})

    def _window(self, ts) -> pd.DataFrame:
        ts = pd.to_datetime(ts)
        t = self.tx
        return t[(t.timestamp >= ts - self.window) & (t.timestamp <= ts) & t.merchant_id.isna()]

    def detect_patterns(self, sender: str, receiver: str, ts, k: int = 2) -> dict:
        w = self._window(ts)
        fan_out = w[w.sender_account == sender].receiver_account.nunique()
        fan_in = w[w.receiver_account == receiver].sender_account.nunique()
        patterns = []
        if fan_out >= 6:
            patterns.append({"pattern": "fan_out", "account": sender, "distinct_receivers": int(fan_out)})
        if fan_in >= 6:
            patterns.append({"pattern": "fan_in", "account": receiver, "distinct_senders": int(fan_in)})

        # scatter-gather: sender -> several intermediaries -> one common collector
        mids = set(w[w.sender_account == sender].receiver_account)
        if len(mids) >= 3:
            onward = w[w.sender_account.isin(mids) & (w.receiver_account != sender)]
            by_collector = onward.groupby("receiver_account").sender_account.nunique()
            hit = by_collector[by_collector >= 3]
            if len(hit):
                patterns.append({"pattern": "scatter_gather", "account": sender,
                                 "collector": hit.idxmax(), "intermediaries": int(hit.max())})

        # cycles through the sender or receiver, bounded length, within the window
        wg = nx.DiGraph(list(zip(w.sender_account, w.receiver_account)))
        for node in (sender, receiver):
            if node not in wg:
                continue
            local = wg.subgraph(nx.single_source_shortest_path_length(wg.to_undirected(as_view=True), node, cutoff=5))
            for c in nx.simple_cycles(local, length_bound=6):
                if node in c and len(c) >= 3:
                    patterns.append({"pattern": "cycle", "account": node, "length": len(c), "cycle": c})
                    break
            if any(p["pattern"] == "cycle" for p in patterns):
                break
        return {"fan_out_count": int(fan_out), "fan_in_count": int(fan_in), "patterns": patterns}

    def k_hop_subgraph(self, accounts: list[str], k: int = 2, max_nodes: int = 60) -> dict:
        """Account/device neighbourhood for the evidence bundle and the UI graph view."""
        nodes: dict[str, int] = {}
        for a in accounts:
            if a in self.A:
                for n, d in nx.single_source_shortest_path_length(self.A, a, cutoff=k).items():
                    nodes[n] = min(d, nodes.get(n, 99))
        # keep the closest nodes and every flagged one
        keep = sorted(nodes, key=lambda n: (n not in accounts, n not in self.flagged, nodes[n]))[:max_nodes]
        keep_set = set(keep)
        edges = []
        for u in keep:
            for v, d in self.G[u].items():
                if d["rel"] == "SENT_TO" and v in keep_set:
                    edges.append({"source": u, "target": v, "rel": "SENT_TO",
                                  "amount": round(d["amount"], 2), "count": d["count"]})
                elif d["rel"] == "USED_DEVICE" and len(self.device_accounts.get(v, ())) > 1:
                    edges.append({"source": u, "target": v, "rel": "USED_DEVICE"})
                    keep_set.add(v)
        return {
            "nodes": [{"id": n, "kind": self.G.nodes[n].get("kind", "account") if n in self.G else "account",
                       "flagged": n in self.flagged, "focus": n in accounts, "hops": nodes.get(n)} for n in keep_set],
            "edges": edges,
            "truncated": len(nodes) > max_nodes,
        }

    # ------------------------------------------------------------ evidence + score
    def evidence(self, sender: str, receiver: str, device: str, ts, k: int = 2) -> dict:
        shared = self.shared_device_accounts(device, sender)
        pats = self.detect_patterns(sender, receiver, ts, k)
        hops = [h for h in (self.hops_to_flagged(sender, True), self.hops_to_flagged(receiver, True)) if h is not None]
        return {
            "k_hop": k,
            "sender_is_flagged": sender in self.flagged,
            "receiver_is_flagged": receiver in self.flagged,
            "receiver_connected_to_flagged_accounts": self.flagged_neighbours(receiver, k),
            "sender_connected_to_flagged_accounts": self.flagged_neighbours(sender, k),
            "hops_to_nearest_flagged_entity": min(hops) if hops else None,
            "shared_device_with_other_accounts": len(shared),
            "shared_device_accounts": shared[:10],
            "fan_in_count_window": pats["fan_in_count"],
            "fan_out_count_window": pats["fan_out_count"],
            "detected_patterns": pats["patterns"],
        }

    @staticmethod
    def score(ev: dict) -> float:
        """Graph suspicion score in [0,1]: noisy-OR over independent graph signals.

        The individual probabilities are a documented heuristic prior; the learned
        fusion layer (fusion/fusion.py) re-weights this score against labels, so
        they don't need to be perfectly calibrated.
        """
        p = []
        if ev["receiver_is_flagged"]:
            p.append(0.7)
        if ev["sender_is_flagged"]:
            p.append(0.5)
        n = ev["receiver_connected_to_flagged_accounts"]
        if n:
            p.append(min(0.15 * n, 0.5))
        if ev["shared_device_with_other_accounts"] >= 2:
            p.append(min(0.15 * ev["shared_device_with_other_accounts"], 0.6))
        for pat in ev["detected_patterns"]:
            p.append({"cycle": 0.6, "scatter_gather": 0.6, "fan_in": 0.4, "fan_out": 0.4}[pat["pattern"]])
        return float(1 - np.prod([1 - x for x in p])) if p else 0.0

    def batch_scores(self, tx: pd.DataFrame, with_patterns: bool = False) -> np.ndarray:
        """Fast per-row graph score for training/evaluating the fusion layer.

        Skips the (expensive) windowed pattern search unless with_patterns=True;
        the per-case investigation always runs the full evidence() query.
        """
        out = np.zeros(len(tx))
        for i, r in enumerate(tx.itertuples(index=False)):
            ev = {
                "receiver_is_flagged": r.receiver_account in self.flagged,
                "sender_is_flagged": r.sender_account in self.flagged,
                "receiver_connected_to_flagged_accounts": self.flagged_neighbours(r.receiver_account, 1),
                "shared_device_with_other_accounts": len(self.shared_device_accounts(r.device_id, r.sender_account)),
                "detected_patterns": self.detect_patterns(r.sender_account, r.receiver_account, r.timestamp)["patterns"]
                if with_patterns else [],
            }
            out[i] = self.score(ev)
        return out
