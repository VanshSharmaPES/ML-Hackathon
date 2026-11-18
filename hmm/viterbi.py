"""HMM / Viterbi route decoding over the classifier's candidate roads.

Hidden state at GPS point t: which of its candidate arcs (up to K, from
pathsnap_p1.features.generate_features) the vehicle is on.

Emission score   log P(candidate is correct), from the classifier, times `emission_weight`.
Transition cost  between the candidate at t-1 and the candidate at t, by road connectivity:

    same      the same arc                                   0
    next      arc continues from the previous one            cost_next       (prev.to == cur.from)
    two_hop   one arc in between (short arcs, fast driving)  cost_two_hop
    twin      the reverse arc of the same road               cost_twin       (direction flip)
    other     anything else                                  cost_other
    reset     after a time gap > gap_seconds, or at the start of the trip: no transition cost

The decoder maximises the sum of emission scores minus transition costs along the
sequence (Viterbi). A single-point flip to the reverse twin costs 2 x cost_twin, so it
only survives if the classifier is very sure of it. Costs are tuned on training trips only
(hmm/tune_decoder.py) and stored in artifacts/hmm_config.json.

    decode(candidates, probs, trip, cfg) -> {observation_id: arc_id}

`trip` only needs `trip.arcs` with columns arc_id, from_node, to_node. The reference
route is never used here.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

SAME, NEXT, TWO_HOP, TWIN, OTHER, RESET = range(6)
DEFAULTS = {"emission_weight": 1.0, "cost_next": 0.5, "cost_two_hop": 1.5, "cost_twin": 3.0,
            "cost_other": 4.0, "gap_seconds": 5.0, "probability_floor": 1e-6}


def _road_graph(arcs: pd.DataFrame):
    frm = dict(zip(arcs.arc_id.to_numpy(), arcs.from_node.to_numpy()))
    to = dict(zip(arcs.arc_id.to_numpy(), arcs.to_node.to_numpy()))
    out_nodes = defaultdict(set)  # node -> nodes reachable by one arc
    for f, t in zip(arcs.from_node.to_numpy(), arcs.to_node.to_numpy()):
        out_nodes[f].add(t)
    return frm, to, out_nodes


def prepare(candidates: pd.DataFrame, probs: np.ndarray, arcs: pd.DataFrame, gap_seconds: float):
    """Arrange one trip's candidates as fixed-width arrays plus a transition-category tensor.

    The categories depend only on the road graph and the candidates, not on the costs, so
    they are computed once and reused when tuning the costs.
    """
    c = candidates[["observation_id", "sequence_index", "candidate_arc_id", "time_gap_s"]].assign(p=probs)
    c = c.sort_values(["sequence_index", "p"], ascending=[True, False], kind="stable")
    groups = c.groupby("sequence_index", sort=True)
    obs_ids = groups.observation_id.first().to_numpy()
    gaps = groups.time_gap_s.first().to_numpy()
    k = int(groups.size().max())
    t_len = len(obs_ids)
    arc = np.full((t_len, k), -1, dtype=np.int64)
    prob = np.zeros((t_len, k))
    slot = groups.cumcount().to_numpy()
    row = np.repeat(np.arange(t_len), groups.size().to_numpy())
    arc[row, slot] = c.candidate_arc_id.to_numpy()
    prob[row, slot] = c.p.to_numpy()

    frm, to, out_nodes = _road_graph(arcs)
    cat = np.full((t_len, k, k), OTHER, dtype=np.int8)
    cat[0] = RESET
    for t in range(1, t_len):
        if not np.isfinite(gaps[t]) or gaps[t] > gap_seconds or gaps[t] <= 0:
            cat[t] = RESET
            continue
        prev, cur = arc[t - 1], arc[t]
        for i, a in enumerate(prev):
            if a < 0:
                continue
            a_from, a_to = frm[a], to[a]
            reach2 = None
            for j, b in enumerate(cur):
                if b < 0:
                    continue
                if a == b:
                    cat[t, i, j] = SAME
                elif frm[b] == a_to and to[b] == a_from:
                    cat[t, i, j] = TWIN
                elif frm[b] == a_to:
                    cat[t, i, j] = NEXT
                else:
                    if reach2 is None:
                        reach2 = out_nodes.get(a_to, set())
                    if frm[b] in reach2:
                        cat[t, i, j] = TWO_HOP
    return {"obs_ids": obs_ids, "arc": arc, "prob": prob, "cat": cat}


def viterbi(prepared: dict, cfg: dict) -> np.ndarray:
    """Most likely candidate arc per point. Returns arc IDs aligned with prepared['obs_ids']."""
    p = {**DEFAULTS, **(cfg or {})}
    arc, prob, cat = prepared["arc"], prepared["prob"], prepared["cat"]
    valid = arc >= 0
    emit = np.where(valid, p["emission_weight"] * np.log(np.clip(prob, p["probability_floor"], 1.0)), -np.inf)
    cost = np.array([0.0, p["cost_next"], p["cost_two_hop"], p["cost_twin"], p["cost_other"], 0.0])
    trans = -cost[cat]                      # (T, K, K): score of moving from prev slot i to cur slot j
    t_len, k = arc.shape
    back = np.zeros((t_len, k), dtype=np.int64)
    score = emit[0].copy()
    for t in range(1, t_len):
        total = score[:, None] + trans[t]   # (K prev, K cur)
        back[t] = np.argmax(total, axis=0)
        score = total[back[t], np.arange(k)] + emit[t]
    path = np.empty(t_len, dtype=np.int64)
    path[-1] = int(np.argmax(score))
    for t in range(t_len - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return arc[np.arange(t_len), path]


def decode(candidates: pd.DataFrame, probs, trip, cfg: dict) -> dict[int, int]:
    p = {**DEFAULTS, **(cfg or {})}
    prepared = prepare(candidates, np.asarray(probs, dtype=float), trip.arcs, p["gap_seconds"])
    arcs = viterbi(prepared, p)
    return {int(o): int(a) for o, a in zip(prepared["obs_ids"], arcs) if a >= 0}
