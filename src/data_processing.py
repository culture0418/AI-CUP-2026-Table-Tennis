"""Data processing — TT-domain knowledge encoding.

Contains:
  - compute_oppair_contexts(): the 58-dim opponent-pair LOO context (V25-A innovation)
    which encodes per-match per-player tactical statistics in a leakage-free way.

Per-stroke encoding (encode_df), rally building (build_rallies), k-sampling
(sample_k) and the PyTorch Dataset/DataLoader are closures inside training.py
because they depend on per-seed vocab encoders. This module hosts the standalone
data-processing logic that does NOT depend on seed-level state.
"""
from __future__ import annotations

import numpy as np

from .config import N_ACTION, N_POINT


def compute_oppair_contexts(combined_df, k_per_rally):
    """V25-A opponent-pair statistics with strict LOO (Leave-One-Out).

    For each rally R in match M:
      k_pred = k_per_rally[R]                 # position to predict at
      ego = gamePlayerId at stroke k_pred
      opp = gamePlayerOtherId at stroke k_pred
      ego_stats = mean over (strokes in M with gamePlayerId=ego,
                             EXCLUDING strokes in rally R) of [pt_freq, ac_freq]
      opp_stats = same for opp player
      ctx       = [ego_pt(10) | ego_act(19) | opp_pt(10) | opp_act(19)] = 58-dim

    Args:
        combined_df: DataFrame with rally_uid, match, pointId, actionId,
                     gamePlayerId, gamePlayerOtherId, strikeNumber.
        k_per_rally: dict {rally_uid: int} mapping rally to the k position
                     used during this seed's training/inference.

    Returns:
        dict {rally_uid: np.ndarray(58,)} — per-rally context vector.
    """
    contexts = {}
    for match_id, gm in combined_df.groupby("match", sort=False):
        # Pre-aggregate per (rally, player) for O(1) LOO subtraction.
        rally_player_stats = {}
        for (rid, gpid), gpg in gm.groupby(["rally_uid", "gamePlayerId"], sort=False):
            pt = np.bincount(gpg["pointId"].clip(0, N_POINT - 1).values, minlength=N_POINT).astype(float)
            ac = np.bincount(gpg["actionId"].clip(0, N_ACTION - 1).values, minlength=N_ACTION).astype(float)
            rally_player_stats[(int(rid), int(gpid))] = (pt, ac, len(gpg))

        # Per-player match totals.
        player_totals = {}
        for (rid, gpid), (pt, ac, n) in rally_player_stats.items():
            if gpid not in player_totals:
                player_totals[gpid] = (np.zeros(N_POINT), np.zeros(N_ACTION), 0)
            pt0, ac0, n0 = player_totals[gpid]
            player_totals[gpid] = (pt0 + pt, ac0 + ac, n0 + n)

        # For each rally R, identify ego/opp at k_pred and compute LOO stats.
        for rid, gr in gm.groupby("rally_uid", sort=False):
            rid_int = int(rid)
            k = k_per_rally[rid_int]
            T = len(gr)
            k_eff = min(k, T - 1) if T > 0 else 0
            row = gr.iloc[k_eff]
            ego = int(row["gamePlayerId"])
            opp = int(row["gamePlayerOtherId"])

            # Ego LOO stats (subtract the rally R's contribution).
            ego_pt_tot, ego_ac_tot, n_ego_tot = player_totals.get(
                ego, (np.zeros(N_POINT), np.zeros(N_ACTION), 0))
            ego_pt_R, ego_ac_R, n_ego_R = rally_player_stats.get(
                (rid_int, ego), (np.zeros(N_POINT), np.zeros(N_ACTION), 0))
            ego_pt = ego_pt_tot - ego_pt_R
            ego_ac = ego_ac_tot - ego_ac_R
            n_ego = n_ego_tot - n_ego_R
            if n_ego > 0:
                ego_pt = ego_pt / n_ego
                ego_ac = ego_ac / n_ego

            # Opp LOO stats.
            opp_pt_tot, opp_ac_tot, n_opp_tot = player_totals.get(
                opp, (np.zeros(N_POINT), np.zeros(N_ACTION), 0))
            opp_pt_R, opp_ac_R, n_opp_R = rally_player_stats.get(
                (rid_int, opp), (np.zeros(N_POINT), np.zeros(N_ACTION), 0))
            opp_pt = opp_pt_tot - opp_pt_R
            opp_ac = opp_ac_tot - opp_ac_R
            n_opp = n_opp_tot - n_opp_R
            if n_opp > 0:
                opp_pt = opp_pt / n_opp
                opp_ac = opp_ac / n_opp

            ctx = np.concatenate([ego_pt, ego_ac, opp_pt, opp_ac]).astype(np.float32)
            contexts[rid_int] = ctx
    return contexts
