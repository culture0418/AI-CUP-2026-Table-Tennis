"""Stage 5: V27 Mode A ensemble — α-search + threshold tuning + base submission.

V27 Mode A ensemble structure:
  - Action  7-way: V3-LSTM + V3-XGB + V3-Cat + v1 + asym + V25-A + V27
  - Point   8-way: above + V3-FTT
  - Winner  4-way: V3-LSTM + V3-XGB + V3-Cat + v1

α-search algorithm: coarse grid (step=0.1) → fine coordinate descent (step=0.05).
After α is fixed, per-class threshold multipliers are tuned with cap=0.75 to
prevent overfitting.

Output: `submissions/submission_v27_modeA_canonical_{ts}.csv` (LB 0.3787, base for
the OLD-lookup injection in validation.py).
"""
from __future__ import annotations

import time
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from .config import (
    ROOT, TEST_CSV, V3_CACHE, CACHE_DIR, SUB_DIR, log,
)


# ===================================================================
# Macro-F1 helper used throughout α-search and threshold tuning.
# ===================================================================
def f1m(y, p):
    return f1_score(y, p, average="macro", zero_division=0)


# ===================================================================
# Bag loaders (mean over the 10 seeds, per-fold test probabilities averaged)
# ===================================================================
def load_bag(tag, seeds):
    """Load v1 (`_ssl_lstm`) or asym (`_ssl_lstm_asym`) bag from
    `oof_test_tt_shuttlenet{seed}{tag}.npz`."""
    a, p, w, ta, tp, tw = [], [], [], [], [], []
    for s in seeds:
        suf = "" if s == 42 else f"_seed{s}"
        d = np.load(CACHE_DIR / f"oof_test_tt_shuttlenet{suf}{tag}.npz")
        a.append(d["oof_tt_a"]); p.append(d["oof_tt_p"]); w.append(d["oof_tt_w"])
        ta.append(d["test_tt_a"].mean(0))
        tp.append(d["test_tt_p"].mean(0))
        tw.append(d["test_tt_w"].mean(0))
    return (np.mean(a, 0), np.mean(p, 0), np.mean(w, 0),
            np.mean(ta, 0), np.mean(tp, 0), np.mean(tw, 0))


def load_v25a_bag(seeds):
    """Load V25-A bag from `oof_test_v25a{_seedN}.npz`."""
    a, p, w, ta, tp, tw = [], [], [], [], [], []
    for s in seeds:
        suf = "" if s == 42 else f"_seed{s}"
        d = np.load(CACHE_DIR / f"oof_test_v25a{suf}.npz")
        a.append(d["oof_tt_a"]); p.append(d["oof_tt_p"]); w.append(d["oof_tt_w"])
        ta.append(d["test_tt_a"].mean(0))
        tp.append(d["test_tt_p"].mean(0))
        tw.append(d["test_tt_w"].mean(0))
    return (np.mean(a, 0), np.mean(p, 0), np.mean(w, 0),
            np.mean(ta, 0), np.mean(tp, 0), np.mean(tw, 0))


def load_v27_bag(seeds):
    """Load V27 bag from `oof_test_v27{_seedN}.npz`."""
    a, p, w, ta, tp, tw = [], [], [], [], [], []
    for s in seeds:
        suf = "" if s == 42 else f"_seed{s}"
        d = np.load(CACHE_DIR / f"oof_test_v27{suf}.npz")
        a.append(d["oof_tt_a"]); p.append(d["oof_tt_p"]); w.append(d["oof_tt_w"])
        ta.append(d["test_tt_a"].mean(0))
        tp.append(d["test_tt_p"].mean(0))
        tw.append(d["test_tt_w"].mean(0))
    return (np.mean(a, 0), np.mean(p, 0), np.mean(w, 0),
            np.mean(ta, 0), np.mean(tp, 0), np.mean(tw, 0))


# ===================================================================
# α-search algorithms
# ===================================================================
def search_grid(probs, y, n, step, metric="f1"):
    """Exhaustive grid search over the (n-1)-simplex (last α = 1 - sum(others))."""
    best, ba = -1, None
    grid = np.arange(0, 1.0001, step)
    for combo in product(grid, repeat=n - 1):
        last = 1 - sum(combo)
        if last < -1e-9 or last > 1 + 1e-9:
            continue
        last = max(0, min(1, last))
        alphas = list(combo) + [last]
        ens = sum(a * p for a, p in zip(alphas, probs))
        s = f1m(y, ens.argmax(-1)) if metric == "f1" else roc_auc_score(y, ens)
        if s > best:
            best, ba = s, tuple(alphas)
    return ba, best


def coord_descent(probs, y, init_alphas, step=0.05, max_iter=10, metric="f1"):
    """Coordinate descent on the simplex with renormalization at each step."""
    alphas = list(init_alphas)
    n = len(alphas)
    grid = np.arange(0, 1.0001, step)
    ens = sum(a * p for a, p in zip(alphas, probs))
    best = f1m(y, ens.argmax(-1)) if metric == "f1" else roc_auc_score(y, ens)
    for _ in range(max_iter):
        improved = False
        for i in range(n):
            best_a_i = alphas[i]
            best_s = best
            best_alphas = alphas[:]
            for new_a_i in grid:
                rest = [alphas[j] for j in range(n) if j != i]
                rest_sum = sum(rest)
                if rest_sum < 1e-9:
                    if abs(new_a_i - 1) > 1e-9:
                        continue
                    cand = [0.0] * n
                    cand[i] = 1.0
                else:
                    scale = (1 - new_a_i) / rest_sum
                    cand = [a * scale for a in alphas]
                    cand[i] = new_a_i
                ens = sum(a * p for a, p in zip(cand, probs))
                s = f1m(y, ens.argmax(-1)) if metric == "f1" else roc_auc_score(y, ens)
                if s > best_s + 1e-6:
                    best_s, best_a_i = s, new_a_i
                    best_alphas = cand[:]
            if best_a_i != alphas[i]:
                alphas = best_alphas
                best = best_s
                improved = True
        if not improved:
            break
    return tuple(alphas), best


def tune_thresh(probs, y, n_iter=4):
    """Per-class threshold multiplier tuning with cap=0.75 to limit overfit."""
    grid = np.concatenate([np.arange(0.5, 3.0, 0.05), np.arange(3.0, 5.0, 0.25)])
    n = probs.shape[1]
    m = np.ones(n)
    base = f1m(y, probs.argmax(-1))
    for _ in range(n_iter):
        improved = False
        for c in range(n):
            best_m, best_s = m[c], base
            for k in grid:
                t = m.copy(); t[c] = k
                s = f1m(y, (probs * t[None, :]).argmax(-1))
                if s > best_s + 1e-6:
                    best_s, best_m = s, k
            if best_m != m[c]:
                m[c] = best_m
                base = best_s
                improved = True
        if not improved:
            break
    # Cap to 0.75 of the range to prevent over-amplification of rare classes.
    cap_ratio = 0.75
    m_min, m_max = m.min(), m.max()
    if m_max - m_min > 1e-9:
        m_clipped = m_min + (m - m_min) * cap_ratio + (1 - cap_ratio) * (m_max + m_min) / 2
        s_clipped = f1m(y, (probs * m_clipped[None, :]).argmax(-1))
        if s_clipped >= base - 1e-4:
            m = m_clipped
            base = s_clipped
    return m, base


def stage5_ensemble_and_submit(seeds):
    """Build V27 Mode A ensemble from cached bags + write the LB 0.3787 base submission.

    This is the "pre-OLD-lookup" base. The OLD lookup is applied separately by
    `validation.stage5b_oldleak_inject()`.

    Returns:
        Path to the generated `submission_v27_modeA_canonical_{ts}.csv`.
    """
    log("=== Stage 5: V27 Mode A ensemble (aug → V25-A, asym_aug → V27) + base submission ===")
    # Verify required caches.
    required = [V3_CACHE]
    for tag in ["_ssl_lstm", "_ssl_lstm_asym"]:
        for s in seeds:
            suf = "" if s == 42 else f"_seed{s}"
            required.append(CACHE_DIR / f"oof_test_tt_shuttlenet{suf}{tag}.npz")
    for s in seeds:
        suf = "" if s == 42 else f"_seed{s}"
        required.append(CACHE_DIR / f"oof_test_v25a{suf}.npz")
        required.append(CACHE_DIR / f"oof_test_v27{suf}.npz")
    missing = [p for p in required if not p.exists()]
    if missing:
        log(f"  ERROR: {len(missing)} required caches missing:")
        for p in missing[:10]:
            log(f"    {p.name}")
        if len(missing) > 10:
            log(f"    ... and {len(missing) - 10} more")
        log("  Run training first (Stage 4 V25-A + V27 bags).")
        raise FileNotFoundError(f"{len(missing)} caches missing")

    # Load V3 baseline cache + bags.
    v3 = np.load(V3_CACHE)
    oy_a, oy_p, oy_w = v3["oof_y_a"], v3["oof_y_p"], v3["oof_y_w"]
    log(f"  V3 baseline loaded: {len(oy_a)} OOF samples")
    v1_a, v1_p, v1_w, tv1_a, tv1_p, tv1_w = load_bag("_ssl_lstm", seeds); log("  v1 bag loaded")
    ay_a, ay_p, ay_w, tay_a, tay_p, tay_w = load_bag("_ssl_lstm_asym", seeds); log("  asym bag loaded")
    v25_a, v25_p, v25_w, tv25_a, tv25_p, tv25_w = load_v25a_bag(seeds); log("  V25-A bag loaded")
    v27_a, v27_p, v27_w, tv27_a, tv27_p, tv27_w = load_v27_bag(seeds); log("  V27 bag loaded")

    log(f"  V25-A standalone: F1_a={f1m(oy_a, v25_a.argmax(-1)):.4f} "
        f"F1_p={f1m(oy_p, v25_p.argmax(-1)):.4f} AUC={roc_auc_score(oy_w, v25_w):.4f}")
    log(f"  V27   standalone: F1_a={f1m(oy_a, v27_a.argmax(-1)):.4f} "
        f"F1_p={f1m(oy_p, v27_p.argmax(-1)):.4f} AUC={roc_auc_score(oy_w, v27_w):.4f}")

    # --- Action 7-way ---
    log("\n  ## Action 7-way α-search (V25-A + V27 both substitute) ##")
    a7 = [v3["oof_lstm_a"], v3["oof_xgb_a"], v3["oof_cat_a"], v1_a, ay_a, v25_a, v27_a]
    ba_a, _ = search_grid(a7, oy_a, 7, 0.1)
    ba_a, fa = coord_descent(a7, oy_a, ba_a, step=0.05)
    log(f"    α={tuple(round(x, 2) for x in ba_a)} F1_a={fa:.4f}")

    # --- Point 8-way (two inits, pick max) ---
    log("\n  ## Point 8-way α-search ##")
    p8 = [v3["oof_lstm_p"], v3["oof_xgb_p"], v3["oof_cat_p"], v3["oof_ftt_p"],
          v1_p, ay_p, v25_p, v27_p]
    candidates = []
    v16_init = (0.0, 0.0, 0.1, 0.29, 0.57, 0.05, 0.0, 0.0)
    a1, s1 = coord_descent(p8, oy_p, v16_init, step=0.05); candidates.append((s1, a1))
    ba2, _ = search_grid(p8, oy_p, 8, 0.2)
    a2, s2 = coord_descent(p8, oy_p, ba2, step=0.05); candidates.append((s2, a2))
    best_s, ba_p = max(candidates, key=lambda x: x[0])
    fp = best_s
    log(f"    α={tuple(round(x, 2) for x in ba_p)} F1_p={fp:.4f}")

    # --- Winner 4-way ---
    log("\n  ## Winner 4-way α-search ##")
    w4 = [v3["oof_lstm_w"], v3["oof_xgb_w"], v3["oof_cat_w"], v1_w]
    ba_w, auc = search_grid(w4, oy_w, 4, 0.05, metric="auc")
    log(f"    α={tuple(round(x, 2) for x in ba_w)} AUC={auc:.4f}")

    # --- Threshold tune + OOF Final ---
    log("\n  ## Per-class threshold mults (cap=0.75) ##")
    ens_a = sum(a * p for a, p in zip(ba_a, a7))
    ens_p = sum(a * p for a, p in zip(ba_p, p8))
    ens_w = sum(a * p for a, p in zip(ba_w, w4))
    m_a, fa_t = tune_thresh(ens_a, oy_a)
    m_p, fp_t = tune_thresh(ens_p, oy_p)
    auc_t = roc_auc_score(oy_w, ens_w)
    final = 0.4 * fa_t + 0.4 * fp_t + 0.2 * auc_t
    log("\n  ## OOF Final ##")
    log(f"    F1_a={fa_t:.4f}  F1_p={fp_t:.4f}  AUC={auc_t:.4f}  Final={final:.4f}")
    log(f"    vs v17 OOF 0.3773: {final - 0.3773:+.4f}")
    log(f"    Expected LB at 0.68x v27 Mode A transfer: {0.3747450 + (final - 0.3773) * 0.68:.4f}")
    log("    Actual LB (2026-05-20): 0.3787701")

    # --- Generate base submission (pre-OLD-lookup) ---
    log("\n  ## Generate base submission ##")
    a_test = [v3["test_lstm_a"].mean(0), v3["test_xgb_a"].mean(0), v3["test_cat_a"].mean(0),
              tv1_a, tay_a, tv25_a, tv27_a]
    p_test = [v3["test_lstm_p"].mean(0), v3["test_xgb_p"].mean(0), v3["test_cat_p"].mean(0),
              v3["test_ftt_p"].mean(0), tv1_p, tay_p, tv25_p, tv27_p]
    w_test = [v3["test_lstm_w"].mean(0), v3["test_xgb_w"].mean(0), v3["test_cat_w"].mean(0),
              tv1_w]
    te_a = sum(a * p for a, p in zip(ba_a, a_test))
    te_p = sum(a * p for a, p in zip(ba_p, p_test))
    te_w = sum(a * p for a, p in zip(ba_w, w_test))
    te_a_pred = (te_a * m_a[None, :]).argmax(-1).astype(int)
    te_p_pred = (te_p * m_p[None, :]).argmax(-1).astype(int)
    test_df = pd.read_csv(TEST_CSV)
    test_rids = [int(rid) for rid, _ in test_df.groupby("rally_uid", sort=False)]
    sub = pd.DataFrame({
        "rally_uid": test_rids,
        "actionId": te_a_pred,
        "pointId": te_p_pred,
        "serverGetPoint": te_w,
    }).sort_values("rally_uid").reset_index(drop=True)
    ts = time.strftime("%Y%m%d_%H%M")
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    out = SUB_DIR / f"submission_v27_modeA_canonical_{ts}.csv"
    sub.to_csv(out, index=False)
    log(f"    saved → {out.relative_to(ROOT)} ({len(sub)} rows)")
    return out
