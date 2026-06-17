"""Stage 4: V25-A and V27 bag training (10 seeds × 5 folds × 30 epochs each).

V25-A and V27 share architecture (TTSSLLSTMHier + 58-dim opp-pair context); they
differ only in the point head's loss function:
  - V25-A: FocalLoss          (γ=2, uniform label-smoothing 0.10)
  - V27:   AsymSpatialFocalLoss (V27 innovation — class-3 spatial smoothing)

Closures kept inside stage4_bag_one_seed (sample_k, encode_df, build_rallies,
DS, collate, class_w_sqrt) for bit-identical reproducibility — they depend on
per-seed vocab encoders and per-seed k_per_rally state.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .config import (
    TRAIN_CSV, TEST_CSV, SSL_CKPT, CACHE_DIR, device, log,
    FEATURES, MAX_LEN_TT, N_FOLDS, N_ACTION, N_POINT, CTX_DIM,
    OOV_TOKEN, PLAYER_MASK_P,
    FT_LR, FT_WEIGHT_DECAY, FT_BS, FT_EPOCHS, GRAD_CLIP,
)
from .data_processing import compute_oppair_contexts
from .models import TTSSLLSTMHier
from .losses import FocalLoss, AsymSpatialFocalLoss


def stage4_bag_one_seed(seed: int, variant: str = "v25a"):
    """Train one seed of the V25-A or V27 bag (5-fold GroupKFold by match).

    Args:
        seed: random seed for this bag member (recommended 42-51 for 10-seed bag).
        variant: 'v25a' (FocalLoss on point head) or 'v27' (AsymSpatialFocalLoss).

    Output: cache/oof_test_{variant}{_seedN}.npz with OOF + per-fold test probs.
    Skips if cache already exists.
    """
    assert variant in ("v25a", "v27"), f"unknown variant: {variant}"
    suffix = "" if seed == 42 else f"_seed{seed}"
    out_path = CACHE_DIR / f"oof_test_{variant}{suffix}.npz"
    if out_path.exists():
        log(f"  seed {seed} ({variant}) cached: {out_path.name}")
        return

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    train_df["strikeNumber"] = train_df["strikeNumber"].clip(0, MAX_LEN_TT)
    test_df["strikeNumber"] = test_df["strikeNumber"].clip(0, MAX_LEN_TT)

    # --- K-truncation sampling (test-distribution-aware) ---
    # Must be computed BEFORE opp-pair context (which needs k to identify ego/opp).
    test_max_strike = test_df.groupby("rally_uid")["strikeNumber"].max()
    _max_k = int(max(test_max_strike.max(), train_df.groupby("rally_uid").size().max()))
    test_k_dist = np.zeros(_max_k + 2)
    for k, c in test_max_strike.value_counts().items():
        test_k_dist[int(k)] = c
    test_k_dist /= test_k_dist.sum()

    def sample_k(T, rng):
        if T < 2:
            return 1
        T_eff = min(int(T), len(test_k_dist))
        valid = test_k_dist[1:T_eff]
        s = valid.sum()
        return int(rng.choice(np.arange(1, T_eff), p=valid / s)) if s > 0 else T - 1

    rng_main = np.random.RandomState(42)
    train_T = train_df.groupby("rally_uid").size()
    test_T = test_df.groupby("rally_uid").size()
    k_per_rally_pre = {}
    for rid in train_T.index:
        k_per_rally_pre[int(rid)] = sample_k(int(train_T[rid]), rng_main)
    for rid in test_T.index:
        k_per_rally_pre[int(rid)] = sample_k(int(test_T[rid]), rng_main)

    # --- Opponent-pair contexts on COMBINED train+test ---
    combined = pd.concat([
        train_df[["rally_uid", "match", "pointId", "actionId",
                  "gamePlayerId", "gamePlayerOtherId", "strikeNumber"]],
        test_df[["rally_uid", "match", "pointId", "actionId",
                 "gamePlayerId", "gamePlayerOtherId", "strikeNumber"]],
    ], ignore_index=True)
    match_contexts = compute_oppair_contexts(combined, k_per_rally_pre)

    # --- Build vocab from train+test combined (cold-start: OOV → token 1) ---
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    encoders, vocab_sizes = {}, {}
    for col in FEATURES:
        cats = sorted(all_df[col].dropna().unique().tolist())
        encoders[col] = {c: i + 2 for i, c in enumerate(cats)}  # 0=PAD, 1=OOV reserved
        vocab_sizes[col] = len(cats) + 2

    def encode_df(df):
        return np.stack(
            [df[c].map(encoders[c]).fillna(OOV_TOKEN).astype(np.int64).values for c in FEATURES],
            axis=1)

    def build_rallies(df, source="train"):
        out = []
        for rid, g in df.groupby("rally_uid", sort=False):
            g = g.sort_values("strikeNumber")
            rid_int = int(rid)
            out.append({
                "rally_uid": rid_int, "match": int(g["match"].iloc[0]),
                "T": len(g), "X": encode_df(g),
                "actions": g["actionId"].values.astype(np.int64),
                "points": g["pointId"].values.astype(np.int64),
                "winner": int(g["serverGetPoint"].iloc[0]) if "serverGetPoint" in g.columns else 0,
                "ctx": match_contexts[rid_int],
                "source": source,
            })
        return out

    all_train = build_rallies(train_df, "train")
    all_test = build_rallies(test_df, "test")
    train_matches = np.array([r["match"] for r in all_train])
    fold_splits = list(GroupKFold(n_splits=N_FOLDS).split(np.arange(len(all_train)),
                                                           groups=train_matches))

    k_per_rally = k_per_rally_pre  # alias
    test_keys = {int(rid): int(g["strikeNumber"].max()) for rid, g in test_df.groupby("rally_uid", sort=False)}

    class DS(Dataset):
        def __init__(self, rallies, mode="train"):
            self.r = rallies
            self.mode = mode

        def __len__(self):
            return len(self.r)

        def __getitem__(self, i):
            r = self.r[i]
            T = r["T"]
            if self.mode == "train":
                k = sample_k(T, np.random)
                ya, yp = r["actions"][k], r["points"][k]
            else:
                k = max(1, min(r["k"], T))
                ya, yp = (r["actions"][k], r["points"][k]) if k < T else (0, 0)
            X = r["X"][:k].copy()
            if self.mode == "train":
                # Player masking — randomly replace gamePlayerId / gamePlayerOtherId with OOV.
                X[np.random.random(k) < PLAYER_MASK_P, 11] = OOV_TOKEN
                X[np.random.random(k) < PLAYER_MASK_P, 12] = OOV_TOKEN
            return X, ya, yp, float(r["winner"]), r["ctx"]

    def collate(batch):
        Xs, ya, yp, yw, ctxs = zip(*batch)
        lens = torch.tensor([len(X) for X in Xs])
        F_ = Xs[0].shape[1]
        X_pad = torch.zeros(len(Xs), int(lens.max()), F_, dtype=torch.long)
        for i, X in enumerate(Xs):
            X_pad[i, :len(X)] = torch.from_numpy(X)
        ctx_t = torch.from_numpy(np.stack(ctxs))
        return X_pad, lens, torch.tensor(ya), torch.tensor(yp), torch.tensor(yw, dtype=torch.float32), ctx_t

    def class_w_sqrt(rallies, n, key):
        """sqrt-balanced class weights (less aggressive than inverse-frequency)."""
        cnt = np.ones(n)
        for r in rallies:
            for v in r[key]:
                if 0 <= v < n:
                    cnt[v] += 1
        w = 1.0 / np.sqrt(cnt)
        return torch.tensor(w * (n / w.sum()), dtype=torch.float32, device=device)

    # --- Load SSL pretrained encoder ---
    ssl = torch.load(SSL_CKPT, map_location=device, weights_only=False)

    n_train = len(all_train)
    rid_to_idx = {r["rally_uid"]: i for i, r in enumerate(all_train)}
    oof_a = np.zeros((n_train, N_ACTION))
    oof_p = np.zeros((n_train, N_POINT))
    oof_w = np.zeros(n_train)
    test_a = np.zeros((N_FOLDS, len(all_test), N_ACTION))
    test_p = np.zeros((N_FOLDS, len(all_test), N_POINT))
    test_w = np.zeros((N_FOLDS, len(all_test)))

    # --- 5-fold GroupKFold by match ---
    for fold, (tr_idx, va_idx) in enumerate(fold_splits):
        torch.manual_seed(seed + fold)
        np.random.seed(seed + fold)
        tr_rids = set(all_train[i]["rally_uid"] for i in tr_idx)
        va_rids = set(all_train[i]["rally_uid"] for i in va_idx)
        # Transductive aug: test rallies (T>=2) added to training set as input sequences.
        usable_test = [r for r in all_test if r["T"] >= 2]
        tr_r = [r for r in all_train if r["rally_uid"] in tr_rids] + usable_test
        va_r = [{**r, "k": k_per_rally[r["rally_uid"]]} for r in all_train if r["rally_uid"] in va_rids]
        te_r = [{**r, "k": test_keys[r["rally_uid"]]} for r in all_test]

        tr_loader = DataLoader(DS(tr_r, "train"), batch_size=FT_BS, shuffle=True, collate_fn=collate)
        va_loader = DataLoader(DS(va_r, "val"), batch_size=128, shuffle=False, collate_fn=collate)
        te_loader = DataLoader(DS(te_r, "val"), batch_size=128, shuffle=False, collate_fn=collate)

        act_w = class_w_sqrt(tr_r, N_ACTION, "actions")
        pt_w = class_w_sqrt(tr_r, N_POINT, "points")
        model = TTSSLLSTMHier(vocab_sizes).to(device)
        n_loaded = model.load_ssl(ssl["state"])
        if fold == 0 and seed == 42:
            log(f"    [{variant}] model params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M, "
                f"SSL loaded {n_loaded} keys, CTX_DIM={CTX_DIM}")

        opt = torch.optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=FT_WEIGHT_DECAY)
        warmup = max(1, len(tr_loader))
        total = len(tr_loader) * FT_EPOCHS
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda s: float(s + 1) / warmup if s < warmup
            else 0.5 * (1 + np.cos(np.pi * (s - warmup) / max(1, total - warmup))))
        ce_a = FocalLoss(weight=act_w, gamma=2.0)
        # Point-head loss is the V25-A vs V27 distinction.
        if variant == "v27":
            ce_p = AsymSpatialFocalLoss(weight=pt_w, gamma=2.0).to(device)
        else:  # v25a
            ce_p = FocalLoss(weight=pt_w, gamma=2.0)
        bce = nn.BCEWithLogitsLoss()

        best_state = None
        best_final = -1.0
        t0 = time.time()
        for ep in range(1, FT_EPOCHS + 1):
            model.train()
            for X, lens, ya, yp, yw, ctx in tr_loader:
                X, lens, ya, yp, yw, ctx = (X.to(device), lens.to(device), ya.to(device),
                                             yp.to(device), yw.to(device), ctx.to(device))
                la, lp, lw = model(X, lens, ctx)
                loss = 0.4 * ce_a(la, ya) + 0.4 * ce_p(lp, yp) + 0.2 * bce(lw, yw)
                if not torch.isfinite(loss):
                    opt.zero_grad(); continue
                opt.zero_grad()
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                if not torch.isfinite(gn):
                    opt.zero_grad(); continue
                opt.step()
                sch.step()
            # --- Validation: pick best epoch by Final score ---
            model.eval()
            va_a, va_p, va_at, va_pt, va_w_t, va_w_p = [], [], [], [], [], []
            with torch.no_grad():
                for X, lens, ya, yp, yw, ctx in va_loader:
                    X, lens, ctx = X.to(device), lens.to(device), ctx.to(device)
                    la, lp, lw = model(X, lens, ctx)
                    va_at += ya.tolist(); va_a += la.argmax(-1).cpu().tolist()
                    va_pt += yp.tolist(); va_p += lp.argmax(-1).cpu().tolist()
                    va_w_t += yw.tolist(); va_w_p += torch.sigmoid(lw).cpu().tolist()
            f1_a = f1_score(va_at, va_a, average="macro", zero_division=0)
            f1_p = f1_score(va_pt, va_p, average="macro", zero_division=0)
            auc = roc_auc_score(va_w_t, va_w_p) if len(set(va_w_t)) > 1 else 0.5
            final = 0.4 * f1_a + 0.4 * f1_p + 0.2 * auc
            if final > best_final:
                best_final = final
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        log(f"    fold {fold}: F1_a={f1_a:.4f} F1_p={f1_p:.4f} AUC={auc:.4f} "
            f"best_Final={best_final:.4f} ({time.time() - t0:.0f}s)")

        # --- Inference on val + test ---
        model.eval()
        pa_v, pp_v, pw_v = [], [], []
        with torch.no_grad():
            for X, lens, ya, yp, yw, ctx in va_loader:
                X, lens, ctx = X.to(device), lens.to(device), ctx.to(device)
                la, lp, lw = model(X, lens, ctx)
                pa_v.append(torch.softmax(la, dim=-1).cpu().numpy())
                pp_v.append(torch.softmax(lp, dim=-1).cpu().numpy())
                pw_v.append(torch.sigmoid(lw).cpu().numpy())
        vA = np.concatenate(pa_v); vP = np.concatenate(pp_v); vW = np.concatenate(pw_v)
        pa_t, pp_t, pw_t = [], [], []
        with torch.no_grad():
            for X, lens, ya, yp, yw, ctx in te_loader:
                X, lens, ctx = X.to(device), lens.to(device), ctx.to(device)
                la, lp, lw = model(X, lens, ctx)
                pa_t.append(torch.softmax(la, dim=-1).cpu().numpy())
                pp_t.append(torch.softmax(lp, dim=-1).cpu().numpy())
                pw_t.append(torch.sigmoid(lw).cpu().numpy())
        tA = np.concatenate(pa_t); tP = np.concatenate(pp_t); tW = np.concatenate(pw_t)
        for j, r in enumerate(va_r):
            i = rid_to_idx[r["rally_uid"]]
            oof_a[i] = vA[j]; oof_p[i] = vP[j]; oof_w[i] = vW[j]
        test_a[fold] = tA; test_p[fold] = tP; test_w[fold] = tW

    np.savez(out_path,
             oof_tt_a=oof_a, oof_tt_p=oof_p, oof_tt_w=oof_w,
             test_tt_a=test_a, test_tt_p=test_p, test_tt_w=test_w)
    log(f"  seed {seed} saved → {out_path.name}")


def stage4_bag(seeds, variant="v25a"):
    """Train all seeds for one bag variant. `variant` ∈ {'v25a', 'v27'}."""
    log(f"=== Stage 4: {variant.upper()} bag training ({len(seeds)} seeds × {N_FOLDS} folds) ===")
    for s in seeds:
        t0 = time.time()
        log(f"  --- {variant} seed {s} ---")
        stage4_bag_one_seed(s, variant=variant)
        log(f"  {variant} seed {s} done in {time.time() - t0:.0f}s")
