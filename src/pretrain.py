"""Stage 1: SSL pretrain on ShuttleSet22 (cross-sport knowledge transfer).

Trains a BiLSTM encoder on ShuttleSet22 (羽球) stroke sequences via Masked
Language Modeling (MLM), then saves the transferable `lstm.*` weights for use as
initialization in the TT model. The cross-sport SSL transfer is one of the key
innovations of this system (LB +0.0044 vs V3 baseline).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .config import (
    ROOT, SS22_TRAIN, SSL_CKPT, device, log,
    SS22_CAT_FEATS, MLM_TARGETS, MASK_PROB, MAX_LEN_PRETRAIN,
    PRETRAIN_BS, PRETRAIN_LR, PRETRAIN_EPOCHS,
    PAD_TOKEN, MASK_TOKEN,
)
from .models import SSLEncoderLSTM


def stage1_ssl_pretrain():
    """Pretrain BiLSTM encoder via MLM on ShuttleSet22 stroke sequences.

    Side effect: writes `cache/ssl_lstm_encoder_shuttleset22.pt` (the LSTM weights
    only — embeddings/projection are badminton-specific and not transferred).
    Skips if the checkpoint already exists.
    """
    log("=== Stage 1: SSL pretrain on ShuttleSet22 ===")
    if SSL_CKPT.exists():
        log(f"  already cached: {SSL_CKPT.relative_to(ROOT)}, skip pretrain")
        return
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.read_csv(SS22_TRAIN)
    df["rally_uid"] = df["match_id"].astype(str) + "_" + df["rally"].astype(str)
    log(f"  {len(df):,} strokes / {df.rally_uid.nunique():,} rallies")

    # Build per-feature encoders (token IDs offset by 2 to reserve PAD=0, OOV=1, MASK=2).
    encoders, vocab_sizes = {}, {}
    for col in SS22_CAT_FEATS:
        cats = sorted(df[col].dropna().unique().tolist())
        encoders[col] = {c: i + 2 for i, c in enumerate(cats)}
        vocab_sizes[col] = len(cats) + 2

    def encode_df_ss22(df_):
        return np.stack([df_[c].map(encoders[c]).fillna(PAD_TOKEN).astype(np.int64).values
                         for c in SS22_CAT_FEATS], axis=1)

    # Build per-rally token sequences, filter by length.
    rallies = []
    for rid, g in df.groupby("rally_uid", sort=False):
        g = g.sort_values("ball_round")
        n = len(g)
        if n < 3 or n > MAX_LEN_PRETRAIN:
            continue
        rallies.append({"T": n, "X": encode_df_ss22(g)})
    log(f"  {len(rallies):,} valid rallies (3 ≤ T ≤ {MAX_LEN_PRETRAIN})")

    class MLMDataset(Dataset):
        """Apply MLM masking per __getitem__ (15% mask rate)."""
        def __init__(self, rallies):
            self.r = rallies

        def __len__(self):
            return len(self.r)

        def __getitem__(self, i):
            r = self.r[i]
            X = r["X"].copy()
            T = X.shape[0]
            mask = (np.random.random(T) < MASK_PROB)
            tgts = {}
            for tgt in MLM_TARGETS:
                idx = SS22_CAT_FEATS.index(tgt)
                t = X[:, idx].copy()
                t[~mask] = -100
                tgts[tgt] = t
            for tgt in MLM_TARGETS:
                idx = SS22_CAT_FEATS.index(tgt)
                X[mask, idx] = MASK_TOKEN
            return X, T, tgts["type"], tgts["landing_area"]

    def collate(batch):
        Xs, Ts, tts, tas = zip(*batch)
        lens = torch.tensor(Ts)
        Tmax = int(lens.max())
        F_ = Xs[0].shape[1]
        B = len(Xs)
        X_pad = torch.zeros(B, Tmax, F_, dtype=torch.long)
        tt_pad = torch.full((B, Tmax), -100, dtype=torch.long)
        ta_pad = torch.full((B, Tmax), -100, dtype=torch.long)
        for i, X in enumerate(Xs):
            L = len(X)
            X_pad[i, :L] = torch.from_numpy(X)
            tt_pad[i, :L] = torch.from_numpy(tts[i])
            ta_pad[i, :L] = torch.from_numpy(tas[i])
        return X_pad, lens, tt_pad, ta_pad

    loader = DataLoader(MLMDataset(rallies), batch_size=PRETRAIN_BS, shuffle=True, collate_fn=collate)
    model = SSLEncoderLSTM(vocab_sizes).to(device)
    log(f"  model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=1e-5)
    total_steps = len(loader) * PRETRAIN_EPOCHS
    warmup_steps = max(1, len(loader))
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: float(s + 1) / warmup_steps if s < warmup_steps
        else 0.5 * (1 + np.cos(np.pi * (s - warmup_steps) / max(1, total_steps - warmup_steps))),
    )

    t0 = time.time()
    for ep in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        n_ct = n_ca = n_mt = n_ma = 0
        for X, lens, tt, ta in loader:
            X, lens, tt, ta = X.to(device), lens.to(device), tt.to(device), ta.to(device)
            outs, valid = model(X, lens)
            tt_eff = tt.clone(); tt_eff[~valid] = -100
            ta_eff = ta.clone(); ta_eff[~valid] = -100
            l_t = F.cross_entropy(
                outs["type"].reshape(-1, outs["type"].size(-1)),
                tt_eff.reshape(-1), ignore_index=-100)
            l_a = F.cross_entropy(
                outs["landing_area"].reshape(-1, outs["landing_area"].size(-1)),
                ta_eff.reshape(-1), ignore_index=-100)
            loss = l_t + l_a
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            with torch.no_grad():
                m_t = (tt_eff != -100); m_a = (ta_eff != -100)
                n_mt += int(m_t.sum()); n_ma += int(m_a.sum())
                n_ct += int((outs["type"].argmax(-1)[m_t] == tt_eff[m_t]).sum())
                n_ca += int((outs["landing_area"].argmax(-1)[m_a] == ta_eff[m_a]).sum())
        if ep % 5 == 0 or ep == PRETRAIN_EPOCHS or ep == 1:
            log(f"  ep {ep:2d}: type_acc={n_ct / max(1, n_mt):.4f} area_acc={n_ca / max(1, n_ma):.4f}")

    log(f"  pretrain done in {time.time() - t0:.0f}s")
    # Save only the `lstm.*` weights — these are the cross-sport transferable parts.
    transfer_state = {k: v for k, v in model.state_dict().items() if k.startswith("lstm.")}
    SSL_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": transfer_state}, SSL_CKPT)
    log(f"  saved {len(transfer_state)} tensors to {SSL_CKPT.relative_to(ROOT)}")
