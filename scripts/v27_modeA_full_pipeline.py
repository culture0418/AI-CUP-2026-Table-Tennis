"""v27 Mode A Canonical Reproducer (Mode B) — LB 0.3787701 (2026-05-20).

╔════════════════════════════════════════════════════════════════════════════╗
║ v27 Mode A architecture (V25-A + V27 ensemble substitute):                 ║
║   v17 (V3 + v1 + asym + aug + asym_aug LSTM bags) with TWO substitutions:  ║
║   - aug bag        → V25-A bag (opponent-pair hierarchical, FocalLoss)     ║
║   - asym_aug bag   → V27 bag   (V25-A architecture + AsymSpatialFocalLoss) ║
║                                                                            ║
║ Both V25-A and V27 share the same architecture (BiLSTM + opp-pair 58-dim   ║
║ context). Only the POINT head loss differs:                                ║
║   - V25-A: standard FocalLoss (γ=2, uniform label smoothing 0.1)          ║
║   - V27:   AsymSpatialFocalLoss (class 3 spatial smoothing to {2,6})      ║
║                                                                            ║
║ Mode A ensemble: aug→V25-A, asym_aug→V27 (substitute BOTH slots).          ║
║                                                                            ║
║ Per rally R in match M, with k_pred = k_per_rally[R]:                      ║
║   ego = gamePlayerId at stroke k_pred                                      ║
║   opp = gamePlayerOtherId at stroke k_pred                                 ║
║   ego_stats = LOO mean of (pointId, actionId) freq over ego's strokes in M ║
║   opp_stats = LOO mean for opp player                                      ║
║   ctx_R = [ego_pt(10) | ego_act(19) | opp_pt(10) | opp_act(19)] = 58-dim   ║
║                                                                            ║
║ Key signals (vs v17 baseline):                                             ║
║   - v17 baseline (reproduce):   OOF 0.3773, LB 0.3747450, internal 0.3592 ║
║   - v25a substitute aug→v25a:   OOF 0.3825 (+0.0052), LB 0.3757252 (+1e-3)║
║   - v27 Mode A (V25+V27):       OOF 0.3869 (+0.0096), LB 0.3787701 (+4e-3)║
║   - Internal LB transfer:       0.65x (V25+V27 vs V25 only)                ║
║   - External LB transfer:       0.68x (V27 Mode A vs V25-A, 健康)          ║
║                                                                            ║
║ Reproducer scope:                                                          ║
║   This script SELF-CONTAINS:                                               ║
║     1. SSL pretrain on ShuttleSet22 raw (~10 min if not cached)           ║
║     2. V25-A bag training (10 seeds × 5 folds, ~30 min)                    ║
║     3. V27 bag training   (10 seeds × 5 folds, ~30 min)                    ║
║     4. Mode A ensemble substitute α-search                                 ║
║     5. Submission generation + MD5 verification                            ║
║                                                                            ║
║   This script REFERENCES (does NOT regenerate):                            ║
║     - V3 baseline cache (cache/oof_test_probs.npz from Ensemble.ipynb)    ║
║     - v17 LSTM bags caches (oof_test_tt_shuttlenet*_ssl_lstm{,_asym}.npz)  ║
║                                                                            ║
║   To run from absolute zero:                                               ║
║     1. Run Ensemble.ipynb to generate V3 cache                             ║
║     2. Run tt_lstm_ssl_full_pipeline.py for v1/asym bags                   ║
║     3. Run this script (v27_modeA_full_pipeline.py)                        ║
╚════════════════════════════════════════════════════════════════════════════╝

Usage:
  python scripts/v27_modeA_full_pipeline.py              # full pipeline
  python scripts/v27_modeA_full_pipeline.py --skip-ssl   # skip SSL pretrain if cached
  python scripts/v27_modeA_full_pipeline.py --seeds 42   # only 1 seed (smoke)
  python scripts/v27_modeA_full_pipeline.py --only-ensemble  # ensemble + submit only

Reference data sources:
  - data/train.csv (14,995 rallies)
  - data/test_new.csv (1,845 rallies, no serverGetPoint label)
  - data/external/shuttleset22/train.csv (30k strokes, 1,407 rallies)

Cache outputs:
  - cache/ssl_lstm_encoder_shuttleset22.pt
  - cache/oof_test_v25a{_seedN}.npz (10 seeds)
  - cache/oof_test_v27{_seedN}.npz (10 seeds)
  - submissions/submission_v27_modeA_canonical_{timestamp}.csv
"""
import argparse, time, hashlib
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score

# ============= Constants =============
ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = ROOT / 'data' / 'train.csv'
TEST_CSV = ROOT / 'data' / 'test_new.csv'
SS22_TRAIN = ROOT / 'data' / 'external' / 'shuttleset22' / 'train.csv'
V3_CACHE = ROOT / 'cache' / 'oof_test_probs.npz'
CACHE_DIR = ROOT / 'cache'
SUB_DIR = ROOT / 'submissions'
SSL_CKPT = CACHE_DIR / 'ssl_lstm_encoder_shuttleset22.pt'

# Architecture
PROJ_DIM = 128; HIDDEN = 128; EMB_DIM = 32; DROPOUT = 0.30
N_LAYERS = 1; BIDIR = True
MAX_LEN_TT = 40
N_FOLDS = 5; N_ACTION = 19; N_POINT = 10
PAD_TOKEN, OOV_TOKEN, MASK_TOKEN = 0, 1, 2
CTX_DIM = 2 * (N_POINT + N_ACTION)  # 58 (ego_pt + ego_act + opp_pt + opp_act)

# SSL pretrain hyperparams
SS22_CAT_FEATS = ['type', 'landing_area', 'player_location_area', 'opponent_location_area']
MLM_TARGETS = ['type', 'landing_area']
MLM_HEAD_KEY = {'type': 'shottype', 'landing_area': 'area'}
MASK_PROB = 0.15; MAX_LEN_PRETRAIN = 60
PRETRAIN_BS = 64; PRETRAIN_LR = 1e-3; PRETRAIN_EPOCHS = 30

# TT finetune hyperparams (matches v1_aug)
FT_LR = 1e-3; FT_WEIGHT_DECAY = 1e-5; FT_BS = 64; FT_EPOCHS = 30
PLAYER_MASK_P = 0.30; LABEL_SMOOTH = 0.10; GRAD_CLIP = 1.0
SEEDS = list(range(42, 52))

# Asym point loss config (V27 ONLY; V25-A uses standard FocalLoss for point)
FOCUS_CLASS = 3                # 反手短球 (rare, 0.9% in train)
FOCUS_NEIGHBORS = [2, 6]       # 中間短, 反手半長 (3x3 grid neighbors)
FOCUS_SPATIAL_EPS = 0.15
FOCUS_UNIFORM_EPS = 0.05

# TT input features (per-stroke)
FEATURES = ['sex','handId','strengthId','spinId','pointId','actionId','positionId','strikeId',
            'scoreSelf','scoreOther','strikeNumber','gamePlayerId','gamePlayerOtherId']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log = lambda msg: print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ============= Stage 1: SSL pretrain on ShuttleSet22 =============

class SSLEncoderLSTM(nn.Module):
    """BiLSTM encoder for ShuttleSet22 MLM pretrain. Transferable: lstm.* weights only."""
    def __init__(self, vocab_sizes):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(vocab_sizes[c], EMB_DIM, padding_idx=PAD_TOKEN) for c in SS22_CAT_FEATS
        ])
        in_dim = len(SS22_CAT_FEATS) * EMB_DIM
        self.input_proj = nn.Linear(in_dim, PROJ_DIM)
        self.input_drop = nn.Dropout(DROPOUT)
        self.lstm = nn.LSTM(PROJ_DIM, HIDDEN, num_layers=N_LAYERS, batch_first=True,
                             bidirectional=BIDIR, dropout=0.0)
        out_dim = HIDDEN * (2 if BIDIR else 1)
        self.mlm_heads = nn.ModuleDict({
            MLM_HEAD_KEY[t]: nn.Linear(out_dim, vocab_sizes[t]) for t in MLM_TARGETS
        })

    def forward(self, X, lens):
        B, T, F_ = X.shape
        emb = torch.cat([self.embs[i](X[:, :, i]) for i in range(F_)], dim=-1)
        h = self.input_drop(self.input_proj(emb))
        packed = nn.utils.rnn.pack_padded_sequence(h, lens.cpu(), batch_first=True, enforce_sorted=False)
        o, _ = self.lstm(packed)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=T)
        outs = {t: self.mlm_heads[MLM_HEAD_KEY[t]](o) for t in MLM_TARGETS}
        valid = (torch.arange(T, device=X.device).unsqueeze(0) < lens.unsqueeze(1))
        return outs, valid


def stage1_ssl_pretrain():
    log('=== Stage 1: SSL pretrain on ShuttleSet22 ===')
    if SSL_CKPT.exists():
        log(f'  already cached: {SSL_CKPT.relative_to(ROOT)}, skip pretrain')
        return
    torch.manual_seed(42); np.random.seed(42)

    df = pd.read_csv(SS22_TRAIN)
    df['rally_uid'] = df['match_id'].astype(str) + '_' + df['rally'].astype(str)
    log(f'  {len(df):,} strokes / {df.rally_uid.nunique():,} rallies')

    encoders, vocab_sizes = {}, {}
    for col in SS22_CAT_FEATS:
        cats = sorted(df[col].dropna().unique().tolist())
        encoders[col] = {c: i + 2 for i, c in enumerate(cats)}
        vocab_sizes[col] = len(cats) + 2

    def encode_df_ss22(df_):
        return np.stack([df_[c].map(encoders[c]).fillna(PAD_TOKEN).astype(np.int64).values
                         for c in SS22_CAT_FEATS], axis=1)

    rallies = []
    for rid, g in df.groupby('rally_uid', sort=False):
        g = g.sort_values('ball_round')
        n = len(g)
        if n < 3 or n > MAX_LEN_PRETRAIN: continue
        rallies.append({'T': n, 'X': encode_df_ss22(g)})
    log(f'  {len(rallies):,} valid rallies (3 ≤ T ≤ {MAX_LEN_PRETRAIN})')

    class MLMDataset(Dataset):
        def __init__(self, rallies): self.r = rallies
        def __len__(self): return len(self.r)
        def __getitem__(self, i):
            r = self.r[i]
            X = r['X'].copy(); T = X.shape[0]
            mask = (np.random.random(T) < MASK_PROB)
            tgts = {}
            for tgt in MLM_TARGETS:
                idx = SS22_CAT_FEATS.index(tgt)
                t = X[:, idx].copy(); t[~mask] = -100
                tgts[tgt] = t
            for tgt in MLM_TARGETS:
                idx = SS22_CAT_FEATS.index(tgt)
                X[mask, idx] = MASK_TOKEN
            return X, T, tgts['type'], tgts['landing_area']

    def collate(batch):
        Xs, Ts, tts, tas = zip(*batch)
        lens = torch.tensor(Ts)
        Tmax = int(lens.max()); F_ = Xs[0].shape[1]; B = len(Xs)
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
    log(f'  model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    opt = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=1e-5)
    total_steps = len(loader) * PRETRAIN_EPOCHS
    warmup_steps = max(1, len(loader))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s:
        float(s + 1)/warmup_steps if s < warmup_steps
        else 0.5 * (1 + np.cos(np.pi * (s - warmup_steps) / max(1, total_steps - warmup_steps))))

    t0 = time.time()
    for ep in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        n_ct = n_ca = n_mt = n_ma = 0
        for X, lens, tt, ta in loader:
            X, lens, tt, ta = X.to(device), lens.to(device), tt.to(device), ta.to(device)
            outs, valid = model(X, lens)
            tt_eff = tt.clone(); tt_eff[~valid] = -100
            ta_eff = ta.clone(); ta_eff[~valid] = -100
            l_t = F.cross_entropy(outs['type'].reshape(-1, outs['type'].size(-1)),
                                   tt_eff.reshape(-1), ignore_index=-100)
            l_a = F.cross_entropy(outs['landing_area'].reshape(-1, outs['landing_area'].size(-1)),
                                   ta_eff.reshape(-1), ignore_index=-100)
            loss = l_t + l_a
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            with torch.no_grad():
                m_t = (tt_eff != -100); m_a = (ta_eff != -100)
                n_mt += int(m_t.sum()); n_ma += int(m_a.sum())
                n_ct += int((outs['type'].argmax(-1)[m_t] == tt_eff[m_t]).sum())
                n_ca += int((outs['landing_area'].argmax(-1)[m_a] == ta_eff[m_a]).sum())
        if ep % 5 == 0 or ep == PRETRAIN_EPOCHS or ep == 1:
            log(f'  ep {ep:2d}: type_acc={n_ct/max(1,n_mt):.4f} area_acc={n_ca/max(1,n_ma):.4f}')

    log(f'  pretrain done in {time.time()-t0:.0f}s')
    transfer_state = {k: v for k, v in model.state_dict().items() if k.startswith('lstm.')}
    SSL_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state': transfer_state}, SSL_CKPT)
    log(f'  saved {len(transfer_state)} tensors to {SSL_CKPT.relative_to(ROOT)}')


# ============= Stage 2: Opponent-pair match context computation =============

def compute_oppair_contexts(combined_df, k_per_rally):
    """V25-A opponent-pair stats with LOO (r' ≠ r enforced).

    For each rally R in match M:
      k_pred = k_per_rally[R]
      ego = gamePlayerId at stroke k_pred
      opp = gamePlayerOtherId at stroke k_pred
      ego_stats = mean over (strokes in M with gamePlayerId=ego,
                             EXCLUDING strokes in rally R) of [pt_freq, ac_freq]
      opp_stats = same for opp
      ctx = [ego_pt(10), ego_act(19), opp_pt(10), opp_act(19)] = 58-dim

    Args:
        combined_df: DataFrame with rally_uid, match, pointId, actionId,
                     gamePlayerId, gamePlayerOtherId, strikeNumber.
        k_per_rally: dict rally_uid → k position.
    Returns:
        dict {rally_uid: np.array (CTX_DIM,)}
    """
    contexts = {}
    for match_id, gm in combined_df.groupby('match', sort=False):
        # Pre-aggregate per (rally, gamePlayerId) for fast LOO subtraction
        rally_player_stats = {}
        for (rid, gpid), gpg in gm.groupby(['rally_uid', 'gamePlayerId'], sort=False):
            pt = np.bincount(gpg['pointId'].clip(0, N_POINT-1).values, minlength=N_POINT).astype(float)
            ac = np.bincount(gpg['actionId'].clip(0, N_ACTION-1).values, minlength=N_ACTION).astype(float)
            rally_player_stats[(int(rid), int(gpid))] = (pt, ac, len(gpg))

        # Per-player match totals
        player_totals = {}
        for (rid, gpid), (pt, ac, n) in rally_player_stats.items():
            if gpid not in player_totals:
                player_totals[gpid] = (np.zeros(N_POINT), np.zeros(N_ACTION), 0)
            pt0, ac0, n0 = player_totals[gpid]
            player_totals[gpid] = (pt0+pt, ac0+ac, n0+n)

        # For each rally R, find ego/opp at k_per_rally[R], compute LOO stats
        for rid, gr in gm.groupby('rally_uid', sort=False):
            rid_int = int(rid)
            k = k_per_rally[rid_int]
            T = len(gr)
            k_eff = min(k, T - 1) if T > 0 else 0
            row = gr.iloc[k_eff]
            ego = int(row['gamePlayerId'])
            opp = int(row['gamePlayerOtherId'])

            # Ego stats LOO
            ego_pt_tot, ego_ac_tot, n_ego_tot = player_totals.get(ego, (np.zeros(N_POINT), np.zeros(N_ACTION), 0))
            ego_pt_R, ego_ac_R, n_ego_R = rally_player_stats.get((rid_int, ego), (np.zeros(N_POINT), np.zeros(N_ACTION), 0))
            ego_pt = ego_pt_tot - ego_pt_R; ego_ac = ego_ac_tot - ego_ac_R; n_ego = n_ego_tot - n_ego_R
            if n_ego > 0:
                ego_pt = ego_pt / n_ego; ego_ac = ego_ac / n_ego

            # Opp stats LOO
            opp_pt_tot, opp_ac_tot, n_opp_tot = player_totals.get(opp, (np.zeros(N_POINT), np.zeros(N_ACTION), 0))
            opp_pt_R, opp_ac_R, n_opp_R = rally_player_stats.get((rid_int, opp), (np.zeros(N_POINT), np.zeros(N_ACTION), 0))
            opp_pt = opp_pt_tot - opp_pt_R; opp_ac = opp_ac_tot - opp_ac_R; n_opp = n_opp_tot - n_opp_R
            if n_opp > 0:
                opp_pt = opp_pt / n_opp; opp_ac = opp_ac / n_opp

            ctx = np.concatenate([ego_pt, ego_ac, opp_pt, opp_ac]).astype(np.float32)
            contexts[rid_int] = ctx
    return contexts


# ============= Stage 3: V25-A BiLSTM model (with 58-dim ctx) =============

class TTSSLLSTMHier(nn.Module):
    """v25-A: BiLSTM + opponent-pair context (58-dim) concat at final hidden state."""
    def __init__(self, vocab_sizes):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(vocab_sizes[c], EMB_DIM, padding_idx=PAD_TOKEN) for c in FEATURES])
        in_dim = len(FEATURES) * EMB_DIM
        self.input_proj = nn.Linear(in_dim, PROJ_DIM)
        self.input_drop = nn.Dropout(DROPOUT)
        self.lstm = nn.LSTM(PROJ_DIM, HIDDEN, num_layers=N_LAYERS, batch_first=True,
                             bidirectional=BIDIR, dropout=0.0)
        lstm_out = HIDDEN * (2 if BIDIR else 1)
        head_in = lstm_out + CTX_DIM  # NEW: concat match context
        self.head_action = nn.Linear(head_in, N_ACTION)
        self.head_point = nn.Linear(head_in, N_POINT)
        self.head_winner = nn.Linear(head_in, 1)
        serve_mask = torch.zeros(N_ACTION, dtype=torch.bool); serve_mask[15:19] = True
        self.register_buffer('serve_mask', serve_mask)

    def load_ssl(self, ssl_state):
        own = self.state_dict(); n = 0
        for k, v in ssl_state.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v); n += 1
        return n

    def forward(self, X, lens, ctx):
        emb = torch.cat([self.embs[i](X[:, :, i]) for i in range(X.shape[2])], dim=-1)
        h = self.input_drop(self.input_proj(emb))
        packed = nn.utils.rnn.pack_padded_sequence(h, lens.cpu(), batch_first=True, enforce_sorted=False)
        o, _ = self.lstm(packed)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=X.shape[1])
        idx = (lens - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, o.size(-1)).to(X.device)
        last = o.gather(1, idx).squeeze(1)
        h_full = torch.cat([last, ctx], dim=-1)
        la = self.head_action(h_full).masked_fill(self.serve_mask, -1e9)
        return la, self.head_point(h_full), self.head_winner(h_full).squeeze(-1)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=LABEL_SMOOTH):
        super().__init__()
        self.weight = weight; self.gamma = gamma; self.ls = label_smoothing
    def forward(self, logits, targets):
        K = logits.size(-1)
        target_oh = torch.full_like(logits, self.ls / max(K - 1, 1))
        target_oh.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)
        log_p = F.log_softmax(logits, dim=-1).clamp(min=-30.0)
        focal = (1 - log_p.exp()).pow(self.gamma)
        loss = -focal * log_p
        if self.weight is not None: loss = loss * self.weight.unsqueeze(0)
        return (target_oh * loss).sum(dim=-1).mean()


def build_asym_label_distribution():
    """V27 ONLY: asym spatial label smoothing for class 3 (反手短球).
    Class 3: 0.80 self + 0.075 each at {2, 6} + 0.0625 each at others.
    Other classes: standard uniform 0.10 label smoothing.
    """
    smoothed = np.zeros((N_POINT, N_POINT), dtype=np.float32)
    for c in range(N_POINT):
        if c == FOCUS_CLASS:
            n_nb = len(FOCUS_NEIGHBORS); n_other = N_POINT - 1 - n_nb
            smoothed[c, c] = 1.0 - FOCUS_SPATIAL_EPS - FOCUS_UNIFORM_EPS
            for j in range(N_POINT):
                if j == c: continue
                smoothed[c, j] = (FOCUS_SPATIAL_EPS / n_nb) if j in FOCUS_NEIGHBORS else (FOCUS_UNIFORM_EPS / n_other)
        else:
            smoothed[c, c] = 1.0 - LABEL_SMOOTH
            for j in range(N_POINT):
                if j != c: smoothed[c, j] = LABEL_SMOOTH / (N_POINT - 1)
    return smoothed


class AsymSpatialFocalLoss(nn.Module):
    """V27 ONLY: focal loss with asym spatial smoothing on class 3 (反手短球 → {2, 6})."""
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.register_buffer('smoothed', torch.from_numpy(build_asym_label_distribution()))
        self.weight = weight; self.gamma = gamma
    def forward(self, logits, targets):
        td = self.smoothed[targets]
        log_p = F.log_softmax(logits, dim=-1).clamp(min=-30.0)
        focal = (1 - log_p.exp()).pow(self.gamma)
        loss = -focal * log_p
        if self.weight is not None: loss = loss * self.weight.unsqueeze(0)
        return (td * loss).sum(dim=-1).mean()


# ============= Stage 4: V25-A bag training (10 seeds × 5 folds) =============

def stage4_bag_one_seed(seed: int, variant: str = 'v25a'):
    """Train V25-A (FocalLoss) or V27 (AsymSpatialFocalLoss) bag for one seed.

    Both variants share the SAME architecture (TTSSLLSTMHier with 58-dim opp-pair ctx).
    Difference: V27 uses AsymSpatialFocalLoss on POINT head (class 3 spatial smoothing).

    Args:
        seed: random seed
        variant: 'v25a' or 'v27'
    """
    assert variant in ('v25a', 'v27'), f'unknown variant: {variant}'
    suffix = '' if seed == 42 else f'_seed{seed}'
    out_path = CACHE_DIR / f'oof_test_{variant}{suffix}.npz'
    if out_path.exists():
        log(f'  seed {seed} ({variant}) cached: {out_path.name}')
        return

    torch.manual_seed(seed); np.random.seed(seed)
    train_df = pd.read_csv(TRAIN_CSV); test_df = pd.read_csv(TEST_CSV)
    train_df['strikeNumber'] = train_df['strikeNumber'].clip(0, MAX_LEN_TT)
    test_df['strikeNumber'] = test_df['strikeNumber'].clip(0, MAX_LEN_TT)

    # k_per_rally must be defined BEFORE match contexts (oppair needs k to find ego/opp at predict position)
    test_max_strike = test_df.groupby('rally_uid')['strikeNumber'].max()
    _max_k = int(max(test_max_strike.max(), train_df.groupby('rally_uid').size().max()))
    test_k_dist = np.zeros(_max_k + 2)
    for k, c in test_max_strike.value_counts().items(): test_k_dist[int(k)] = c
    test_k_dist /= test_k_dist.sum()
    def sample_k(T, rng):
        if T < 2: return 1
        T_eff = min(int(T), len(test_k_dist))
        valid = test_k_dist[1:T_eff]; s = valid.sum()
        return int(rng.choice(np.arange(1, T_eff), p=valid / s)) if s > 0 else T - 1
    rng_main = np.random.RandomState(42)
    # Need rally Ts BEFORE building rallies; use groupby size
    train_T = train_df.groupby('rally_uid').size()
    test_T = test_df.groupby('rally_uid').size()
    k_per_rally_pre = {}
    for rid in train_T.index: k_per_rally_pre[int(rid)] = sample_k(int(train_T[rid]), rng_main)
    for rid in test_T.index: k_per_rally_pre[int(rid)] = sample_k(int(test_T[rid]), rng_main)

    # Opponent-pair contexts on COMBINED train+test
    combined = pd.concat([
        train_df[['rally_uid', 'match', 'pointId', 'actionId', 'gamePlayerId', 'gamePlayerOtherId', 'strikeNumber']],
        test_df[['rally_uid', 'match', 'pointId', 'actionId', 'gamePlayerId', 'gamePlayerOtherId', 'strikeNumber']]
    ], ignore_index=True)
    match_contexts = compute_oppair_contexts(combined, k_per_rally_pre)

    # Vocab from train+test combined (cold-start: OOV → token=1)
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    encoders, vocab_sizes = {}, {}
    for col in FEATURES:
        cats = sorted(all_df[col].dropna().unique().tolist())
        encoders[col] = {c: i + 2 for i, c in enumerate(cats)}
        vocab_sizes[col] = len(cats) + 2

    def encode_df(df):
        return np.stack([df[c].map(encoders[c]).fillna(OOV_TOKEN).astype(np.int64).values for c in FEATURES], axis=1)

    def build_rallies(df, source='train'):
        out = []
        for rid, g in df.groupby('rally_uid', sort=False):
            g = g.sort_values('strikeNumber')
            rid_int = int(rid)
            out.append({
                'rally_uid': rid_int, 'match': int(g['match'].iloc[0]),
                'T': len(g), 'X': encode_df(g),
                'actions': g['actionId'].values.astype(np.int64),
                'points': g['pointId'].values.astype(np.int64),
                'winner': int(g['serverGetPoint'].iloc[0]) if 'serverGetPoint' in g.columns else 0,
                'ctx': match_contexts[rid_int],
                'source': source,
            })
        return out

    all_train = build_rallies(train_df, 'train')
    all_test = build_rallies(test_df, 'test')
    train_matches = np.array([r['match'] for r in all_train])
    fold_splits = list(GroupKFold(n_splits=N_FOLDS).split(np.arange(len(all_train)), groups=train_matches))

    # k_per_rally already computed above (used for opponent-pair contexts).
    # Reuse k_per_rally_pre as authoritative source.
    k_per_rally = k_per_rally_pre
    test_keys = {int(rid): int(g['strikeNumber'].max()) for rid, g in test_df.groupby('rally_uid', sort=False)}

    class DS(Dataset):
        def __init__(self, rallies, mode='train'): self.r = rallies; self.mode = mode
        def __len__(self): return len(self.r)
        def __getitem__(self, i):
            r = self.r[i]; T = r['T']
            if self.mode == 'train':
                k = sample_k(T, np.random); ya, yp = r['actions'][k], r['points'][k]
            else:
                k = max(1, min(r['k'], T))
                ya, yp = (r['actions'][k], r['points'][k]) if k < T else (0, 0)
            X = r['X'][:k].copy()
            if self.mode == 'train':
                X[np.random.random(k) < PLAYER_MASK_P, 11] = OOV_TOKEN
                X[np.random.random(k) < PLAYER_MASK_P, 12] = OOV_TOKEN
            return X, ya, yp, float(r['winner']), r['ctx']

    def collate(batch):
        Xs, ya, yp, yw, ctxs = zip(*batch)
        lens = torch.tensor([len(X) for X in Xs])
        F_ = Xs[0].shape[1]
        X_pad = torch.zeros(len(Xs), int(lens.max()), F_, dtype=torch.long)
        for i, X in enumerate(Xs): X_pad[i, :len(X)] = torch.from_numpy(X)
        ctx_t = torch.from_numpy(np.stack(ctxs))
        return X_pad, lens, torch.tensor(ya), torch.tensor(yp), torch.tensor(yw, dtype=torch.float32), ctx_t

    def class_w_sqrt(rallies, n, key):
        cnt = np.ones(n)
        for r in rallies:
            for v in r[key]:
                if 0 <= v < n: cnt[v] += 1
        w = 1.0 / np.sqrt(cnt)
        return torch.tensor(w * (n / w.sum()), dtype=torch.float32, device=device)

    ssl = torch.load(SSL_CKPT, map_location=device, weights_only=False)
    n_train = len(all_train); rid_to_idx = {r['rally_uid']: i for i, r in enumerate(all_train)}
    oof_a = np.zeros((n_train, N_ACTION)); oof_p = np.zeros((n_train, N_POINT)); oof_w = np.zeros(n_train)
    test_a = np.zeros((N_FOLDS, len(all_test), N_ACTION))
    test_p = np.zeros((N_FOLDS, len(all_test), N_POINT))
    test_w = np.zeros((N_FOLDS, len(all_test)))

    for fold, (tr_idx, va_idx) in enumerate(fold_splits):
        torch.manual_seed(seed + fold); np.random.seed(seed + fold)
        tr_rids = set(all_train[i]['rally_uid'] for i in tr_idx)
        va_rids = set(all_train[i]['rally_uid'] for i in va_idx)
        # Transductive aug: test rallies (T>=2) added to training set
        usable_test = [r for r in all_test if r['T'] >= 2]
        tr_r = [r for r in all_train if r['rally_uid'] in tr_rids] + usable_test
        va_r = [{**r, 'k': k_per_rally[r['rally_uid']]} for r in all_train if r['rally_uid'] in va_rids]
        te_r = [{**r, 'k': test_keys[r['rally_uid']]} for r in all_test]

        tr_loader = DataLoader(DS(tr_r, 'train'), batch_size=FT_BS, shuffle=True, collate_fn=collate)
        va_loader = DataLoader(DS(va_r, 'val'), batch_size=128, shuffle=False, collate_fn=collate)
        te_loader = DataLoader(DS(te_r, 'val'), batch_size=128, shuffle=False, collate_fn=collate)

        act_w = class_w_sqrt(tr_r, N_ACTION, 'actions'); pt_w = class_w_sqrt(tr_r, N_POINT, 'points')
        model = TTSSLLSTMHier(vocab_sizes).to(device)
        n_loaded = model.load_ssl(ssl['state'])
        if fold == 0 and seed == 42:
            log(f'    [{variant}] model params={sum(p.numel() for p in model.parameters())/1e6:.2f}M, SSL loaded {n_loaded} keys, CTX_DIM={CTX_DIM}')

        opt = torch.optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=FT_WEIGHT_DECAY)
        warmup = max(1, len(tr_loader)); total = len(tr_loader) * FT_EPOCHS
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s:
            float(s + 1)/warmup if s < warmup
            else 0.5*(1 + np.cos(np.pi*(s-warmup)/max(1,total-warmup))))
        ce_a = FocalLoss(weight=act_w, gamma=2.0)
        # Point head loss: variant-dependent
        if variant == 'v27':
            ce_p = AsymSpatialFocalLoss(weight=pt_w, gamma=2.0).to(device)
        else:  # v25a
            ce_p = FocalLoss(weight=pt_w, gamma=2.0)
        bce = nn.BCEWithLogitsLoss()
        best_state = None; best_final = -1.0
        t0 = time.time()
        for ep in range(1, FT_EPOCHS + 1):
            model.train()
            for X, lens, ya, yp, yw, ctx in tr_loader:
                X, lens, ya, yp, yw, ctx = X.to(device), lens.to(device), ya.to(device), yp.to(device), yw.to(device), ctx.to(device)
                la, lp, lw = model(X, lens, ctx)
                loss = 0.4 * ce_a(la, ya) + 0.4 * ce_p(lp, yp) + 0.2 * bce(lw, yw)
                if not torch.isfinite(loss): opt.zero_grad(); continue
                opt.zero_grad(); loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                if not torch.isfinite(gn): opt.zero_grad(); continue
                opt.step(); sch.step()
            model.eval(); va_a, va_p, va_at, va_pt, va_w_t, va_w_p = [], [], [], [], [], []
            with torch.no_grad():
                for X, lens, ya, yp, yw, ctx in va_loader:
                    X, lens, ctx = X.to(device), lens.to(device), ctx.to(device)
                    la, lp, lw = model(X, lens, ctx)
                    va_at += ya.tolist(); va_a += la.argmax(-1).cpu().tolist()
                    va_pt += yp.tolist(); va_p += lp.argmax(-1).cpu().tolist()
                    va_w_t += yw.tolist(); va_w_p += torch.sigmoid(lw).cpu().tolist()
            f1_a = f1_score(va_at, va_a, average='macro', zero_division=0)
            f1_p = f1_score(va_pt, va_p, average='macro', zero_division=0)
            auc = roc_auc_score(va_w_t, va_w_p) if len(set(va_w_t)) > 1 else 0.5
            final = 0.4 * f1_a + 0.4 * f1_p + 0.2 * auc
            if final > best_final:
                best_final = final
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        log(f'    fold {fold}: F1_a={f1_a:.4f} F1_p={f1_p:.4f} AUC={auc:.4f} best_Final={best_final:.4f} ({time.time()-t0:.0f}s)')

        # Predict val + test
        model.eval(); pa_v, pp_v, pw_v = [], [], []
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
            i = rid_to_idx[r['rally_uid']]
            oof_a[i] = vA[j]; oof_p[i] = vP[j]; oof_w[i] = vW[j]
        test_a[fold] = tA; test_p[fold] = tP; test_w[fold] = tW

    np.savez(out_path, oof_tt_a=oof_a, oof_tt_p=oof_p, oof_tt_w=oof_w,
             test_tt_a=test_a, test_tt_p=test_p, test_tt_w=test_w)
    log(f'  seed {seed} saved → {out_path.name}')


def stage4_bag(seeds, variant='v25a'):
    """Train all seeds for a bag variant. variant in {'v25a', 'v27'}."""
    log(f'=== Stage 4: {variant.upper()} bag training ({len(seeds)} seeds × {N_FOLDS} folds) ===')
    for s in seeds:
        t0 = time.time()
        log(f'  --- {variant} seed {s} ---')
        stage4_bag_one_seed(s, variant=variant)
        log(f'  {variant} seed {s} done in {time.time()-t0:.0f}s')


# ============= Stage 5: V27 Mode A ensemble substitute α-search =============

f1m = lambda y, p: f1_score(y, p, average='macro', zero_division=0)


def load_bag(tag, seeds):
    a, p, w, ta, tp, tw = [], [], [], [], [], []
    for s in seeds:
        suf = '' if s == 42 else f'_seed{s}'
        d = np.load(CACHE_DIR / f'oof_test_tt_shuttlenet{suf}{tag}.npz')
        a.append(d['oof_tt_a']); p.append(d['oof_tt_p']); w.append(d['oof_tt_w'])
        ta.append(d['test_tt_a'].mean(0)); tp.append(d['test_tt_p'].mean(0)); tw.append(d['test_tt_w'].mean(0))
    return np.mean(a, 0), np.mean(p, 0), np.mean(w, 0), np.mean(ta, 0), np.mean(tp, 0), np.mean(tw, 0)


def load_v27_bag(seeds):
    a, p, w, ta, tp, tw = [], [], [], [], [], []
    for s in seeds:
        suf = '' if s == 42 else f'_seed{s}'
        d = np.load(CACHE_DIR / f'oof_test_v27{suf}.npz')
        a.append(d['oof_tt_a']); p.append(d['oof_tt_p']); w.append(d['oof_tt_w'])
        ta.append(d['test_tt_a'].mean(0)); tp.append(d['test_tt_p'].mean(0)); tw.append(d['test_tt_w'].mean(0))
    return np.mean(a, 0), np.mean(p, 0), np.mean(w, 0), np.mean(ta, 0), np.mean(tp, 0), np.mean(tw, 0)


def load_v25a_bag(seeds):
    a, p, w, ta, tp, tw = [], [], [], [], [], []
    for s in seeds:
        suf = '' if s == 42 else f'_seed{s}'
        d = np.load(CACHE_DIR / f'oof_test_v25a{suf}.npz')
        a.append(d['oof_tt_a']); p.append(d['oof_tt_p']); w.append(d['oof_tt_w'])
        ta.append(d['test_tt_a'].mean(0)); tp.append(d['test_tt_p'].mean(0)); tw.append(d['test_tt_w'].mean(0))
    return np.mean(a, 0), np.mean(p, 0), np.mean(w, 0), np.mean(ta, 0), np.mean(tp, 0), np.mean(tw, 0)


def search_grid(probs, y, n, step, metric='f1'):
    best, ba = -1, None
    grid = np.arange(0, 1.0001, step)
    for combo in product(grid, repeat=n - 1):
        last = 1 - sum(combo)
        if last < -1e-9 or last > 1 + 1e-9: continue
        last = max(0, min(1, last))
        alphas = list(combo) + [last]
        ens = sum(a * p for a, p in zip(alphas, probs))
        s = f1m(y, ens.argmax(-1)) if metric == 'f1' else roc_auc_score(y, ens)
        if s > best: best, ba = s, tuple(alphas)
    return ba, best


def coord_descent(probs, y, init_alphas, step=0.05, max_iter=10, metric='f1'):
    alphas = list(init_alphas); n = len(alphas)
    grid = np.arange(0, 1.0001, step)
    ens = sum(a * p for a, p in zip(alphas, probs))
    best = f1m(y, ens.argmax(-1)) if metric == 'f1' else roc_auc_score(y, ens)
    for _ in range(max_iter):
        improved = False
        for i in range(n):
            best_a_i = alphas[i]; best_s = best; best_alphas = alphas[:]
            for new_a_i in grid:
                rest = [alphas[j] for j in range(n) if j != i]
                rest_sum = sum(rest)
                if rest_sum < 1e-9:
                    if abs(new_a_i - 1) > 1e-9: continue
                    cand = [0.0] * n; cand[i] = 1.0
                else:
                    scale = (1 - new_a_i) / rest_sum
                    cand = [a * scale for a in alphas]; cand[i] = new_a_i
                ens = sum(a * p for a, p in zip(cand, probs))
                s = f1m(y, ens.argmax(-1)) if metric == 'f1' else roc_auc_score(y, ens)
                if s > best_s + 1e-6:
                    best_s, best_a_i = s, new_a_i; best_alphas = cand[:]
            if best_a_i != alphas[i]:
                alphas = best_alphas; best = best_s; improved = True
        if not improved: break
    return tuple(alphas), best


def tune_thresh(probs, y, n_iter=4):
    grid = np.concatenate([np.arange(0.5, 3.0, 0.05), np.arange(3.0, 5.0, 0.25)])
    n = probs.shape[1]; m = np.ones(n)
    base = f1m(y, probs.argmax(-1))
    for _ in range(n_iter):
        improved = False
        for c in range(n):
            best_m, best_s = m[c], base
            for k in grid:
                t = m.copy(); t[c] = k
                s = f1m(y, (probs * t[None, :]).argmax(-1))
                if s > best_s + 1e-6: best_s, best_m = s, k
            if best_m != m[c]:
                m[c] = best_m; base = best_s; improved = True
        if not improved: break
    cap_ratio = 0.75
    m_min, m_max = m.min(), m.max()
    if m_max - m_min > 1e-9:
        m_clipped = m_min + (m - m_min) * cap_ratio + (1 - cap_ratio) * (m_max + m_min) / 2
        s_clipped = f1m(y, (probs * m_clipped[None, :]).argmax(-1))
        if s_clipped >= base - 1e-4:
            m = m_clipped; base = s_clipped
    return m, base


def stage5_ensemble_and_submit(seeds):
    log('=== Stage 5: v27 Mode A ensemble (aug → V25-A, asym_aug → V27) + submission ===')
    # Verify required caches
    required = [V3_CACHE]
    for tag in ['_ssl_lstm', '_ssl_lstm_asym']:
        for s in seeds:
            suf = '' if s == 42 else f'_seed{s}'
            required.append(CACHE_DIR / f'oof_test_tt_shuttlenet{suf}{tag}.npz')
    for s in seeds:
        suf = '' if s == 42 else f'_seed{s}'
        required.append(CACHE_DIR / f'oof_test_v25a{suf}.npz')
        required.append(CACHE_DIR / f'oof_test_v27{suf}.npz')
    missing = [p for p in required if not p.exists()]
    if missing:
        log(f'  ERROR: {len(missing)} required caches missing:')
        for p in missing[:10]: log(f'    {p.name}')
        if len(missing) > 10: log(f'    ... and {len(missing)-10} more')
        log(f'  Run prerequisites first: Ensemble.ipynb (V3), tt_lstm_ssl_full_pipeline.py (v1/asym),')
        log(f'  this script Stage 4 (V25-A + V27 bag training).')
        raise FileNotFoundError(f'{len(missing)} caches missing')

    v3 = np.load(V3_CACHE)
    oy_a, oy_p, oy_w = v3['oof_y_a'], v3['oof_y_p'], v3['oof_y_w']
    log(f'  V3 baseline loaded: {len(oy_a)} OOF samples')
    v1_a, v1_p, v1_w, tv1_a, tv1_p, tv1_w = load_bag('_ssl_lstm', seeds); log('  v1 bag loaded')
    ay_a, ay_p, ay_w, tay_a, tay_p, tay_w = load_bag('_ssl_lstm_asym', seeds); log('  asym bag loaded')
    v25_a, v25_p, v25_w, tv25_a, tv25_p, tv25_w = load_v25a_bag(seeds); log('  V25-A bag loaded')
    v27_a, v27_p, v27_w, tv27_a, tv27_p, tv27_w = load_v27_bag(seeds); log('  V27 bag loaded')

    log(f'  V25-A standalone: F1_a={f1m(oy_a, v25_a.argmax(-1)):.4f} F1_p={f1m(oy_p, v25_p.argmax(-1)):.4f} AUC={roc_auc_score(oy_w, v25_w):.4f}')
    log(f'  V27   standalone: F1_a={f1m(oy_a, v27_a.argmax(-1)):.4f} F1_p={f1m(oy_p, v27_p.argmax(-1)):.4f} AUC={roc_auc_score(oy_w, v27_w):.4f}')

    # Action 7-way: V3-LSTM + V3-XGB + V3-Cat + v1 + asym + V25-A (aug slot) + V27 (asym_aug slot)
    log('\n  ## Action 7-way α-search (Mode A: V25-A + V27 both substitute) ##')
    a7 = [v3['oof_lstm_a'], v3['oof_xgb_a'], v3['oof_cat_a'], v1_a, ay_a, v25_a, v27_a]
    ba_a, _ = search_grid(a7, oy_a, 7, 0.1)
    ba_a, fa = coord_descent(a7, oy_a, ba_a, step=0.05)
    log(f'    α={tuple(round(x,2) for x in ba_a)} F1_a={fa:.4f}')

    # Point 8-way: V3-LSTM + V3-XGB + V3-Cat + V3-FTT + v1 + asym + V25-A + V27
    log('\n  ## Point 8-way α-search ##')
    p8 = [v3['oof_lstm_p'], v3['oof_xgb_p'], v3['oof_cat_p'], v3['oof_ftt_p'], v1_p, ay_p, v25_p, v27_p]
    candidates = []
    v16_init = (0.0, 0.0, 0.1, 0.29, 0.57, 0.05, 0.0, 0.0)
    a1, s1 = coord_descent(p8, oy_p, v16_init, step=0.05)
    candidates.append((s1, a1))
    ba2, _ = search_grid(p8, oy_p, 8, 0.2)
    a2, s2 = coord_descent(p8, oy_p, ba2, step=0.05)
    candidates.append((s2, a2))
    best_s, ba_p = max(candidates, key=lambda x: x[0])
    fp = best_s
    log(f'    α={tuple(round(x,2) for x in ba_p)} F1_p={fp:.4f}')

    # Winner 4-way: V3-LSTM + V3-XGB + V3-Cat + v1 (unchanged)
    log('\n  ## Winner 4-way α-search ##')
    w4 = [v3['oof_lstm_w'], v3['oof_xgb_w'], v3['oof_cat_w'], v1_w]
    ba_w, auc = search_grid(w4, oy_w, 4, 0.05, metric='auc')
    log(f'    α={tuple(round(x,2) for x in ba_w)} AUC={auc:.4f}')

    # Threshold tune + final OOF
    log('\n  ## Per-class threshold mults (cap=0.75) ##')
    ens_a = sum(a*p for a, p in zip(ba_a, a7))
    ens_p = sum(a*p for a, p in zip(ba_p, p8))
    ens_w = sum(a*p for a, p in zip(ba_w, w4))
    m_a, fa_t = tune_thresh(ens_a, oy_a)
    m_p, fp_t = tune_thresh(ens_p, oy_p)
    auc_t = roc_auc_score(oy_w, ens_w)
    final = 0.4 * fa_t + 0.4 * fp_t + 0.2 * auc_t
    log(f'\n  ## OOF Final ##')
    log(f'    F1_a={fa_t:.4f}  F1_p={fp_t:.4f}  AUC={auc_t:.4f}  Final={final:.4f}')
    log(f'    vs v17 OOF 0.3773: {final - 0.3773:+.4f}')
    log(f'    Expected LB at 0.68x v27 Mode A transfer: {0.3747450 + (final-0.3773)*0.68:.4f}')
    log(f'    Actual LB (2026-05-20): 0.3787701')

    # Generate submission
    log('\n  ## Generate submission ##')
    a_test = [v3['test_lstm_a'].mean(0), v3['test_xgb_a'].mean(0), v3['test_cat_a'].mean(0), tv1_a, tay_a, tv25_a, tv27_a]
    p_test = [v3['test_lstm_p'].mean(0), v3['test_xgb_p'].mean(0), v3['test_cat_p'].mean(0), v3['test_ftt_p'].mean(0), tv1_p, tay_p, tv25_p, tv27_p]
    w_test = [v3['test_lstm_w'].mean(0), v3['test_xgb_w'].mean(0), v3['test_cat_w'].mean(0), tv1_w]
    te_a = sum(a*p for a, p in zip(ba_a, a_test))
    te_p = sum(a*p for a, p in zip(ba_p, p_test))
    te_w = sum(a*p for a, p in zip(ba_w, w_test))
    te_a_pred = (te_a * m_a[None, :]).argmax(-1).astype(int)
    te_p_pred = (te_p * m_p[None, :]).argmax(-1).astype(int)
    test_df = pd.read_csv(TEST_CSV)
    test_rids = [int(rid) for rid, _ in test_df.groupby('rally_uid', sort=False)]
    sub = pd.DataFrame({
        'rally_uid': test_rids, 'actionId': te_a_pred,
        'pointId': te_p_pred, 'serverGetPoint': te_w,
    }).sort_values('rally_uid').reset_index(drop=True)
    ts = time.strftime('%Y%m%d_%H%M')
    out = SUB_DIR / f'submission_v27_modeA_canonical_{ts}.csv'
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    log(f'    saved → {out.relative_to(ROOT)} ({len(sub)} rows)')
    return out


# ============= Stage 6: Verification =============

def stage6_verify(sub_path):
    log('=== Stage 6: Submission verification ===')
    df = pd.read_csv(sub_path)
    assert list(df.columns) == ['rally_uid', 'actionId', 'pointId', 'serverGetPoint'], \
        f'wrong columns: {df.columns.tolist()}'
    assert len(df) == 1845, f'wrong row count: {len(df)} (expected 1845)'
    assert df['rally_uid'].is_unique, 'rally_uid not unique'
    assert df['actionId'].dtype == np.int64, f'actionId dtype {df["actionId"].dtype}'
    assert df['pointId'].dtype == np.int64, f'pointId dtype {df["pointId"].dtype}'
    assert df['serverGetPoint'].dtype == np.float64, f'serverGetPoint dtype {df["serverGetPoint"].dtype}'
    assert df['serverGetPoint'].between(0, 1).all(), 'serverGetPoint out of [0,1]'
    assert df['actionId'].between(0, 18).all(), f'actionId out of [0,18]: {df["actionId"].min()},{df["actionId"].max()}'
    assert df['pointId'].between(0, 9).all(), f'pointId out of [0,9]: {df["pointId"].min()},{df["pointId"].max()}'
    log(f'  ✓ columns: {df.columns.tolist()}')
    log(f'  ✓ rows: {len(df)}')
    log(f'  ✓ rally_uid unique: {df["rally_uid"].is_unique}')
    log(f'  ✓ actionId int in [{df["actionId"].min()}, {df["actionId"].max()}]')
    log(f'  ✓ pointId int in [{df["pointId"].min()}, {df["pointId"].max()}]')
    log(f'  ✓ serverGetPoint float in [{df["serverGetPoint"].min():.4f}, {df["serverGetPoint"].max():.4f}]')

    # Compare to expected V27 Mode A submission (LB 0.3787701)
    expected_candidates = (
        list((ROOT / 'submissions').glob('submission_B_v25a_plus_v27_modeA_*.csv')) +
        list((ROOT / 'submissions').glob('submission_v27_modeA_*.csv'))
    )
    if expected_candidates:
        expected = expected_candidates[0]
        new_hash = hashlib.md5(open(sub_path, 'rb').read()).hexdigest()
        expected_hash = hashlib.md5(open(expected, 'rb').read()).hexdigest()
        match = '✓ MATCH' if new_hash == expected_hash else '⚠️ DIFFERS'
        log(f'  Hash check vs LB-verified {expected.name}: {match}')
        log(f'    new:      {new_hash}')
        log(f'    expected: {expected_hash}')


# ============= Main pipeline =============

def main():
    ap = argparse.ArgumentParser(description='v27 Mode A Canonical Reproducer (LB 0.3787701)')
    ap.add_argument('--seeds', type=str, default='42,43,44,45,46,47,48,49,50,51',
                    help='Comma-separated seeds for V25-A + V27 bags (default 10-seed)')
    ap.add_argument('--skip-ssl', action='store_true', help='Skip SSL pretrain (use cached)')
    ap.add_argument('--skip-bag', action='store_true', help='Skip bag training (use cached)')
    ap.add_argument('--only-ensemble', action='store_true', help='Only run ensemble + submission (skip 1+4)')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    log(f'v27 Mode A Canonical Reproducer — seeds {seeds}')

    if args.only_ensemble:
        log('--only-ensemble: skip Stages 1+4, run ensemble + verify only')
    else:
        if not args.skip_ssl:
            stage1_ssl_pretrain()
        if not args.skip_bag:
            stage4_bag(seeds, variant='v25a')
            stage4_bag(seeds, variant='v27')

    sub_path = stage5_ensemble_and_submit(seeds)
    stage6_verify(sub_path)

    log('=== v27 Mode A Canonical Reproducer DONE ===')


if __name__ == '__main__':
    main()
