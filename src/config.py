"""Centralized configuration: paths, hyperparameters, constants.

All other modules import from here. Modifying paths or hyperparameters here
affects the entire pipeline consistently.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

# ===================================================================
# Paths — all relative to the deliverable root (parent of src/)
# ===================================================================
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
SUB_DIR = ROOT / "submissions"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test_new.csv"
OLD_TEST_CSV = DATA_DIR / "test.csv"
SS22_TRAIN = DATA_DIR / "external" / "shuttleset22" / "train.csv"
V3_CACHE = CACHE_DIR / "oof_test_probs.npz"
SSL_CKPT = CACHE_DIR / "ssl_lstm_encoder_shuttleset22.pt"

# ===================================================================
# Model architecture dimensions
# ===================================================================
EMB_DIM = 32         # per-feature embedding dim
PROJ_DIM = 128       # input projection after concat
HIDDEN = 128         # BiLSTM hidden state size
N_LAYERS = 1
BIDIR = True
DROPOUT = 0.30

# ===================================================================
# Data dimensions
# ===================================================================
N_ACTION = 19        # 0=none, 1-7=Attack, 8-11=Control, 12-14=Defensive, 15-18=Serve
N_POINT = 10         # 0=out-of-bounds, 1-9=3x3 grid
MAX_LEN_TT = 40      # max strokes per rally for TT
CTX_DIM = 2 * (N_POINT + N_ACTION)  # 58 = ego_pt(10) + ego_act(19) + opp_pt(10) + opp_act(19)
N_FOLDS = 5

# Special vocab tokens
PAD_TOKEN = 0
OOV_TOKEN = 1
MASK_TOKEN = 2

# Per-stroke categorical features for TT data
FEATURES = [
    "sex", "handId", "strengthId", "spinId", "pointId", "actionId",
    "positionId", "strikeId", "scoreSelf", "scoreOther", "strikeNumber",
    "gamePlayerId", "gamePlayerOtherId",
]

# ===================================================================
# SSL pretrain hyperparameters (ShuttleSet22)
# ===================================================================
SS22_CAT_FEATS = ["type", "landing_area", "player_location_area", "opponent_location_area"]
MLM_TARGETS = ["type", "landing_area"]
MLM_HEAD_KEY = {"type": "shottype", "landing_area": "area"}
MASK_PROB = 0.15
MAX_LEN_PRETRAIN = 60
PRETRAIN_BS = 64
PRETRAIN_LR = 1e-3
PRETRAIN_EPOCHS = 30

# ===================================================================
# Bag training hyperparameters
# ===================================================================
FT_LR = 1e-3
FT_WEIGHT_DECAY = 1e-5
FT_BS = 64
FT_EPOCHS = 30
PLAYER_MASK_P = 0.30
LABEL_SMOOTH = 0.10
GRAD_CLIP = 1.0
SEEDS = list(range(42, 52))  # 10 seeds: 42..51

# ===================================================================
# AsymSpatialFocalLoss focus zone (V27 point-head innovation)
# ===================================================================
FOCUS_CLASS = 3            # 反手短球 (backhand short, rarest 0.9% in train)
FOCUS_NEIGHBORS = [2, 6]   # row neighbor (中間短) + column neighbor (反手半長)
FOCUS_SPATIAL_EPS = 0.15   # mass redistributed to spatial neighbors
FOCUS_UNIFORM_EPS = 0.05   # mass redistributed to non-neighbor classes

# ===================================================================
# Reproducibility verification — expected MD5 of the canonical submission
# ===================================================================
EXPECTED_MD5 = "c10097155c0942354f81ea188b43f111"

# ===================================================================
# Compute device + log helper
# ===================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    """Timestamped console logger used across the pipeline."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
