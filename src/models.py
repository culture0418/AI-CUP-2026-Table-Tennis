"""Neural network architectures.

Two classes:
  - SSLEncoderLSTM: BiLSTM + MLM heads, used for cross-sport SSL pretrain on ShuttleSet22.
  - TTSSLLSTMHier:  BiLSTM + 58-dim opp-pair context + 3 task heads, used for TT finetune.

Both share the same BiLSTM core; SSL-pretrained `lstm.*` weights transfer to TT.
Embeddings and input projection are NOT transferred (different feature counts).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import (
    PROJ_DIM, HIDDEN, EMB_DIM, DROPOUT, N_LAYERS, BIDIR,
    PAD_TOKEN, N_ACTION, N_POINT, CTX_DIM,
    SS22_CAT_FEATS, FEATURES, MLM_TARGETS, MLM_HEAD_KEY,
)


class SSLEncoderLSTM(nn.Module):
    """BiLSTM encoder for ShuttleSet22 MLM pretrain (羽球).

    After SSL training, only the `lstm.*` weights are transferable to the TT model
    (the badminton-specific embeddings and projection are discarded on transfer).
    """

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


class TTSSLLSTMHier(nn.Module):
    """V25-A / V27 backbone — BiLSTM + 58-dim opponent-pair context + 3 heads.

    Shared architecture for both V25-A and V27 variants. The only difference
    between the two is the point-head loss (FocalLoss vs AsymSpatialFocalLoss).
    """

    def __init__(self, vocab_sizes):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(vocab_sizes[c], EMB_DIM, padding_idx=PAD_TOKEN) for c in FEATURES
        ])
        in_dim = len(FEATURES) * EMB_DIM
        self.input_proj = nn.Linear(in_dim, PROJ_DIM)
        self.input_drop = nn.Dropout(DROPOUT)
        self.lstm = nn.LSTM(PROJ_DIM, HIDDEN, num_layers=N_LAYERS, batch_first=True,
                            bidirectional=BIDIR, dropout=0.0)
        lstm_out = HIDDEN * (2 if BIDIR else 1)
        head_in = lstm_out + CTX_DIM  # 256 + 58 = 314
        # Three parallel task heads.
        self.head_action = nn.Linear(head_in, N_ACTION)
        self.head_point = nn.Linear(head_in, N_POINT)
        self.head_winner = nn.Linear(head_in, 1)
        # Mask: serve actions (15-18) cannot appear at non-first strokes.
        serve_mask = torch.zeros(N_ACTION, dtype=torch.bool)
        serve_mask[15:19] = True
        self.register_buffer("serve_mask", serve_mask)

    def load_ssl(self, ssl_state):
        """Load matching keys from an SSL-pretrained state_dict (only `lstm.*` transfers)."""
        own = self.state_dict()
        n = 0
        for k, v in ssl_state.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v)
                n += 1
        return n

    def forward(self, X, lens, ctx):
        emb = torch.cat([self.embs[i](X[:, :, i]) for i in range(X.shape[2])], dim=-1)
        h = self.input_drop(self.input_proj(emb))
        packed = nn.utils.rnn.pack_padded_sequence(h, lens.cpu(), batch_first=True, enforce_sorted=False)
        o, _ = self.lstm(packed)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=X.shape[1])
        # Gather the last visible stroke's hidden state per sequence.
        idx = (lens - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, o.size(-1)).to(X.device)
        last = o.gather(1, idx).squeeze(1)
        # Concat with opp-pair context.
        h_full = torch.cat([last, ctx], dim=-1)
        # Action head with serve mask (serves only at strikeNumber=1, never as next-stroke prediction).
        la = self.head_action(h_full).masked_fill(self.serve_mask, -1e9)
        return la, self.head_point(h_full), self.head_winner(h_full).squeeze(-1)
