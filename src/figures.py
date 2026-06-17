"""Generate the 6 figures referenced in docs/aicup2026_report.md 陸段.

Outputs to docs/figures/fig1_*.png ... fig6_*.png. English labels are used
because the system fonts available do not include CJK glyphs.

Re-runnable: judges can regenerate figures via the entry point
`python scripts/run_full_pipeline.py --figures` after the canonical pipeline
has populated `cache/`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from .config import ROOT, CACHE_DIR, SEEDS, log


FIG_DIR = ROOT / "docs" / "figures"

# Matplotlib defaults — clean publication look
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 180,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _load_bag(name: str):
    """Load a bag (mean over seeds) — returns (oof_a, oof_p, oof_w)."""
    oa, op, ow = [], [], []
    for s in SEEDS:
        suf = "" if s == 42 else f"_seed{s}"
        d = np.load(CACHE_DIR / f"oof_test_{name}{suf}.npz")
        oa.append(d["oof_tt_a"])
        op.append(d["oof_tt_p"])
        ow.append(d["oof_tt_w"])
    return np.mean(oa, 0), np.mean(op, 0), np.mean(ow, 0)


# ============================================================
# Fig 1: V38 chain — OOF gates all passed but LB crashed
# ============================================================
def fig1_v38_chain():
    stages = ["Single-seed\nFinal Δ", "10-seed bag\nFinal Δ",
              "Frozen-α OOF\nFinal Δ", "External LB\nFinal Δ"]
    values = [0.0070, 0.0047, 0.0031, -0.0196]
    colors = ["#4CAF50", "#4CAF50", "#4CAF50", "#E53935"]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bars = ax.bar(stages, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", lw=0.8)
    for bar, v in zip(bars, values):
        ypos = v + (0.0008 if v > 0 else -0.0014)
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{v:+.4f}", ha="center", fontweight="bold")
    ax.set_ylabel("Final Score Δ vs V27 Mode A baseline")
    ax.set_title("Fig 1. V38 (head-decoupled adapters + soft cascade) — every offline\n"
                 "gate passed but external LB crashed (worst arch-integration failure on record)")
    ax.set_ylim(-0.027, 0.011)
    ax.annotate("OOF → LB inversion\n(every offline gate passed)",
                xy=(3, -0.0196), xytext=(1.4, -0.024),
                color="#E53935", ha="center", fontweight="bold", fontsize=10.5,
                arrowprops=dict(arrowstyle="->", color="#E53935", lw=1.2))
    plt.tight_layout()
    out = FIG_DIR / "fig1_v38_chain.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    log(f"✓ {out.relative_to(ROOT)}")


# ============================================================
# Fig 2: Flip count vs LB delta — monotonic harm relationship
# ============================================================
def fig2_flip_vs_lb():
    data = [
        ("V25A60 frozen-α",  31, -0.00012),
        ("V37 add",         421, -0.0081),
        ("V27-60 frozen-α", 422, -0.0091),
        ("V27-60 re-α",     477, -0.0111),
        ("V38 frozen-α",    504, -0.0196),
    ]
    names = [d[0] for d in data]
    flips = np.array([d[1] for d in data])
    deltas = np.array([d[2] for d in data])

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.scatter(flips, deltas, s=140, color="#1976D2", edgecolor="black", linewidth=0.7, zorder=3)
    for name, x, y in zip(names, flips, deltas):
        offset_y = 0.0006 if name == "V25A60 frozen-α" else -0.0010
        ax.annotate(name, (x, y), xytext=(x + 12, y + offset_y),
                    fontsize=10, fontweight="bold")
    z = np.polyfit(flips, deltas, 1)
    xs = np.linspace(flips.min() * 0.9, flips.max() * 1.05, 50)
    ax.plot(xs, np.polyval(z, xs), "--", color="gray", lw=1.2,
            label=f"linear fit: ΔLB ≈ {z[0]:.6f}·flips + {z[1]:.4f}")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Total flips (action + point) vs best submission")
    ax.set_ylabel("LB Final Δ vs best (0.4472604)")
    ax.set_title("Fig 2. Monotonic harm — more flips = larger LB loss\n"
                 "(loss-per-flip also rises with architectural novelty)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = FIG_DIR / "fig2_flip_vs_lb.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    log(f"✓ {out.relative_to(ROOT)}")


# ============================================================
# Fig 3: V25-A vs V3 baseline — per-class action F1 (success case)
# ============================================================
def fig3_v25a_vs_v3_per_class():
    v3 = np.load(CACHE_DIR / "oof_test_probs.npz")
    v25a_oof_a, _, _ = _load_bag("v25a")
    y_a = v3["oof_y_a"]
    base_a = v3["oof_lstm_a"]
    base_pred = base_a.argmax(1)
    v25_pred = v25a_oof_a.argmax(1)

    f1_v3 = f1_score(y_a, base_pred, average=None, zero_division=0)[:15]
    f1_v25 = f1_score(y_a, v25_pred, average=None, zero_division=0)[:15]

    x = np.arange(15)
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w / 2, f1_v3, w, color="#90A4AE", edgecolor="black", linewidth=0.5,
           label="V3-LSTM baseline")
    ax.bar(x + w / 2, f1_v25, w, color="#43A047", edgecolor="black", linewidth=0.5,
           label="V25-A (BiLSTM+SSL+opp-pair ctx)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in range(15)])
    ax.set_xlabel("actionId (class 0-14)")
    ax.set_ylabel("Per-class macro-F1 (OOF)")
    ax.set_title("Fig 3. V25-A improvement over V3-LSTM baseline on action head\n"
                 "(SSL+opp-pair context lifts rare attack/control classes)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    out = FIG_DIR / "fig3_v25a_vs_v3_per_class.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    log(f"✓ {out.relative_to(ROOT)}")


# ============================================================
# Fig 4: AsymSpatial loss — class 3 label distribution
# ============================================================
def fig4_asym_spatial_label():
    N_POINT = 10
    LABEL_SMOOTH = 0.10
    FOCUS_SPATIAL_EPS = 0.15
    FOCUS_UNIFORM_EPS = 0.05
    FOCUS_NEIGHBORS = [2, 6]
    FOCUS_CLASS = 3

    v25 = np.full(N_POINT, LABEL_SMOOTH / (N_POINT - 1))
    v25[FOCUS_CLASS] = 1.0 - LABEL_SMOOTH

    v27 = np.zeros(N_POINT)
    n_other = N_POINT - 1 - len(FOCUS_NEIGHBORS)
    v27[FOCUS_CLASS] = 1.0 - FOCUS_SPATIAL_EPS - FOCUS_UNIFORM_EPS
    for j in range(N_POINT):
        if j == FOCUS_CLASS:
            continue
        v27[j] = (FOCUS_SPATIAL_EPS / len(FOCUS_NEIGHBORS)) if j in FOCUS_NEIGHBORS \
                 else (FOCUS_UNIFORM_EPS / n_other)

    classes = np.arange(N_POINT)
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(classes - w / 2, v25, w, color="#90A4AE", edgecolor="black", linewidth=0.5,
           label="V25-A FocalLoss (uniform label-smoothing 0.10)")
    ax.bar(classes + w / 2, v27, w, color="#FB8C00", edgecolor="black", linewidth=0.5,
           label="V27 AsymSpatialFocalLoss (class 3 → {2,6} spatial neighbors)")
    for j in FOCUS_NEIGHBORS:
        ax.annotate("spatial\nneighbor",
                    xy=(j + w / 2, v27[j]),
                    xytext=(j + w / 2 + 0.4, v27[j] + 0.05),
                    fontsize=9, color="#E65100",
                    arrowprops=dict(arrowstyle="->", color="#E65100"))
    ax.set_xticks(classes)
    ax.set_xticklabels(["0\n(out)", "1\nFH-short", "2\nMid-short", "3\nBH-short★",
                         "4\nFH-half", "5\nMid-half", "6\nBH-half",
                         "7\nFH-long", "8\nMid-long", "9\nBH-long"],
                        fontsize=8.5)
    ax.set_xlabel("pointId (★=class 3, focus class, rarest 0.9% in train)")
    ax.set_ylabel("Label distribution mass (for a class-3 sample)")
    ax.set_title("Fig 4. AsymSpatial Focal Loss — encoding 9-zone grid topology\n"
                 "into the loss function so class 3 retains mass at row/column neighbors")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    out = FIG_DIR / "fig4_asym_spatial_loss.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    log(f"✓ {out.relative_to(ROOT)}")


# ============================================================
# Fig 5: V38 failure — per-class action F1 comparison V27 vs V38
# ============================================================
def fig5_v38_failure_perclass():
    v3 = np.load(CACHE_DIR / "oof_test_probs.npz")
    y_a = v3["oof_y_a"]
    v27_oof_a, _, _ = _load_bag("v27")
    v38_oof_a, _, _ = _load_bag("v38")

    v27_pred = v27_oof_a.argmax(1)
    v38_pred = v38_oof_a.argmax(1)
    f1_v27 = f1_score(y_a, v27_pred, average=None, zero_division=0)[:15]
    f1_v38 = f1_score(y_a, v38_pred, average=None, zero_division=0)[:15]
    diff = f1_v38 - f1_v27

    classes = np.arange(15)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["#43A047" if d > 0 else "#E53935" for d in diff]
    bars = ax.bar(classes, diff, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.6)
    for bar, d in zip(bars, diff):
        ypos = d + (0.003 if d > 0 else -0.005)
        ax.text(bar.get_x() + bar.get_width() / 2, ypos, f"{d:+.3f}",
                ha="center", fontsize=8.5)
    ax.set_xticks(classes)
    ax.set_xticklabels([str(c) for c in classes])
    ax.set_xlabel("actionId (class 0-14)")
    ax.set_ylabel("Per-class F1 Δ:  V38 − V27 (OOF)")
    ax.set_title("Fig 5. V38 per-class action F1 deltas vs V27 (OOF)\n"
                 "Even though aggregate F1_a is higher (+0.0087), per-class shows V38 trades wins for losses —\n"
                 "this divergent prediction surface is what crashed on 609 NEW-only OOD test rallies (LB −0.0196)")
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    out = FIG_DIR / "fig5_v38_perclass_delta.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    log(f"✓ {out.relative_to(ROOT)}")


# ============================================================
# Fig 6: Consensus inversion — failed models against base, OOF-negative everywhere
# ============================================================
def fig6_consensus_inversion():
    summary_path = CACHE_DIR / "consensus_microflip" / "summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        col = "oof_delta" if "oof_delta" in df.columns else df.columns[2]
        action = df[df["head"] == "action"].sort_values("k") if "head" in df.columns else None
        point = df[df["head"] == "point"].sort_values("k") if "head" in df.columns else None
        both = df[df["head"] == "both"].sort_values("k") if "head" in df.columns else None
        ks = [2, 3, 4]
        a_vals = action[col].tolist() if action is not None else [-0.0083, -0.0030, -0.0008]
        p_vals = point[col].tolist() if point is not None else [-0.0067, -0.0040, -0.0017]
        b_vals = both[col].tolist() if both is not None else [-0.0151, -0.0070, -0.0026]
    else:
        ks = [2, 3, 4]
        a_vals = [-0.0083, -0.0030, -0.0008]
        p_vals = [-0.0067, -0.0040, -0.0017]
        b_vals = [-0.0151, -0.0070, -0.0026]

    x = np.arange(len(ks))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - w, a_vals, w, color="#1976D2", edgecolor="black", linewidth=0.5,
           label="action head")
    ax.bar(x,     p_vals, w, color="#FB8C00", edgecolor="black", linewidth=0.5,
           label="point head")
    ax.bar(x + w, b_vals, w, color="#6A1B9A", edgecolor="black", linewidth=0.5,
           label="both heads")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}\n({k} of 4 models agree)" for k in ks], fontsize=10)
    ax.set_xlabel("Consensus strictness — minimum # failed models that must agree",
                  fontsize=11)
    ax.set_ylabel("Nested OOF Final Δ vs best (in-distribution)", fontsize=11)
    ax.set_title("Fig 6. Consensus inversion — failed models (V25A60/V27-60/V37/V38) as 'error detectors'\n"
                 "flipping toward consensus is OOF-NEGATIVE at every (head, k) — premise is inverted",
                 fontsize=12)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_ylim(-0.018, 0.003)
    ax.annotate("Even unanimous consensus (k=4)\nis OOF-negative\n→ anti-signal, not signal",
                xy=(2, b_vals[2]), xytext=(1.3, -0.014),
                fontsize=10, color="#E53935", fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color="#E53935", lw=1.2))
    plt.tight_layout()
    out = FIG_DIR / "fig6_consensus_inversion.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    log(f"✓ {out.relative_to(ROOT)}")


def generate_all():
    """Generate all 6 figures into docs/figures/."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    log(f"=== Figure generation (output dir: {FIG_DIR.relative_to(ROOT)}/) ===")
    fig1_v38_chain()
    fig2_flip_vs_lb()
    fig3_v25a_vs_v3_per_class()
    fig4_asym_spatial_label()
    fig5_v38_failure_perclass()
    fig6_consensus_inversion()
    log(f"已生成 6 張圖至 {FIG_DIR.relative_to(ROOT)}/")
