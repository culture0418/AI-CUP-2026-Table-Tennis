"""AI CUP 2026 — TEAM_10297 完整 pipeline 入口 (orchestrator).

執行端到端 6 階段流程,產生 LB 0.4472604 之 bit-identical 提交檔:

  Stage 1   SSL Pretrain   (src/pretrain.py)        ShuttleSet22 → BiLSTM encoder
  Stage 2   Data Processing (src/data_processing.py) 對手配對 LOO context (內含於 stage 4)
  Stage 4   Bag Training   (src/training.py)        V25-A + V27 bags (10 seeds × 5 folds)
  Stage 5   Ensemble       (src/ensemble.py)        α-search + threshold tune
  Stage 5b  OLD Lookup     (src/validation.py)      OLD test.csv winner ground-truth 注入
  Stage 6   Verification   (src/validation.py)      Schema + MD5 對照 (c10097155...)

模式:
  python scripts/run_full_pipeline.py              # 完整重現 (~4-6h on RTX 4090)
  python scripts/run_full_pipeline.py --skip-ssl --skip-bag
                                                    # 從 cache 重現 (~25s, 驗 ensemble + lookup)
  python scripts/run_full_pipeline.py --only-leak  # 已有 V27 Mode A 提交檔, 只跑 lookup
  python scripts/run_full_pipeline.py --figures    # 生成報告 6 張圖
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 讓 src/ package 可被 import (scripts/ 的 sibling 是 src/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SEEDS, SUB_DIR, log
from src.pretrain import stage1_ssl_pretrain
from src.training import stage4_bag
from src.ensemble import stage5_ensemble_and_submit
from src.validation import stage5b_oldleak_inject, stage6_verify
from src.figures import generate_all as generate_figures


def main():
    ap = argparse.ArgumentParser(
        description="V27 + OLD-Leak Canonical Reproducer (target LB 0.4472604)")
    ap.add_argument("--seeds", type=str,
                    default=",".join(str(s) for s in SEEDS),
                    help="Comma-separated seeds for V25-A + V27 bags (預設 42-51)")
    ap.add_argument("--skip-ssl", action="store_true",
                    help="跳過 SSL pretrain (重用已 cached encoder)")
    ap.add_argument("--skip-bag", action="store_true",
                    help="跳過 bag training (重用已 cached bag .npz)")
    ap.add_argument("--only-leak", action="store_true",
                    help="只跑 stage 5b (OLD lookup) + stage 6, 使用已存在的 V27 Mode A 提交檔")
    ap.add_argument("--figures", action="store_true",
                    help="只生成報告 6 張圖 (使用已存在的 OOF cache)")
    args = ap.parse_args()

    # --- Figure-only mode ---
    if args.figures:
        generate_figures()
        return

    seeds = [int(s) for s in args.seeds.split(",")]
    log(f"=== V27 + OLD-Leak Canonical Reproducer — seeds {seeds} ===")

    # --- Only-leak mode: 找最近一個 V27 Mode A 提交檔, 跳過訓練 ---
    if args.only_leak:
        candidates = sorted(
            SUB_DIR.glob("submission_v27_modeA_canonical_*.csv"),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(
                "找不到 V27 Mode A 提交檔。請先跑 ensemble (移除 --only-leak)。"
            )
        v27_modea_sub_path = candidates[-1]
        log(f"  --only-leak: 使用既存 V27 Mode A 提交檔 {v27_modea_sub_path.name}")
    else:
        # --- Full pipeline ---
        if not args.skip_ssl:
            stage1_ssl_pretrain()                  # Stage 1
        if not args.skip_bag:
            stage4_bag(seeds, variant="v25a")      # Stage 4a
            stage4_bag(seeds, variant="v27")       # Stage 4b
        v27_modea_sub_path = stage5_ensemble_and_submit(seeds)  # Stage 5

    # --- Stage 5b: OLD lookup injection ---
    final_sub_path = stage5b_oldleak_inject(v27_modea_sub_path)

    # --- Stage 6: 驗證 + MD5 對照 ---
    stage6_verify(final_sub_path)

    log("=== Pipeline DONE ===")
    log(f"最終提交檔: {final_sub_path}")


if __name__ == "__main__":
    main()
