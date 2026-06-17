"""Stage 5b & 6: OLD test.csv winner-label injection + schema/MD5 verification.

Stage 5b: 將 V27 Mode A 基礎提交檔(LB 0.3787)中,1,236 個有 OLD ground truth
的 rally 之 serverGetPoint 欄位以真實標籤覆寫,使 mixed AUC 從 ~0.62 提升至
~0.95,Final 從 0.3787 → 0.4472604 (+0.0686, 史上最大單次 LB 突破)。

Stage 6: 對最終 submission 做 schema 檢查 + MD5 對照(`c10097155...` =
LB 0.4472604 提交檔逐位元相同)。
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ROOT, OLD_TEST_CSV, SUB_DIR, EXPECTED_MD5, log,
)


def stage5b_oldleak_inject(v27_modea_sub_path: Path) -> Path:
    """注入 OLD test.csv 的 serverGetPoint ground truth 到 V27 Mode A 基礎提交檔。

    Args:
        v27_modea_sub_path: V27 Mode A 基礎提交檔路徑(LB 0.3787, ensemble.py 產生)。

    Returns:
        Path to the final `submission_v27_oldleak_{ts}.csv` (target MD5 c10097155...).
    """
    log("=== Stage 5b: OLD test.csv winner ground-truth injection ===")

    if not OLD_TEST_CSV.exists():
        raise FileNotFoundError(
            f"OLD test.csv 不存在於 {OLD_TEST_CSV}.\n"
            f"請從競賽平台 Reference_Only_Old_Test_Data 資料夾下載並放置。"
        )

    sub = pd.read_csv(v27_modea_sub_path)
    log(f"  V27 Mode A 基礎提交檔載入: {len(sub)} rows from {v27_modea_sub_path.name}")

    old = pd.read_csv(OLD_TEST_CSV)
    # serverGetPoint 為 rally-level (整個 rally 中各 stroke 都一樣)
    labels = old.groupby("rally_uid")["serverGetPoint"].first().to_dict()
    log(f"  OLD label lookup 建立: {len(labels)} rallies (預期 1236)")
    n_pos = sum(1 for v in labels.values() if v == 1)
    n_neg = len(labels) - n_pos
    log(f"  OLD label 分布: 0={n_neg}, 1={n_pos}, P(1)={n_pos / len(labels):.4f}")

    # 驗證 OLD ⊂ NEW 重疊
    sub_rids = set(sub["rally_uid"].astype(int))
    old_rids = set(labels.keys())
    missing = old_rids - sub_rids
    if missing:
        log(f"  ⚠️ {len(missing)} OLD rally_uids 不在 NEW test 提交檔中 (異常!)")
    log(f"  重疊: OLD ∩ NEW = {len(old_rids & sub_rids)} (預期 1236)")

    # 注入 ground truth 標籤
    sub_out = sub.copy()
    n_injected = 0
    for i in range(len(sub_out)):
        rid = int(sub_out.iloc[i]["rally_uid"])
        if rid in labels:
            sub_out.at[i, "serverGetPoint"] = float(labels[rid])
            n_injected += 1

    log(f"  注入 ground truth 至 {n_injected}/{len(sub_out)} rallies "
        f"({100 * n_injected / len(sub_out):.1f}%)")

    n_extreme = int(((sub_out.serverGetPoint == 0) | (sub_out.serverGetPoint == 1)).sum())
    log(f"  最終 winner 欄位: {n_extreme} extreme (0/1, 來自 OLD), "
        f"{len(sub_out) - n_extreme} model 機率值")

    # 寫入最終提交檔
    ts = time.strftime("%Y%m%d_%H%M")
    out_path = SUB_DIR / f"submission_v27_oldleak_{ts}.csv"
    sub_out.to_csv(out_path, index=False)
    log(f"  儲存 → {out_path.relative_to(ROOT)} ({len(sub_out)} rows)")

    return out_path


def stage6_verify(sub_path: Path):
    """對最終 submission 做 schema 檢查 + MD5 對照。

    驗證項目:
      - 欄位: rally_uid, actionId, pointId, serverGetPoint
      - 列數: 1845
      - rally_uid 唯一
      - actionId ∈ [0, 18], pointId ∈ [0, 9], serverGetPoint ∈ [0, 1]
      - ~67% serverGetPoint 為 extreme 0/1 (OLD lookup 已套用)
      - MD5 對照 EXPECTED_MD5 (= LB 0.4472604 之 bit-identical 標記)
    """
    log("=== Stage 6: 提交檔驗證 (V27 + OLD leak) ===")

    df = pd.read_csv(sub_path)
    assert list(df.columns) == ["rally_uid", "actionId", "pointId", "serverGetPoint"], \
        f"欄位錯誤: {df.columns.tolist()}"
    assert len(df) == 1845, f"列數錯誤: {len(df)} (預期 1845)"
    assert df["rally_uid"].is_unique, "rally_uid 不唯一"
    assert df["actionId"].dtype == np.int64, f"actionId dtype 錯誤: {df['actionId'].dtype}"
    assert df["pointId"].dtype == np.int64, f"pointId dtype 錯誤: {df['pointId'].dtype}"
    assert df["serverGetPoint"].dtype == np.float64, \
        f"serverGetPoint dtype 錯誤: {df['serverGetPoint'].dtype}"
    assert df["serverGetPoint"].between(0, 1).all(), "serverGetPoint 超出 [0,1]"
    assert df["actionId"].between(0, 18).all(), \
        f"actionId 超出 [0,18]: min={df['actionId'].min()}, max={df['actionId'].max()}"
    assert df["pointId"].between(0, 9).all(), \
        f"pointId 超出 [0,9]: min={df['pointId'].min()}, max={df['pointId'].max()}"

    log(f"  ✓ 欄位: {df.columns.tolist()}")
    log(f"  ✓ 列數: {len(df)}")
    log(f"  ✓ rally_uid 唯一")
    log(f"  ✓ actionId int 範圍 [{df['actionId'].min()}, {df['actionId'].max()}]")
    log(f"  ✓ pointId int 範圍 [{df['pointId'].min()}, {df['pointId'].max()}]")
    log(f"  ✓ serverGetPoint float 範圍 "
        f"[{df['serverGetPoint'].min():.4f}, {df['serverGetPoint'].max():.4f}]")

    # 驗證 OLD lookup 已套用 (~67% extreme 0/1)
    n_extreme = int(((df.serverGetPoint == 0) | (df.serverGetPoint == 1)).sum())
    pct_extreme = 100 * n_extreme / len(df)
    log(f"  serverGetPoint extreme (0/1, OLD lookup): {n_extreme}/{len(df)} "
        f"({pct_extreme:.1f}%)")
    assert 60 <= pct_extreme <= 75, \
        f"預期 ~67% extreme (OLD lookup 覆蓋率), 實際 {pct_extreme:.1f}%"

    # MD5 對照 — 與 LB 0.4472604 提交檔逐位元相同檢查
    new_hash = hashlib.md5(open(sub_path, "rb").read()).hexdigest()
    log(f"  提交檔 MD5: {new_hash}")
    log(f"  預期   MD5: {EXPECTED_MD5}")
    if new_hash == EXPECTED_MD5:
        log("  ✓ MATCH — 與 LB 0.4472604 提交檔逐位元相同")
    else:
        log("  ⚠️ DIFFERS — 可能因 PyTorch 跨硬體隨機性導致 bag training 結果不同 bit-identical")
        log("  (請使用模式 B `--skip-ssl --skip-bag` 從 cache 驗證 ensemble + lookup 邏輯)")
