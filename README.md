# AI CUP 2026 春季賽 — TEAM_10297

> 基於時序資料之桌球戰術與結果預測競賽

## 🏆 Key Results

| 指標 | 數值 |
|---|---|
| **Public Leaderboard 分數** | **0.4472604** |
| **Public Rank** | **24 / 365** |
| **Private Leaderboard 分數** | **0.3682964** |
| **Private Rank** | **32 / 365** |
| **OOF Final**(集成前) | 0.3869 (F1_a 0.4269 / F1_p 0.2303 / AUC 0.6200) |
| **LB 累積提升** | V3 baseline 0.3649 → **0.4472604** (+0.0823) |

### LB 進展時間軸

| 日期 | 版本 | LB | Δ | 關鍵新機制 |
|---|---|---|---|---|
| 2026-05-05 | V3 baseline | 0.3649 | — | LSTM + XGB + Cat + FTT cascade |
| 2026-05-08 | v12 | 0.3702 | +0.0053 | ShuttleSet22 跨運動 SSL pretrain |
| 2026-05-09 | v17 | 0.3747 | +0.0046 | Asym loss + transductive aug |
| 2026-05-19 | V25-A | 0.3757 | +0.0010 | 58 維對手配對 LOO ctx |
| 2026-05-20 | V27 Mode A | 0.3787 | +0.0030 | AsymSpatial loss (class 3 空間平滑) |
| **2026-05-25** | **v27_oldleak** ⭐ | **0.4472604** | **+0.0686** | **OLD test.csv winner lookup**(主辦核可) |

### 核心創新點(七項)

1. **跨運動 SSL 遷移** — 羽球(ShuttleSet22)→ 桌球 MLM pretrain,LB +0.0044
2. **對手配對 LOO Context** — 58 維 ego/opp 戰術歷史,確定性非參數,leak-free
3. **AsymSpatial Focal Loss** — 把桌球 9 宮格空間拓撲編碼進損失函數
4. **V27 Mode A 雙取代集成** — 同架構雙 loss objective diversity
5. **Transductive Augmentation** — test rallies (T≥2) 入訓練集 (OOF +0.0004 → LB +0.0032, transfer 8x)
6. **Winner head α-search 優化** — V3-Cat + v1 SSL-LSTM 融合
7. **主辦核可外部資料合規利用** — OLD test.csv 1,236 winner ground truth lookup

---

**Entry point**: `scripts/run_full_pipeline.py` — 單一指令端到端執行 6 階段流程
**對應報告**: [`docs/aicup2026_report.md`](docs/aicup2026_report.md)
**程式碼結構**: `src/` 8 個模組(config, data_processing, models, losses, pretrain, training, ensemble, validation, figures)按功能拆分

---

## 目錄

- [運行環境](#運行環境)
- [資料](#資料)
- [檔案結構](#檔案結構)
- [快速重現(模式 B,~25 秒)](#快速重現模式-b25-秒)
- [完整從零重現(模式 A,~4-6 小時)](#完整從零重現模式-a4-6-小時)
- [產生報告圖](#產生報告圖)
- [src/ 模組介紹](#src-模組介紹)
- [重現驗證](#重現驗證)
- [Troubleshooting](#troubleshooting)
- [競賽合規說明](#競賽合規說明)

---

## 運行環境

| 項目 | 版本 / 規格 |
|---|---|
| 作業系統 | Linux (Ubuntu 22.04 以上) |
| Python | 3.12 (3.10+ 應可) |
| GPU | NVIDIA GPU,CUDA 11.8+ (測試於 RTX 4090 / CUDA 13.0) |
| GPU 記憶體 | ≥ 8 GB |
| 系統 RAM | ≥ 16 GB |
| 磁碟空間 | ≥ 10 GB (含 cache + submissions) |

### 安裝步驟

```bash
# 1. Clone repo (含全部程式碼、資料、cache, ~310 MB)
git clone https://github.com/culture0418/AI-CUP-2026-Table-Tennis.git
cd AI-CUP-2026-Table-Tennis

# 2. 建立 Python 虛擬環境
python3 -m venv venv
source venv/bin/activate

# 3. 安裝相依套件
pip install --upgrade pip
pip install -r requirements.txt

# 4. 確認 PyTorch 可用 GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

CPU-only 模式可運作但 bag training 將大幅變慢(預估 > 24 小時)。

---

## 資料

本繳交包**已內附全部所需資料**(總大小 ~8 MB):

```
data/
├── train.csv                          主辦提供:訓練集 (14,995 rallies)
├── test_new.csv                       主辦提供:測試集 (1,845 rallies, 無 serverGetPoint)
├── test.csv                           主辦提供:Reference_Only_Old_Test_Data (1,236 rallies 含 serverGetPoint)
├── dataset_description.md             欄位定義
└── external/shuttleset22/train.csv    外部:ShuttleSet22 羽球資料
```

### 資料 MD5 完整性

```
3aa1f55b6178461c870bb138c59a3b68  data/train.csv
8c49a01f8428291623efc33c2e8df399  data/test_new.csv
6008387d786c9f3b29d2a3f9d039ecbb  data/test.csv
37e7e4fb0e087d0e1581d67ad4d0f947  data/external/shuttleset22/train.csv
```

---

## 檔案結構

```
aicup2026_deliverable/
├── README.md                                       本檔
├── requirements.txt                                Python 套件版本
├── .gitignore
│
├── scripts/
│   └── run_full_pipeline.py                       ⭐ 唯一入口 (orchestrator, ~85 行)
│
├── src/                                            核心模組(按功能拆分)
│   ├── config.py                                  配置(路徑、超參數、常數)
│   ├── data_processing.py                         資料處理 — 對手配對 LOO context
│   ├── models.py                                  模型架構 — SSLEncoderLSTM + TTSSLLSTMHier
│   ├── losses.py                                  損失函數 — FocalLoss + AsymSpatialFocalLoss
│   ├── pretrain.py                                Stage 1 — ShuttleSet22 SSL pretrain
│   ├── training.py                                Stage 4 — V25-A + V27 bag training
│   ├── ensemble.py                                Stage 5 — α-search + threshold tune
│   ├── validation.py                              Stage 5b + 6 — OLD lookup + verify
│   └── figures.py                                 6 張報告圖生成
│
├── data/                                           ⭐ 全部資料已內附(~8 MB)
│   ├── dataset_description.md
│   ├── train.csv / test_new.csv / test.csv
│   └── external/shuttleset22/train.csv
│
├── cache/                                          ⭐ 全部 cache 已內附(~301 MB, 52 檔)
│   ├── ssl_lstm_encoder_shuttleset22.pt           SSL encoder
│   ├── oof_test_v25a*.npz                         V25-A bag (10 seeds)
│   ├── oof_test_v27*.npz                          V27 bag (10 seeds)
│   ├── oof_test_v38*.npz                          V38 bag (dead-end, for figures)
│   ├── oof_test_tt_shuttlenet*.npz                v1/asym bags (10 seeds each)
│   ├── oof_test_probs.npz                         V3 baseline cascade
│   └── consensus_microflip/summary.csv            diagnostic summary for fig 6
│
├── submissions/
│   └── submission_v27_oldleak_20260525_0103.csv   ⭐ 官方 LB 提交檔 (MD5 對照)
│
└── docs/
    ├── aicup2026_report.md                        ⭐ 競賽報告 (中文 ~4000 字)
    └── figures/                                    報告陸段引用的 6 張圖
        ├── fig1_v38_chain.png
        ├── fig2_flip_vs_lb.png
        ├── fig3_v25a_vs_v3_per_class.png
        ├── fig4_asym_spatial_loss.png
        ├── fig5_v38_perclass_delta.png
        └── fig6_consensus_inversion.png
```

---

## 快速重現(模式 B,~25 秒)

從內附 cache 重現,跳過 SSL pretrain + bag training,只跑 ensemble + lookup:

```bash
python scripts/run_full_pipeline.py --skip-ssl --skip-bag
```

**預期輸出最後幾行**:
```
[XX:XX:XX]   提交檔 MD5: c10097155c0942354f81ea188b43f111
[XX:XX:XX]   預期   MD5: c10097155c0942354f81ea188b43f111
[XX:XX:XX]   ✓ MATCH — 與 LB 0.4472604 提交檔逐位元相同
```

這是審查者**最快驗證系統正確性**的方式。

---

## 完整從零重現(模式 A,~4-6 小時)

從 raw data 開始,執行 SSL pretrain + 10-seed bag training + ensemble + OLD lookup + MD5 驗證:

```bash
python scripts/run_full_pipeline.py
```

**預估時間** (NVIDIA RTX 4090):

| Stage | 動作 | 耗時 |
|---|---|---|
| 1 | SSL pretrain on ShuttleSet22 (`src/pretrain.py`) | ~10 min |
| 4a | V25-A bag (10 seeds × 5 folds × 30 ep, FocalLoss, `src/training.py`) | ~30 min |
| 4b | V27 bag (同規模, AsymSpatialFocalLoss, `src/training.py`) | ~30 min |
| 5 | Ensemble α-search + threshold (`src/ensemble.py`) | ~1 min |
| 5b | OLD test.csv winner lookup (`src/validation.py`) | <1 sec |
| 6 | Schema + MD5 verification (`src/validation.py`) | <1 sec |

> **重要**:Mode A 完整重現依賴 PyTorch 隨機性在你機器上的可預測性。若 GPU / CUDA 版本與我們的 RTX 4090 / CUDA 13.0 不同,bag training 結果可能無法完全 bit-identical。**Mode B**(從內附 cache)在所有硬體上保證 MD5 完全一致。

---

## 產生報告圖

```bash
python scripts/run_full_pipeline.py --figures
```

生成報告陸段引用的 6 張圖至 `docs/figures/`(從 cache 讀數據,只需幾秒)。

---

## src/ 模組介紹

### `src/config.py`

集中管理路徑、模型超參數、資料維度、SSL/finetune 超參數、AsymSpatialFocalLoss 焦點區常數、device / log helper。所有其他模組統一從這裡 import。

### `src/data_processing.py`

**對應規範「資料處理」**

- `compute_oppair_contexts(combined_df, k_per_rally)` — 58 維對手配對 LOO context 計算,V25-A 創新核心。嚴格 leakage-free(LOO 排除當前 rally)。

### `src/models.py`

**模型架構**

- `SSLEncoderLSTM` — ShuttleSet22 MLM pretrain 用的 BiLSTM encoder。
- `TTSSLLSTMHier` — V25-A / V27 共享 backbone(BiLSTM + 58-dim opp-pair ctx + 3 heads,~0.34M 參數)。

### `src/losses.py`

**損失函數**

- `FocalLoss` — V25-A 用(γ=2, uniform label smoothing 0.10)。
- `AsymSpatialFocalLoss` — V27 創新。對 class 3 做空間鄰居(class 2, 6)非對稱平滑,將桌球 9 宮格戰術知識編碼進損失。

### `src/pretrain.py`(Stage 1)

**對應規範「訓練流程 — pretrain」**

- `stage1_ssl_pretrain()` — ShuttleSet22 MLM pretrain(30 epochs, batch 64, lr 1e-3)。
- **輸入**: `data/external/shuttleset22/train.csv`
- **輸出**: `cache/ssl_lstm_encoder_shuttleset22.pt`(~0.34 MB)
- **跳過旗標**: `--skip-ssl`

### `src/training.py`(Stage 4)

**對應規範「訓練流程 — training」**

- `stage4_bag(seeds, variant)` — 訓練 V25-A 或 V27 bag(10 seeds × 5 folds × 30 epochs)。
- `stage4_bag_one_seed(seed, variant)` — 單一 seed 訓練(內含 sample_k、encode_df、build_rallies、DS、collate、class_w_sqrt 等 closure 邏輯)。
- **輸入**: `data/train.csv`, `data/test_new.csv`, SSL encoder cache
- **輸出**: `cache/oof_test_{variant}{_seedN}.npz`(per-seed OOF + test probabilities)
- **跳過旗標**: `--skip-bag`

### `src/ensemble.py`(Stage 5)

**對應規範「預測 — ensemble」**

- `stage5_ensemble_and_submit(seeds)` — V27 Mode A 集成(7-way action / 8-way point / 4-way winner α-search + cap=0.75 threshold tune)。
- `search_grid`, `coord_descent`, `tune_thresh` — α-search 演算法。
- `load_bag`, `load_v25a_bag`, `load_v27_bag` — bag cache loader。
- **輸出**: `submissions/submission_v27_modeA_canonical_{ts}.csv`(LB 0.3787 base)

### `src/validation.py`(Stage 5b + 6)

**對應規範「預測 — final + validation」**

- `stage5b_oldleak_inject(v27_modea_sub_path)` — 對 1,236 個重疊 rally 注入 OLD test.csv `serverGetPoint` ground truth。
- `stage6_verify(sub_path)` — Schema + MD5 對照(EXPECTED_MD5 = `c10097155c...`)。
- **輸出**: `submissions/submission_v27_oldleak_{ts}.csv`(LB 0.4472604)

### `src/figures.py`

**報告圖生成**

- `generate_all()` — 生成 6 張報告陸段引用的圖至 `docs/figures/`。
- 從 `cache/` 內附 OOF 數據自動產出,審查者可一鍵重生。

---

## 重現驗證

完整跑完 Mode A 或 Mode B,`stage6_verify` 自動印出 MD5 對照:

```
[XX:XX:XX]   提交檔 MD5: c10097155c0942354f81ea188b43f111
[XX:XX:XX]   預期   MD5: c10097155c0942354f81ea188b43f111
[XX:XX:XX]   ✓ MATCH — 與 LB 0.4472604 提交檔逐位元相同
```

亦可手動比對:

```bash
md5sum submissions/submission_v27_oldleak_*.csv submissions/submission_v27_oldleak_20260525_0103.csv
```

---

## Troubleshooting

### Q1. `FileNotFoundError: data/external/shuttleset22/train.csv`

確認 ShuttleSet22 已下載並放置於正確路徑(已內附於繳交包,正常不會發生)。

### Q2. `RuntimeError: CUDA out of memory`

降低 batch size。編輯 `src/config.py` 中 `FT_BS = 64` → `FT_BS = 32`。注意降低 batch size 可能影響最終 MD5 重現(改用 Mode B 從 cache 驗證即可)。

### Q3. Stage 6 MD5 顯示 ⚠️ DIFFERS

- **硬體差異**: 不同 GPU / CUDA 版本下 PyTorch 隨機性可能不完全一致 → 改用 Mode B(`--skip-ssl --skip-bag`)從 cache 驗證。
- **資料版本不同**: 確認 `data/test.csv` 是主辦 2026-05-21 公告的版本(MD5 應為 `6008387d...`)。

### Q4. PyTorch 警告 `weights_only=False`

PyTorch 2.4+ 新警告,不影響功能,可忽略。

---

## 競賽合規說明

### 外部資料使用揭露

本系統使用 **2 項外部資料**,皆符合競賽規則:

1. **ShuttleSet22**: 公開的羽球(非桌球)資料集,用於跨運動 SSL 預訓練。屬公開研究資料,非反查 test。
   - 來源: [CoachAI-Projects](https://github.com/wywyWang/CoachAI-Projects) (Wang et al. 2023)

2. **`data/test.csv`** (Reference_Only_Old_Test_Data): 主辦單位 2026-05-21 公告開放當訓練資料,含 1,236 / 1,845 test rallies 的 `serverGetPoint` ground truth。以 lookup 注入方式利用,屬主辦明確核可的使用方式。

### 生成式 AI 工具揭露

本隊伍於開發過程中使用 **Anthropic Claude (Opus 4.7)** 與 **OpenAI Codex** 作為程式輔助與分析協作工具。所有架構設計、實驗方向、最終提交,皆由人類隊員審查決策。詳見 [`docs/aicup2026_report.md`](docs/aicup2026_report.md) 壹段揭露說明。

### 繳交內容

本繳交包含:
- 重現 LB 0.4472604 必要的全部程式碼(`src/` + `scripts/`)
- README.md(本檔)+ requirements.txt
- 全部資料檔(`data/`, ~8 MB)
- 全部 cache 檔(`cache/`, ~301 MB, 52 檔, 用於 25 秒快速驗證)
- 參考 submission CSV(`submissions/submission_v27_oldleak_20260525_0103.csv`)
- 競賽報告 Markdown + 6 張報告圖

---

**TEAM_10297** — 如審查過程有任何問題,請透過競賽平台聯繫。

_本繳交包對應之完整方法論、創新性、實驗分析: [`docs/aicup2026_report.md`](docs/aicup2026_report.md)_
