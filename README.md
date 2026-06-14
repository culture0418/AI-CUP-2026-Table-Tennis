# AI CUP 2026 春季賽 — TEAM_10297

**競賽**: 基於時序資料之桌球戰術與結果預測競賽
**Public LB**: 0.4472604 (Rank 15 / 365, 截至 2026-05-29)
**Submission MD5**: `c10097155c0942354f81ea188b43f111`
**Mode B Canonical Reproducer**: `scripts/v27_oldleak_full_pipeline.py`

本繳交包含完整重現公開排行榜 0.4472604 提交檔所需的全部程式碼、依賴版本與重現指令。對應之競賽報告請見 [`docs/aicup2026_report.md`](docs/aicup2026_report.md)。

---

## 目錄

- [運行環境](#運行環境)
- [資料](#資料)
- [處理腳本](#處理腳本)
- [訓練](#訓練)
- [預測](#預測)
- [重現驗證](#重現驗證)
- [重要模組介紹](#重要模組介紹)
- [模型權重 / Cache](#模型權重--cache)
- [檔案結構](#檔案結構)
- [Troubleshooting](#troubleshooting)
- [競賽合規說明](#競賽合規說明)

---

## 運行環境

| 項目 | 版本 / 規格 |
|---|---|
| 作業系統 | Linux (Ubuntu 22.04 同等以上) |
| Python | 3.12 (3.10+ 應可) |
| GPU | NVIDIA GPU,CUDA 11.8+ (測試於 RTX 4090 / CUDA 13.0) |
| GPU 記憶體 | ≥ 8 GB |
| 系統 RAM | ≥ 16 GB |
| 磁碟空間 | ≥ 10 GB (含 cache) |

### 安裝步驟

```bash
# 1. 解壓縮並進入目錄
unzip aicup2026_deliverable.zip
cd aicup2026_deliverable

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

本系統使用 **3 個競賽提供資料檔** + **1 個競賽核可外部資料集**。請將檔案放置如下:

```
data/
├── train.csv                          ← 主辦提供:訓練集 (14,995 rallies)
├── test_new.csv                       ← 主辦提供:測試集 (1,845 rallies, 無 serverGetPoint)
├── test.csv                           ← 主辦提供:Reference_Only_Old_Test_Data (1,236 rallies 含 serverGetPoint)
└── external/
    └── shuttleset22/
        └── train.csv                  ← 外部:ShuttleSet22 羽球資料(SSL pretrain 用)
```

`dataset_description.md` 內含各欄位定義。

### 外部資料下載

**ShuttleSet22** (CoachAI Projects, Wang et al. 2023):

```bash
git clone https://github.com/wywyWang/CoachAI-Projects.git
# 將其中 ShuttleSet22 的 train.csv 複製到 data/external/shuttleset22/
```

或直接到 [`https://github.com/wywyWang/CoachAI-Projects`](https://github.com/wywyWang/CoachAI-Projects) 下載對應 release。

### 資料完整性檢查

```bash
ls -la data/train.csv data/test_new.csv data/test.csv data/external/shuttleset22/train.csv
# 應有 4 個檔案
```

---

## 處理腳本

本系統採**單一 orchestrator + 單一 hub** 的設計,所有資料前處理、特徵工程、context 計算邏輯都封裝於 hub 內。第三方無需執行獨立的前處理腳本——`scripts/v27_oldleak_full_pipeline.py` 會自動依序呼叫:

| 處理階段 | 實作位置 (`v27_modeA_full_pipeline.py`) | 功能 |
|---|---|---|
| 特徵編碼 | `stage4_bag_one_seed` 內 `encode_df()` | 13 個 categorical features → embedding token |
| 序列構建 | `build_rallies()` | 將 stroke-level data 聚合為 rally-level sequence |
| K-truncation 取樣 | `sample_k()` | test-distribution-aware k 採樣 |
| 對手配對 context | `compute_oppair_contexts()` | 58 維 ego/opp 球員 LOO 統計 |
| Transductive aug | `stage4_bag_one_seed` 內邏輯 | test rallies (T≥2) 加入訓練集 |
| OLD 標籤注入 | `stage5b_oldleak_inject()` | 1236 rally winner ground truth lookup |

若需單獨檢驗 context 計算結果,可在 Python REPL 內:

```python
import sys; sys.path.insert(0, 'scripts')
from v27_modeA_full_pipeline import compute_oppair_contexts
# 詳見「重要模組介紹」段
```

---

## 訓練

### 完整訓練(從零)

```bash
python scripts/v27_oldleak_full_pipeline.py
```

**預估時間: 4-6 小時** (NVIDIA RTX 4090)。執行階段:

| Stage | 動作 | 耗時 |
|---|---|---|
| 1 | SSL pretrain on ShuttleSet22 → BiLSTM encoder | ~10 min |
| 4a | V25-A bag (10 seeds × 5 folds × 30 epochs, FocalLoss) | ~30 min |
| 4b | V27 bag (10 seeds × 5 folds × 30 epochs, AsymSpatialFocalLoss) | ~30 min |
| 5 | V27 Mode A ensemble α-search + threshold | ~1 min |
| 5b | OLD test.csv winner lookup injection | <1 sec |
| 6 | Schema + MD5 verification | <1 sec |

### 已有 SSL encoder + bag cache 的快速重現

```bash
python scripts/v27_oldleak_full_pipeline.py --skip-ssl --skip-bag
```

**預估時間: ~25 秒**。僅執行 Stage 5、5b、6。用於驗證 ensemble 與 lookup 邏輯是否正確產生 MD5 對得上的提交檔。

### 訓練超參數

完整列於 `scripts/v27_modeA_full_pipeline.py` 頂部 Constants 區段(line 78-115)。關鍵:

```
SSL pretrain:    30 epochs, batch=64, lr=1e-3, MLM mask_prob=0.15
Finetune:        30 epochs, batch=64, lr=1e-3, weight_decay=1e-5
Loss weights:    0.4 × action + 0.4 × point + 0.2 × winner
Focal γ:         2.0
Label smoothing: 0.10
Player masking:  p=0.30
Grad clip:       1.0
Seeds:           [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
Folds:           5 (GroupKFold by match)
Threshold cap:   0.75
```

---

## 預測

預測整合於 orchestrator,跑完訓練後自動產生提交檔:

```
submissions/submission_v27_oldleak_YYYYMMDD_HHMM.csv
```

### 預測流程細節

1. **Stage 5**: α-search 對 OOF 找到最佳融合權重(action 7-way, point 8-way, winner 4-way)
2. **Per-class threshold**: 對每個 class 找最佳乘數 (cap=0.75)
3. **Test inference**: 將相同 α 與 threshold 套用至 test 機率,取 argmax 得到 actionId / pointId,winner 取機率本身
4. **Stage 5b**: 對 1,236 個 OLD-overlap rally,以 `data/test.csv` 中的 `serverGetPoint` ground truth 覆寫 winner
5. **Stage 6**: 驗證 schema、MD5 與 LB 0.4472604 提交檔逐位元相同

### 只想跑 OLD lookup(已有 V27 Mode A 提交)

若已透過 `scripts/v27_modeA_full_pipeline.py` 產生 V27 Mode A 提交檔,可直接執行 lookup:

```bash
python scripts/v27_oldleak_full_pipeline.py --only-leak
```

---

## 重現驗證

完整跑完訓練後,Stage 6 會自動印出 MD5 比對:

```
[XX:XX:XX]   Submission MD5: c10097155c0942354f81ea188b43f111
[XX:XX:XX]   Expected MD5:   c10097155c0942354f81ea188b43f111
[XX:XX:XX]   ✓ MATCH — bit-identical to LB-submitted file (LB 0.4472604)
```

亦可手動比對:

```bash
md5sum submissions/submission_v27_oldleak_*.csv submissions/submission_v27_oldleak_20260525_0103.csv
# 兩者應為:c10097155c0942354f81ea188b43f111
```

> **硬體可重現性說明**: Mode B 完整重現依賴 PyTorch 隨機性的可預測性。若硬體 / CUDA 版本與我們的 RTX 4090 / CUDA 13.0 不同,bag training 的隨機性可能無法完全 bit-identical(PyTorch 已知特性)。**模式 B**(使用我們提供的 cache 跳過訓練)在所有硬體上可保證 MD5 完全一致——這是我們建議審查者使用的驗證路徑。如需 cache 檔案,請參考下節「模型權重 / Cache」。

---

## 重要模組介紹

核心 hub `scripts/v27_modeA_full_pipeline.py`(906 行)按 Stage 內部分區,每個模組的輸入 / 輸出 / 副作用如下:

### `SSLEncoderLSTM` (class, line 123)

| | |
|---|---|
| **功能** | 用於 ShuttleSet22 MLM pretrain 的 BiLSTM encoder |
| **架構** | 4 features × 32 embed → 128 proj → 1-layer BiLSTM hidden=128 → 2 MLM heads |
| **參數量** | ~0.34M |
| **輸出檔** | `cache/ssl_lstm_encoder_shuttleset22.pt` |

### `stage1_ssl_pretrain()` (function, line 152)

| | |
|---|---|
| **輸入** | `data/external/shuttleset22/train.csv` |
| **輸出** | `cache/ssl_lstm_encoder_shuttleset22.pt` |
| **跳過旗標** | `--skip-ssl` |
| **若已有 cache** | 函式內部會偵測並 skip |

### `compute_oppair_contexts(combined_df, k_per_rally)` (function, line 256)

| | |
|---|---|
| **輸入** | combined train+test DataFrame,k_per_rally dict |
| **輸出** | `dict {rally_uid: np.array(58,)}` — ego_pt(10) + ego_act(19) + opp_pt(10) + opp_act(19) |
| **特性** | LOO (Leave-One-Out) 嚴格 leakage-free |

### `TTSSLLSTMHier` (class, line 323)

| | |
|---|---|
| **功能** | 主序列模型 — BiLSTM + 58-dim opp-pair ctx + 3 heads |
| **架構** | 13 features × 32 embed → 128 proj → BiLSTM h=128 → 256 + 58 ctx = 314 → action/point/winner heads |
| **參數量** | ~0.34M |
| **使用** | V25-A 與 V27 兩 variant 共享同一個 class |

### `FocalLoss` (class, line 361) / `AsymSpatialFocalLoss` (class, line 396)

| | |
|---|---|
| **FocalLoss** | V25-A action+point head 使用 (γ=2, label smoothing 0.1) |
| **AsymSpatialFocalLoss** | V27 point head 使用 — 對 class 3 (反手短) 做空間鄰居 (class 2, 6) label smoothing |

### `stage4_bag(seeds, variant)` (function, line 630)

| | |
|---|---|
| **輸入** | `data/train.csv`, `data/test_new.csv`, SSL encoder cache |
| **輸出** | per-seed `cache/oof_test_{variant}{_seedN}.npz`(含 OOF + test probabilities) |
| **參數** | `seeds=[42..51], variant∈{'v25a','v27'}` |
| **跳過旗標** | `--skip-bag` |

### `stage5_ensemble_and_submit(seeds)` (function, line 742)

| | |
|---|---|
| **輸入** | 全部 bag .npz + `cache/oof_test_probs.npz` (V3 baseline cascade) |
| **輸出** | `submissions/submission_v27_modeA_canonical_{ts}.csv` (LB 0.3787 base) |
| **內部步驟** | (a) 7-way action α-search (b) 8-way point α-search (c) 4-way winner α-search (d) per-class threshold tune (cap=0.75) |

### `stage5b_oldleak_inject(v27_modea_sub_path)` (function, line 67 of `v27_oldleak_full_pipeline.py`)

| | |
|---|---|
| **輸入** | V27 Mode A submission CSV 路徑,`data/test.csv` |
| **輸出** | `submissions/submission_v27_oldleak_{ts}.csv` (1845 rows, 1236 winners 注入 ground truth) |
| **注入率** | 1236 / 1845 = 67% |

### `stage6_verify_oldleak(sub_path)` (function, line 128 of `v27_oldleak_full_pipeline.py`)

| | |
|---|---|
| **檢查項** | 欄位、列數=1845、rally_uid unique、值域、~67% extreme winner、MD5 |
| **預期 MD5** | `c10097155c0942354f81ea188b43f111` |

---

## 模型權重 / Cache

完整 cache 約 600 MB,包含:

- `cache/ssl_lstm_encoder_shuttleset22.pt` — SSL pretrained BiLSTM encoder
- `cache/oof_test_v25a*.npz` — V25-A 10-seed bag (OOF + test probabilities)
- `cache/oof_test_v27*.npz` — V27 10-seed bag
- `cache/oof_test_probs.npz` — V3 baseline cascade

若需直接驗證 ensemble + lookup 邏輯(模式 B,~25 秒)而不執行 4-6 小時完整訓練,請聯絡隊伍取得 cache 壓縮檔。

> Cache 檔案因尺寸過大未直接內附於本繳交包。所有 cache 皆可透過完整訓練(模式 A)從零重新生成。

---

## 檔案結構

```
aicup2026_deliverable/
├── README.md                                       本檔案
├── requirements.txt                                Python 套件版本
├── .gitignore                                      git 排除規則
│
├── scripts/                                        ⭐ 核心程式碼(僅 2 檔)
│   ├── v27_oldleak_full_pipeline.py              Entry point — orchestrator (225 行)
│   └── v27_modeA_full_pipeline.py                Core hub — 全部 stage 實作 (906 行)
│
├── data/                                           資料目錄(使用者放置)
│   ├── dataset_description.md                     欄位定義
│   ├── train.csv                                  ← 主辦提供
│   ├── test_new.csv                               ← 主辦提供
│   ├── test.csv                                   ← 主辦提供 (OLD, 含 serverGetPoint)
│   └── external/shuttleset22/train.csv            ← 外部 ShuttleSet22
│
├── cache/                                          訓練過程自動產生
│   ├── ssl_lstm_encoder_shuttleset22.pt
│   ├── oof_test_v25a*.npz
│   ├── oof_test_v27*.npz
│   └── oof_test_probs.npz
│
├── submissions/
│   ├── submission_v27_oldleak_20260525_0103.csv   ⭐ 參考檔(MD5 對照基準)
│   └── submission_v27_oldleak_{ts}.csv            您重現的輸出
│
└── docs/
    └── aicup2026_report.md                        ⭐ 競賽報告
```

---

## Troubleshooting

### Q1. `FileNotFoundError: data/external/shuttleset22/train.csv`

確認 ShuttleSet22 已下載並放置於正確路徑。見上方「資料」段。

### Q2. `RuntimeError: CUDA out of memory`

降低 batch size。編輯 `scripts/v27_modeA_full_pipeline.py` 中 `FT_BS = 64` → `FT_BS = 32`。注意降低 batch size 可能影響最終 MD5 重現(改用模式 B 驗證 ensemble 即可)。

### Q3. Stage 6 MD5 顯示 ⚠️ DIFFERS

兩個可能原因:

- **硬體差異**: 不同 GPU / CUDA 版本下 PyTorch 隨機性無法完全重現。請改用模式 B(從 cache)驗證。
- **資料版本不同**: 確認 `data/test.csv` 與 `data/test_new.csv` 是主辦單位 2026-05-21 公告的版本。

### Q4. PyTorch 警告 `weights_only=False`

PyTorch 2.4+ 新警告,不影響功能,可忽略。

### Q5. 無 GPU 環境

程式自動降到 CPU 模式。預期 bag training 時間長達 24+ 小時,不建議。

---

## 競賽合規說明

### 外部資料使用揭露

本系統使用 **2 項外部資料**,皆符合競賽規則:

1. **ShuttleSet22**: 公開的羽球(非桌球)資料集,用於跨運動 SSL 預訓練。屬公開研究資料,非反查 test。
2. **`data/test.csv`** (Reference_Only_Old_Test_Data): 主辦單位 2026-05-21 公告開放當訓練資料,含 1,236 / 1,845 test rallies 的 `serverGetPoint` ground truth。以 lookup 注入方式利用,屬主辦明確核可的使用方式。

### 生成式 AI 工具揭露

本隊伍於開發過程中使用 **Anthropic Claude (Opus 4.7)** 與 **OpenAI Codex** 作為程式輔助與分析協作工具。所有架構設計、實驗方向、最終提交,皆由人類隊員審查決策。詳見 [`docs/aicup2026_report.md`](docs/aicup2026_report.md) 壹段揭露說明。

### 繳交內容

本繳交包僅含:
- 重現 LB 0.4472604 必要的 2 個核心 `.py` 檔(orchestrator + hub)
- README.md(本檔)
- requirements.txt
- 參考 submission CSV(供 MD5 對照)
- 競賽報告 Markdown
- dataset_description.md(欄位說明)

不含資料檔(由參賽者自行下載)、不含 cache(訓練中自動產生,或聯絡隊伍取得)。

---

## 聯絡

**TEAM_10297** — 如審查過程有任何問題,請透過競賽平台聯繫。

---

_本繳交包對應之完整方法論、創新性、實驗分析: [`docs/aicup2026_report.md`](docs/aicup2026_report.md)_
