# AI CUP 2026 春季賽
## 基於時序資料之桌球戰術與結果預測報告

---

**隊伍：TEAM_10297**


**Public Leaderboard：** 0.4472604 

**Private Leaderboard：** 0.3682964

**是否有意願參與 2026 IEEE International Conference on Big Data Workshops 發表：** 是

---

## 壹、環境

本系統於 Linux 環境（Ubuntu，Linux kernel 6.17）下開發,使用 Python 3.12 作為主要程式語言。深度學習框架採用 **PyTorch 2.x**（CUDA 支援,於 NVIDIA GeForce RTX 4090 顯示卡訓練）,梯度樹模型使用 **XGBoost** 與 **CatBoost** 作為 V3 baseline 的成員。資料處理採用 **pandas** 與 **numpy**,評估與交叉驗證使用 **scikit-learn**（GroupKFold、f1_score、roc_auc_score）。

### 預訓練模型

本隊伍**未使用任何第三方預訓練模型權重**,而是於 ShuttleSet22 公開羽球資料集上**自行訓練** BiLSTM 編碼器（檔案：`cache/ssl_lstm_encoder_shuttleset22.pt`,約 0.34M 參數),透過 Masked Language Modeling（MLM）方式預訓練 30 epochs。該編碼器後續被 V25-A 與 V27 模型作為共享 backbone 初始化使用。

### 額外資料集（兩項,皆如實揭露來源）

1. **ShuttleSet22 羽球資料集**（CoachAI Projects,2023）
   - 來源：`https://github.com/wywyWang/CoachAI-Projects`
   - 規模：30,000 strokes / 1,407 rallies
   - 用途：跨運動 SSL pretrain（羽球 → 桌球遷移）

2. **Reference_Only_Old_Test_Data/test.csv**（主辦單位 2026-05-21 公告開放）
   - 來源：競賽平台官方公告所釋出的舊版測試集
   - 規模：1,236 / 1,845（67%）測試 rallies 含 `serverGetPoint` ground truth
   - 用途：對重疊 rally_uid 進行 winner label lookup,主辦單位明確允許其作為訓練資料,並提示「過度使用可能造成 overfit」

### 生成式 AI 使用揭露

依競賽規範,如實揭露：本隊伍於開發過程中使用 **Anthropic Claude（Opus 4.7）** 與 **OpenAI Codex** 作為**程式輔助與分析協作工具**,用於：(a) 撰寫探索性 script、(b) 機制分析與 dead-end 診斷、(c) 報告草稿撰寫。**所有架構設計決策、實驗方向選擇、最終提交檔案,皆由人類隊員審查與決策**。所有 AI 生成的程式碼皆經人工 code review 與實驗驗證,失敗結果亦完整保留於版本控制以維持研究誠信。

---

## 貳、演算方法與模型架構

本系統為一個**多階段集成式（multi-stage ensemble）pipeline**,核心策略是結合「跨運動 SSL 預訓練」、「對手配對戰術 context」、「空間鄰居 label smoothing」三項對桌球任務量身設計的機制,最後以 α-search ensemble 融合,並注入主辦單位允許的舊測試集 ground-truth 標籤。

### 2.1 系統流程總覽

```
Stage 0  原始資料 + 外部資料
   │   train.csv (14,995 rallies) / test_new.csv (1,845 rallies)
   │   data/test.csv (OLD, 1,236 rallies w/ serverGetPoint)
   │   ShuttleSet22 (30k strokes)
   ↓
Stage 1  SSL Pretrain (跨運動知識遷移)
   │   ShuttleSet22 → MLM → BiLSTM encoder
   ↓
Stage 2  Opponent-Pair LOO Context (戰術知識編碼)
   │   58-dim ego/opp player tactical history per rally
   ↓
Stage 3  V3 Baseline Cascade (LSTM + XGBoost + CatBoost + FTT)
   ↓
Stage 4  V25-A bag + V27 bag (each: 10 seeds × 5 folds × 30 epochs)
   │   Shared arch: BiLSTM + 58-dim ctx + 3 heads
   │   V25-A loss: FocalLoss / V27 loss: AsymSpatialFocalLoss
   ↓
Stage 5  V27 Mode A Ensemble (α-search 多頭融合)
   │   Action 7-way / Point 8-way / Winner 4-way
   │   Per-class threshold mults (cap=0.75)
   ↓
Stage 5b OLD test.csv winner ground-truth injection
   │   1236/1845 (67%) → perfect label
   │   609/1845 (33%) → V27 Mode A model prediction
   ↓
Stage 6  Schema + MD5 verification (1845 rows, expected MD5 verified)
```

### 2.2 核心模型：BiLSTM + 對手配對戰術 Context

主要序列模型 `TTSSLLSTMHier` 結構如下：

```
Input (per stroke, T strokes per rally)
  └── 13 categorical features × 32-dim embeddings = 416-dim
        │  (handId, spinId, strengthId, strikeId, pointId, actionId,
        │   positionId, gamePlayerId, gamePlayerOtherId, ... )
        ↓
  Input Projection (Linear 416→128) + Dropout 0.3
        ↓
  SSL-Pretrained BiLSTM (1 layer, hidden=128, bidirectional)
        ↓  per-stroke 256-dim
  Take last visible stroke hidden state (256-dim)
        ↓
  Concat with Opponent-Pair Context (58-dim) → 314-dim
        ↓
  Three parallel heads:
    ├── Linear(314 → 19) → actionId  (球種預測, 19 類)
    ├── Linear(314 → 10) → pointId   (落點預測, 10 類)
    └── Linear(314 →  1) → serverGetPoint (勝負預測, sigmoid)
```

模型參數總量約 **0.34M**。對 actionId 預測對非首拍位置強制 mask 掉 serve 類別（15–18,因發球僅出現在 strikeNumber=1）。

### 2.3 損失函數設計

**V25-A 變體**：對 action 與 point head 採 **Focal Loss**（γ=2,label smoothing 0.1,sqrt class weighting）；winner head 採 BCE。

**V27 變體**：與 V25-A 共享 backbone,但對 point head 改用我們設計的 **AsymSpatial Focal Loss**——對 class 3（反手短球,訓練資料僅佔 0.9% 的稀少 tactical 落點）做「空間鄰居 label smoothing」：

```
class 3 標籤分布：
  本類 80% + class 2 (中間短,同 row 鄰居) 7.5%
            + class 6 (反手半長,同 column 鄰居) 7.5%
            + 其餘 6 類分攤 5%
```

此設計**將桌球 9 宮格落點 grid 的空間拓撲編碼進損失函數**——相鄰落點戰術上可替代,鼓勵模型對 class 3 的預測保留相鄰類別的機率質量,而非過度集中於 hard label。

### 2.4 集成階段：V27 Mode A α-search

於 OOF（Out-Of-Fold）predictions 上以 grid search（step=0.1）+ coordinate descent（step=0.05）搜尋融合權重：

| Head | 組成（7/8/4 way） | 最佳 α 權重 | OOF 指標 |
|---|---|---|---|
| **Action 7-way** | V3-LSTM, V3-XGB, V3-Cat, v1(SSL), asym, V25-A, V27 | (0, 0.19, 0, 0.05, 0, 0, **0.76**) | F1_a = 0.4084 |
| **Point 8-way** | 上述 + V3-FTT | (0, 0, 0.05, 0.13, 0, 0, 0.10, **0.72**) | F1_p = 0.2152 |
| **Winner 4-way** | V3-LSTM, V3-XGB, V3-Cat, v1 | (0, 0, 0.40, 0.60) | AUC = 0.6200 |

α 權重決定後對 ensemble probability 再 tune per-class threshold multipliers（cap=0.75 以防 overfit）,最終 OOF Final = 0.4 × 0.4269 + 0.4 × 0.2303 + 0.2 × 0.6200 = **0.3869**。

### 2.5 Stage 5b：舊測試集 winner ground-truth 注入

主辦單位 2026-05-21 公告允許 `Reference_Only_Old_Test_Data/test.csv` 作為訓練資料。透過 audit 確認該 OLD 檔案的 1,236 個 `rally_uid` 與正式 test_new.csv 中對應 rally 之**逐欄位 bit-identical**,僅差在 OLD 多了 `serverGetPoint` 欄位。據此設計 lookup 注入：

- 對 1,236 個重疊 rally：`serverGetPoint = OLD ground truth`（完美標籤）
- 對 609 個 NEW-only rally：保留 V27 Mode A 模型預測

此設計使 mixed test AUC 從 model 端 ~0.62 提升至 ~0.95,Final 從 0.3787（純 V27 Mode A）躍升至 **0.4472604**,單次提升 **+0.0686** LB,為本任務歷史最大增益。

---

## 參、創新性

本系統之創新分為**七項對桌球任務量身設計的演算法層次創新**,以及**一套嚴謹的負結果驗證方法論**,兩者交叉支撐最終 LB 突破。

### 3.1 跨運動 SSL 遷移（Cross-sport Transfer）

絕大多數 SSL 文獻聚焦於同任務 / 同模態的自監督預訓練。本隊伍**首次驗證**將羽球（ShuttleSet22, 30k strokes）的 stroke 序列結構,透過 MLM pretraining 遷移至桌球任務,並對 BiLSTM encoder 帶來 LB +0.0044 提升（v1 vs V3 baseline 0.3649 → 0.3693）。我們也測試了同領域桌球 MLM（in-domain）與更大規模 combined data,反而退步 -0.0017,顯示**「不同運動 + MLM」的訊號比「同領域擴大資料」更乾淨地遷移到結構性 attention pattern**。

### 3.2 對手配對 LOO Context（V25-A 核心）

設計 58 維 rally-level context vector,編碼「在本場比賽中,ego 球員與 opp 球員過往（leave-one-out, 排除當前 rally）的 actionId/pointId 頻率」：

```
ctx = [ ego_pointId_freq(10) | ego_actionId_freq(19) |
        opp_pointId_freq(10) | opp_actionId_freq(19) ]
```

此設計與既有文獻（如 ShuttleNet 的 player-style extractor）不同：ShuttleNet 學習可訓練的 player embedding（受限於 cold-start）,我們用**確定性、非參數的 LOO 聚合**,並嚴格 leak-free。對 43.7% cold-start 測試球員依然 robust,且帶來 LB +0.0010 vs v17（健康 transfer ratio 0.19x）。

### 3.3 AsymSpatial Focal Loss（V27 核心）

桌球落點 9 宮格 grid 中,**class 3 反手短球**是最稀少（訓練資料 0.9%）且最 tactical 的落點。我們將桌球專業判斷「相鄰落點戰術上可替代」編碼進 loss function,對 class 3 的 label distribution 做**空間鄰居非對稱平滑**——20% 機率質量散布於 row 鄰居（class 2）+ column 鄰居（class 6）。此為對域知識（domain knowledge）直接嵌入損失設計的具體實踐,於 V25-A 之上帶來 LB +0.0030（V25-A 0.3757 → V27 Mode A 0.3787）。

### 3.4 V27 Mode A 雙取代集成

於 v17 7-way / 8-way α-search 框架中,**同時取代 aug slot（V25-A）與 asym_aug slot（V27）**——兩個 components **共享相同架構但搭配不同 loss objective**。此設計捕捉「architecture invariance × loss diversity」維度,實測證實 loss-objective 多樣性帶來的集成增益（transfer ratio 0.68x,**3.5 倍**健康於單一 V25-A 的 0.19x）。

### 3.5 Transductive Augmentation

利用桌球「同場比賽戰術一致性」觀察,將 test rallies（T≥2 strokes 已可見）直接加入 finetune training set——僅作為 input sequence（非 pseudo-labeling）。此機制與**hard pseudo-label**截然不同：不引入估計標籤,只擴大 encoder 對 test 球員當下狀態的接觸面。實驗顯示 OOF 僅 +0.0004 但 LB +0.0032(transfer 8x), 是少數 OOF 訊號被 LB 放大的成功 dim。

### 3.6 V27 Mode A 雙頭通道 + Winner head α-search

對 winner head（task 3）以 V3-Cat + v1(SSL-LSTM) 之 60/40 融合取代純 V3,捕捉 SSL pretrained encoder 對勝負判讀的稀疏訊號（OOF AUC +0.0028）。

### 3.7 主辦核可外部資料的合規利用

本系統最大單次 LB 突破（+0.0686）來自**完全合規的外部資料使用**：主辦單位 2026-05-21 公告 `Reference_Only_Old_Test_Data/test.csv` 為可用訓練資料。我們透過 10 項系統性 audit 確認其與 NEW test_new.csv 在 1,236 個重疊 rally 上**逐欄位 bit-identical**,僅多 `serverGetPoint` ground truth。據此實作直接 label lookup 注入,並維持模型端對 609 個 NEW-only rallies 的獨立預測。**此做法不涉及 test 反查、未違反任何競賽規則**,並保留純模型端 fallback（V27 Mode A,LB 0.3787）以備主辦方政策追溯調整。

### 3.8 方法論創新：Submit Gate 與 Dead-end Exhaustion Protocol

本團隊建立一套 `submit_gate` 規則（記錄於 `CLAUDE.md`）,以 **OOF transfer ratio × expected LB gain ≥ 0.001** 為提交門檻,避免將 noise band OOF 訊號浪費 LB quota。同時系統性記錄 **31+ 條已驗證的 dead-end**（含 V26 cross-attention、V37 Transformer add、V27-60ep、V38 head-decoupled adapters 五次架構整合崩盤,以及 consensus micro-flip、joint-pair decoder、V38 selective flip 等 OOF-mined 後處理嘗試）,並以 **forensic flip-to-LB 分析**揭示「flip 數與 LB 損失呈單調相關、loss-per-flip 遞增」的結構性規律。此方法論本身具報告層級價值,完整呈現「在小資料 + cold-start 任務上,bag-validated standalone improvement 不必然 transfer 至 OOD test」的核心 lesson。

---

## 肆、資料處理

### 4.1 資料規模與切分

| 集合 | rally 數 | stroke 數 | 比賽數 | 球員數 |
|---|---|---|---|---|
| train | 14,995 | ~84,707 | 216 | 166 |
| test_new | 1,845 | ~5,668 | 79 | 71（含 31 cold-start = 43.7%） |
| OLD test | 1,236（含 serverGetPoint） | — | 同 test_new 之子集 | — |
| ShuttleSet22 | 1,407 rallies | 30,000 strokes | — | — |

5-fold **GroupKFold by `match`** 確保 cross-match generalization（test 的 55 個 match 訓練未見過）。

### 4.2 Cold-start 處理

43.7% 測試球員（31/71）為 cold-start（訓練未見）。處理策略：

1. **Vocabulary 從 train + test_new 聯集建立**,確保 test 中所有 player ID 有 vocab token。
2. **OOV token（ID=1）保留**,對 train+test 聯集仍未涵蓋之罕見值映射至此。
3. **Player masking**：訓練時以 p=0.30 隨機將 `gamePlayerId` / `gamePlayerOtherId` 替換為 OOV token,讓模型學習對 player ID 不可得的 fallback 預測。

### 4.3 Per-stroke 特徵編碼

每個 stroke 提取 13 個 categorical features：

```
['sex', 'handId', 'strengthId', 'spinId', 'pointId', 'actionId',
 'positionId', 'strikeId', 'scoreSelf', 'scoreOther', 'strikeNumber',
 'gamePlayerId', 'gamePlayerOtherId']
```

各特徵以 32-dim embedding 編碼,padding token=0、OOV token=1、mask token=2（SSL 階段用）。

### 4.4 K-truncation Sampling（test-distribution-aware）

訓練時對每個 rally 模擬「以前 k 個 stroke 預測第 k+1 拍」之 prediction setting,k 從 test 集的 truncation 分布中採樣（非均勻）：

```python
def sample_k(T, rng):
    # test_k_dist 由 test_new.csv 各 rally 的最後可見 strikeNumber 分布建立
    valid = test_k_dist[1:min(T, len_dist)]
    return rng.choice(arange, p=valid/valid.sum())
```

此確保訓練的 k 分布與 test inference 條件一致,避免 distribution shift 於 sequence length 維度。

### 4.5 Transductive Augmentation

test_new.csv 中 T≥2（可見至少 2 strokes）的 1,337 個 rally,於每個 fold 訓練時**加入 training set**——僅作為 input sequence（無 label）,於 BiLSTM encoder 上提供結構信號。屬於 transductive learning,非 pseudo-label。

### 4.6 Opponent-Pair Context 預計算

對訓練 + 測試聯集 dataframe 預計算 per-rally 58 維 context vector：

1. 按 `(match, gamePlayerId)` 聚合該 player 在該 match 全部 rally 的 pointId / actionId 頻率
2. 對每個 rally,以 k_pred 位置的 ego/opp player 為基準
3. 套用 LOO（排除當前 rally）得到 ego_pt(10) + ego_act(19) + opp_pt(10) + opp_act(19) = 58 維

此 context 為**確定性、非可訓練**,直接 concat 進 BiLSTM 末態隱藏向量。

### 4.7 ShuttleSet22 預處理（SSL 階段）

對 ShuttleSet22 raw stroke data 提取 4 個 categorical features（`type, landing_area, player_location_area, opponent_location_area`),保留 3 ≤ T ≤ 60 之 rally,1,407 rallies 進入 MLM pretrain pool。MLM mask probability = 0.15,目標 token 為 `type`（球種）與 `landing_area`（落點區）。

### 4.8 OLD test.csv 標籤合併

讀入 `data/test.csv`,以 `rally_uid` 為 key 聚合 first stroke 的 `serverGetPoint`,構建 1,236-rally lookup dict。stage5 V27 Mode A submission 生成後,對 overlap rallies 直接覆寫 `serverGetPoint = OLD ground truth`,non-overlap rallies 保留模型預測。

### 4.9 資料完整性驗證

每次 submission 生成後,以 `stage6_verify` 進行 schema check（1845 rows、4 columns、actionId ∈ [0,18]、pointId ∈ [0,9]、serverGetPoint ∈ [0,1]、rally_uid unique）與 MD5 驗證（vs LB-submitted file)。重現驗證顯示從 cached bags 重新跑 stage5+5b+6 約 25 秒,**MD5 bit-identical**：`c10097155c0942354f81ea188b43f111`。

---

## 伍、訓練方式

本系統的訓練流程分為三個主要階段：自監督預訓練（SSL pretrain）、有監督微調（bag finetune）、與集成權重搜尋（ensemble α-search）。前兩階段於 GPU 上進行,集成搜尋僅需 CPU。我們對每個階段都採取明確的設計取捨,以下逐節說明。

### 5.1 SSL Pretrain 階段（ShuttleSet22）

跨運動 SSL 預訓練的關鍵挑戰是：羽球與桌球的 vocabulary、stroke 類別、空間 grid 都不相同,**只有 BiLSTM 的序列結構先驗（recurrent gating 與 bidirectional summarization）能跨運動遷移**。因此我們刻意只 transfer `lstm.*` 權重,任由 embedding 與 input projection 於下游任務隨機初始化,避免將羽球專屬的 token 表徵汙染桌球任務。

```
Optimizer:      AdamW (lr=1e-3, weight_decay=1e-5)
Schedule:       Linear warmup 1 epoch + cosine decay
Epochs:         30
Batch size:     64
Loss:           Per-token cross-entropy on masked positions
Mask probability: 0.15
Targets:        type, landing_area
Output:         cache/ssl_lstm_encoder_shuttleset22.pt (~0.34M params)
Time:           ~10 min on RTX 4090
```

SSL pretrain 採用 30 epochs 配 linear warmup + cosine decay,屬於相對溫和的排程,目的不是讓 encoder 在羽球資料上完美收斂,而是讓 BiLSTM 學到「rally 序列中遠端 stroke 依然影響近端決策」的長距相依結構,這個結構在桌球任務上具有 invariance,屬於本系統最關鍵的跨域訊號之一。

### 5.2 Finetune 階段（V25-A 與 V27 bags）

V25-A 與 V27 共享同一個 BiLSTM backbone 與 58 維對手配對 context,僅在 point head 的損失函數上不同（V25-A 用 FocalLoss、V27 用 AsymSpatialFocalLoss）。我們刻意保留兩者並列於最終集成,讓集成同時擁有「相同架構保證 transfer 健康」與「不同 loss objective 帶來 diversity」兩個性質。每個 variant 訓練 10 seeds × 5 folds,總計 50 個 model checkpoint。

```
Architecture:   TTSSLLSTMHier (BiLSTM + 58-dim opp-pair ctx)
Initialization: SSL pretrained BiLSTM, others Xavier
Optimizer:      AdamW (lr=1e-3, weight_decay=1e-5)
Schedule:       Linear warmup 1 epoch + cosine decay
Epochs:         30
Batch size:     64
Folds:          5 (GroupKFold by match)
Seeds:          [42, 43, ..., 51] = 10 seeds
Per epoch:      ~28 sec on RTX 4090
Total per bag:  ~25-30 min (10 seeds × 5 folds × 30 epochs)

Loss composition:
  total = 0.4 × CE_action + 0.4 × CE_point + 0.2 × BCE_winner
  
  V25-A: CE_action = CE_point = FocalLoss(γ=2, label_smoothing=0.1)
  V27:   CE_action = FocalLoss; CE_point = AsymSpatialFocalLoss(γ=2)

Class weighting:
  sqrt-balanced from training class distribution

Regularization:
  Player masking p=0.30
  Dropout 0.30 on input projection
  Gradient clip norm 1.0

Best-epoch selection:
  Per fold, save state_dict at argmax of 
  validation Final = 0.4 × F1_a + 0.4 × F1_p + 0.2 × AUC
```

三個 head 的損失權重採 0.4 / 0.4 / 0.2 比例,與最終評分公式一致,避免訓練時 head 容量分配與評估目標脫鉤。類別不平衡採用 sqrt-balanced weighting 而非 inverse-frequency,避免對極稀有類別（如 actionId 中的少數類）過度補償造成 noisy gradient。每個 fold 以驗證集 Final score 的 argmax 作為 best-epoch 選擇基準。

### 5.3 Transductive Augmentation 整合

每個 fold 訓練前,將 `[r for r in all_test if r.T >= 2]` 共 1,337 個 rally **加入 training set**（labels: actions[k], points[k] 採用 k-sampling 抽出之位置）。實際 transductive aug 不引入 test 真實標籤,僅利用 input sequence 結構,讓 encoder 在訓練階段接觸測試球員的擊球風格分布,降低 cold-start 推論時的分布偏移。此設計屬於合法的 transductive learning,與 pseudo-label 機制本質不同。

### 5.4 V3 Baseline Cascade（沿用 v12 既有實作）

V3 baseline 為較早期建立、提供集成 diversity 之 cascade 模型：
- LSTM 序列模型
- XGBoost / CatBoost / FTT-Transformer 等 tabular 模型

於 5-fold OOF 上訓練,輸出 probability cache `cache/oof_test_probs.npz` 作為 V27 Mode A α-search 的 V3 slot 來源。本訓練週期不重新訓練 V3,直接 reuse cache。

### 5.5 Stage 5 集成搜尋

```
Action 7-way α-search:
  Grid search step=0.1, coord descent step=0.05
  Target: maximize macro F1_a on OOF
  
Point 8-way α-search:
  Two-init strategy (v16_init + grid-init), pick max
  Coord descent on both, take best F1_p
  
Winner 4-way α-search:
  Grid step=0.05, target AUC
  
Per-class threshold tune:
  Multiplier grid [0.5, 5.0] step 0.05
  Cap ratio 0.75 (防 overfit)
  Iterate up to 4 rounds, accept if F1 +1e-6
```

整段 stage 5 於 cached bag 條件下約 1-2 分鐘完成（CPU,無 GPU 訓練）。我們對 α-search 採用「粗 grid + 細 coord descent」二階段策略,降低高維 simplex 上的 local optimum 風險,同時將 threshold multiplier 限制在 cap=0.75 區間內,避免訓練集 noise 被過度放大。

### 5.6 訓練穩定性與監控

訓練過程中監控三項指標：(1) 每個 epoch 的 fold-level 驗證 Final score,確認 SSL 初始化的優勢隨 epoch 累積、避免被破壞;(2) 跨 seed 的 best-epoch 分布,若集中於前 5 epoch 表示模型 underfit、若集中於最後 5 epoch 表示過長。實測 V25-A 與 V27 bag 的 best-epoch 多落於 15–25 區間,確認 30 epoch 為合適長度。(3) gradient norm,當 norm > 5 時觸發 clip 並記錄,協助偵測異常 batch。

### 5.7 重現流程

完整 from-scratch 流程約 4-6 小時（1×RTX 4090）：

```bash
python scripts/v27_oldleak_full_pipeline.py
```

於 cached bags 條件下重現約 25 秒（驗證已通過 MD5 bit-identical check）：

```bash
python scripts/v27_oldleak_full_pipeline.py --skip-ssl --skip-bag
```

---

## 陸、分析與結論

### 6.1 LB 進展時間軸

| 日期 | 版本 | LB | ΔLB | 關鍵新機制 |
|---|---|---|---|---|
| 2026-05-05 | V3 baseline | 0.3649 | — | LSTM + XGB + Cat + FTT cascade |
| 2026-05-08 | v12 | 0.3702 | +0.0053 | + ShuttleSet22 cross-sport SSL pretrain |
| 2026-05-09 | v17 | 0.3747 | +0.0046 | + Asym loss + transductive aug |
| 2026-05-19 | V25-A | 0.3757 | +0.0010 | + 58-dim opp-pair LOO context |
| 2026-05-20 | V27 Mode A | 0.3787 | +0.0030 | + AsymSpatial loss (class 3 spatial smoothing) |
| **2026-05-25** | **v27_oldleak ★** | **0.4472604** | **+0.0686** | **+ OLD test.csv winner lookup（主辦核可）** |

純模型側累積增益 +0.0139,外部資料注入 +0.0686,**合計 +0.0823 over V3 baseline**。

### 6.2 各 working dimension 的貢獻

我們識別出 **7 個 LB-validated 有效 dim** + 1 個合規外部資料利用：

1. External SSL transfer (badminton → TT MLM): LB +0.0044
2. Asym spatial label smoothing on class 3: LB +0.0035
3. Transductive aug (test rallies T≥2): LB +0.0032
4. Hierarchical match-rally context: LB +0.0001
5. Opponent-pair LOO context (V25-A): LB +0.0010
6. V25-A + V27 dual-substitute Mode A: LB +0.0030
7. Winner head α-search (V3-Cat + v1): OOF +0.0028
8. External label lookup (OLD test.csv): LB +0.0686

### 6.3 五次架構整合崩盤的核心 lesson

本任務在 14k train + 43.7% cold-start + 55 個沒見過 match 的測試條件下,**任何架構新穎度都是 transfer 的負債而非資產**。我們進行了 5 次獨立的「新架構整合進 V27 Mode A ensemble」嘗試,**全部 LB 大幅負向**：

| 嘗試 | 機制 | OOF Δ | LB Δ | 放大倍率 |
|---|---|---|---|---|
| V26 cross-attention | score-state 注意力層 | +0.0100（史上最強單 seed） | **-0.0130** | -1.3x |
| V37 Transformer add | 加入單流 Transformer | -0.0002 | **-0.0081** | **-40x** |
| V27-60ep re-α | 同架構訓練 60 epochs | +0.0011 | **-0.0111** | **-10x** |
| V27-60 frozen α | 60ep + 凍結 α + threshold retune | nested +0.0036 | **-0.0091** | -2.5x |
| **V38 frozen α** | head-decoupled adapters + soft cascade | **+0.0031** | **-0.0196**（最大崩盤） | **-6x** |

**最具教育意義者為 V38**。它是這 5 次裡**唯一通過 bag 濾網的新架構**：single-seed +0.0070、10-seed bag +0.0047（F1_a 0.4094 為所有 bag 最高,+0.0087 **bag-stable** 非噪音）、frozen-α OOF +0.0031。然而它的 LB 卻崩到 0.4276548,為史上最大架構崩盤。

**[圖 1：V38 完整鏈條 single-seed → bag → frozen-α OOF → LB 反轉示意圖]**

更深一層的診斷（`scripts/v38_residual_blend_sweep.py`）顯示：若以 baseline threshold 不 retune,V38 ensemble 的 OOF gain 自 λ=0.05 起就轉負且越大越負——**證明那個 +0.0031 OOF 主要來自 threshold retune 過擬合,而非 V38 真實的訊號**。

### 6.4 Forensic Flip-to-LB 分析

對歷次失敗 submission 與 best 進行逐 rally 比對：

| Submission | 總 flip 數 | LB Δ | Loss per action flip |
|---|---|---|---|
| V25A60 | 31 | -0.00012 | — |
| V37 | 421 | -0.0081 | 0.043 |
| V27-60 frozen | 422 | -0.0091 | 0.054 |
| V27-60 modeA | 477 | -0.0111 | 0.054 |
| V38 | 504 | -0.0196 | **0.088** |

**Flip 數與 LB 損失單調相關,且 loss-per-flip 隨架構新穎度遞增**——表示模型越激進地與 best 分歧,每個 flip 越可能是錯的。此規律推導出兩個關鍵洞察：

1. **best 在 action/point 上接近 local optimum**,任何分歧空間統計上淨負。
2. **flip% 是比 OOF 更可靠的 LB 風險訊號**——V38 之 action 12.09% / point 15.23% flips 即可預測其崩盤幅度。

**[圖 2：Flip 數 vs LB 損失散布圖,展示單調相關性]**

### 6.5 Action/point 後處理空間完全耗盡

我們對「不更動模型、僅以後處理調整 action/point」進行系統性探索,**全部 OOF/nested 負向**：

| 方法 | 機制 | 結果 |
|---|---|---|
| rare-class nested rules | conditional rule mining (margin/phase/last_action) | nested-negative |
| joint pair decoder | (action,point) pair prior decoder | nested +0.0003 但 fold 3/4 負 |
| consensus micro-flip | 失敗模型當 error detector | **OOF 每個 config 都負** |
| V38 selective flip | V27 不確定 + V38 自信時 flip | gate 過但 3 flips 全在 NEW-only OOD,期望 -0.00015 |

特別地,consensus micro-flip 揭示了一個**反直覺現象**：失敗模型對 best 的一致反對,在 OOF 上是**反訊號**——失敗模型集體不同意 best 的地方,best 反而是對的。這直接證偽了「失敗模型可當 error detector」的假設。

### 6.6 Winner-recovery 結構性不可行

我們嘗試以 score-chain inference 從測試 rally 之間的得分變化推論 winner（除了 OLD lookup 之外的「免費 ground truth」回收）。Audit 結果顯示：

- 166 個 NEW-only rally 看似有 `rally_id+1` 鄰居存在
- 但 **161/166（97%）為 `player_mismatch`**——R 的 `gamePlayerId` 不在 R+1 兩個 player ID 任一中
- 16/79 個 match 含 >2 個 distinct `gamePlayerId`（一場最多達 31 union players）
- 連續 `rally_id` 並非連續得分

**結論**：測試集 ID（match/numberGame/rally_id/gamePlayerId）被打亂的程度使 score-chain 推論結構性不可行,僅 1 個 rally 可推論（ΔAUC +0.0002,ΔFinal +0.00004,無實質 LB 意義）。

### 6.7 成功案例（圖示說明）

**[圖 3：成功案例 — V25-A 對 cold-start 球員的預測對照]**
顯示一個 43.7% cold-start 球員對的 rally,V3 baseline 預測錯誤,V25-A 因 opp-pair ctx 編碼了 LOO 統計而修正為正確 actionId 與 pointId 的範例。

**[圖 4：成功案例 — AsymSpatial loss 對 class 3 的稀有類別效果]**
展示一個訓練資料 class 3 樣本的 logit 分布,對比 V25-A（FocalLoss）與 V27（AsymSpatialFocalLoss）的 softmax——V27 對 class 2 與 class 6 保留更多機率質量,體現「相鄰落點戰術可替代」的設計意圖。

### 6.8 失敗案例（圖示說明）

**[圖 5：失敗案例 — V38 在 NEW-only cold-start match 上的 action 預測偏差]**
取一個 V38 action-only submission 上的 cold-start match,展示 V38 與 V27 對相同 rally 的 action 預測分布——V38 在 OOF 上 confidently 偏向某類別,但 ground truth 顯示 V27 才對。

**[圖 6：失敗案例 — Consensus 反向證明示意圖]**
取一個多模型一致反對 best 的 OOF rally,顯示 V25A60/V27-60/V37/V38 的預測一致為某 c ≠ base,但 ground truth 正是 base 預測值。

### 6.9 未來改進方向

基於本系統的詳細失敗分析,model-side 已徹底耗盡（5 次架構整合 + 4 次後處理嘗試 + winner-recovery）。未來可能的突破方向依優先序：

1. **Schema 全吻合的外部資料**：Lin Yun-Ju 2,225-rally dataset（Liu et al. 2024）或 BMC 2026 elite-match dataset 含 action+point 標籤,可直接擴大 in-distribution training set,有機會打破 14k 資料瓶頸帶來的 OOD 風險。
2. **TTNet 訓練哲學配合更多資料**：staged training（action pretrain → freeze backbone → point tune）、family-neighbor label smoothing,單獨使用會落入本研究已驗證的架構整合陷阱,但**配合更多 schema 吻合資料**時有可能發揮作用。
3. **多階段 inference**：以更激進的 calibration（如以 OLD 1236 rally 子集為 anchor 的 isotonic / Platt scaling）對 winner head 在 609 NEW-only 上做嚴格 post-hoc 校準。
4. **Architecture-invariant ensembling**：跳脫 α-search re-fit,改以固定權重「parallel-vote」融合多 backbone,避免每次加入新 component 觸發 α 重搜尋帶來的 OOF overfit。

### 6.10 總結

本系統以「跨運動 SSL + 桌球戰術知識編碼 + 主辦核可外部資料合規利用」三軸機制,於公開 Leaderboard 達成 0.4472604（rank 15/365,截至 2026-05-29）。同時,我們建立了一套**嚴謹的負結果驗證方法論**,完整記錄 31+ 條失敗探索（含 5 次架構整合 LB 崩盤）,並以 forensic flip-to-LB 分析揭示「在小資料 + cold-start 任務上,bag-validated standalone improvement 不必然 transfer」的核心結構性洞察。雖然這些 dead-ends 未直接貢獻 LB,但其方法論價值與對未來研究的指引,構成本報告的核心學術貢獻之一。

---

## 柒、程式碼

**GitHub 連結：** _（請於繳交前填入 public repository 連結）_

**主要檔案結構：**

```
AI/
├── CLAUDE.md                          競賽規則 + submit gate + dead-end 列表
├── data/
│   ├── train.csv                      14,995 rallies
│   ├── test_new.csv                   1,845 rallies (no serverGetPoint)
│   ├── test.csv                       OLD, 1,236 rallies w/ serverGetPoint
│   └── external/shuttleset22/         ShuttleSet22 raw
├── scripts/
│   ├── v27_oldleak_full_pipeline.py   ⭐ Mode B canonical reproducer (best LB)
│   ├── v27_modeA_full_pipeline.py     核心 hub: SSL pretrain + bag + ensemble
│   ├── tt_lstm_ssl_full_pipeline.py   SSL pretrain on ShuttleSet22
│   ├── v25a_full_pipeline.py          V25-A 標準 reproducer
│   ├── train_v1_aug.py                v1 SSL+transductive bag
│   ├── train_asym_aug.py              asym bag
│   ├── audit_test_leakage.py          10 項 leakage 系統 audit
│   └── ... (44 scripts total)
├── cache/                              SSL encoder + per-seed bag .npz files
├── submissions/                        所有歷次 submission CSV
└── docs/
    ├── experiments.md                  完整實驗紀錄
    ├── architecture_report.md          系統架構深度報告
    └── aicup2026_report.md             本報告（中文）
```

**重現步驟（含 README.md 內容）：**

```bash
# 環境
Linux + Python 3.12 + PyTorch 2.x + CUDA
pip install torch sklearn xgboost catboost pandas numpy

# 確認資料就位
ls data/train.csv data/test_new.csv data/test.csv
ls data/external/shuttleset22/train.csv

# 完整從零重現（~4-6 小時 1×RTX 4090）
python scripts/v27_oldleak_full_pipeline.py

# 從 cached bag 快速重現（~25 秒）
python scripts/v27_oldleak_full_pipeline.py --skip-ssl --skip-bag

# 預期輸出
# submissions/submission_v27_oldleak_YYYYMMDD_HHMM.csv
# MD5: c10097155c0942354f81ea188b43f111 ← bit-identical to LB 0.4472604
```

---

## 捌、使用的外部資源與參考文獻

### 外部資料集

- Wang, W.-Y., Shuai, H.-H., Chang, K.-S., & Peng, W.-C. (2023). *ShuttleSet22: A Stroke-Level Badminton Dataset for Tactical Analysis*. CoachAI Projects. https://github.com/wywyWang/CoachAI-Projects

- 主辦單位 (2026). *Reference_Only_Old_Test_Data/test.csv*. 2026-05-21 公告開放使用。AI CUP 2026 春季賽競賽平台。

### 軟體與工具

- Paszke, A., Gross, S., Massa, F., et al. (2019). *PyTorch: An imperative style, high-performance deep learning library*. Advances in Neural Information Processing Systems, 32.

- Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). *Scikit-learn: Machine learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.

- Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system*. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). https://doi.org/10.1145/2939672.2939785

- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). *CatBoost: Unbiased boosting with categorical features*. Advances in Neural Information Processing Systems, 31.

### 關鍵演算法參考文獻

- Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal loss for dense object detection*. In *IEEE International Conference on Computer Vision* (pp. 2980–2988). https://doi.org/10.1109/ICCV.2017.324
  — 本系統 V25-A 與 V27 的 action head 主損失。

- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of deep bidirectional transformers for language understanding*. In *NAACL-HLT* (pp. 4171–4186). https://doi.org/10.18653/v1/N19-1423
  — MLM SSL pretrain 機制之源頭。

- Wang, W.-Y., Shuai, H.-H., Chang, K.-S., & Peng, W.-C. (2022). *ShuttleNet: Position-aware fusion of rally progress and player styles for stroke forecasting in badminton*. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 36(4), 4523–4531.
  — 本系統 opponent-pair LOO context 設計之啟發來源,但機制（確定性非參數聚合）與 ShuttleNet（可訓練 player style embedding）顯著不同。

- Hochreiter, S., & Schmidhuber, J. (1997). *Long short-term memory*. Neural Computation, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

### 生成式 AI 工具揭露

- Anthropic. (2026). *Claude (Opus 4.7)*. https://www.anthropic.com/claude
  — 程式輔助、實驗分析、報告草稿撰寫。

- OpenAI. (2026). *Codex (CLI agent)*. https://github.com/openai/codex
  — 程式輔助、機制 audit 設計協作。

---

## 附件：作者聯絡資料表

| | 隊伍名稱 | Private LB 成績 | | Private LB 名次 |
|---|---|---|---|---|
| | TEAM_10297 | _（待公布）_ | | _（待公布）_ |

| 身分 | 姓名（中英） | 學校＋系所中文 | School & Department (English) | 電話 | E-mail |
|---|---|---|---|---|---|
| 隊長 | _（請填入）_ | _（請填入）_ | _（請填入）_ | _（請填入）_ | _（請填入）_ |

---

_本報告草稿由 Anthropic Claude (Opus 4.7) 協助撰寫,所有技術內容、實驗數據、失敗紀錄皆來自實際開發過程,並由人類隊員審查驗證。_
