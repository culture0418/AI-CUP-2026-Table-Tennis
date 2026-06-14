## 資料集欄位說明 (Dataset Feature Specifications)

| 欄位 (Features) | 說明 (Description) | 定義 (Definition) |
| :--- | :--- | :--- |
| **rally_uid** | 小分的唯一識別碼 | Unique ID for each rally |
| **sex** | 比賽性別（男 1 / 女 2） | Gender category of the match |
| **match** | 比賽的唯一識別碼 | Unique ID of the match |
| **numberGame** | 小局數（第幾局） | Game (set) number within the match |
| **rally_id** | 小局內的小分編號 | Rally ID within the game |
| **strikeNumber** | 小分內的揮拍次序 | Stroke number within the rally |
| **scoreSelf** | 主視角選手的得分 | Points won by the main-view player |
| **scoreOther** | 對側選手的得分 | Points won by the opponent player |
| **serverGetPoint** | 發球者是否得分（1=是, 0=否） | Whether the server won the point |
| **gamePlayerId** | 主視角選手的 ID | ID of the main-view player |
| **gamePlayerOtherId** | 對側視角選手的 ID | ID of the opponent player |
| **strikeId** | 揮拍狀態或動作類型 | Stroke action type or state identifier |
| **handId** | 正手或反手揮拍 | Forehand or backhand stroke indicator |
| **strengthId** | 擊球力道 | Stroke strength level |
| **spinId** | 球的旋轉方式 | Type of spin applied to the ball |
| **pointId** | 球的落點位置 | Landing position of the ball on the table |
| **actionId** | 擊球方式 | Stroke or action type |
| **positionId** | 球員站位區域 | Player’s court position |

### 欄位詳細說明

* **serverGetPoint (Label)**:
    * 此欄位為本預測模型的標籤（Label）。
    * 用於判定該回合（Rally）最終是否由發球方取得分數。
* **pointId (Feature)**:
    * 紀錄球在球桌上的物理位置座標或區域編號。
* **actionId (Feature)**:
    * 紀錄該次擊球的技術動作類型（如：正手、反手、切球等）。


### 1. 揮拍狀態與手別
| 類別 | ID | 說明 | Definition |
| :--- | :---: | :--- | :--- |
| **strikeId** | 1 | 發球 | serving |
| | 2 | 接發球 | reserve |
| | 4 | 第三板之後 | rally |
| | 8 | 無(未錄影) | zero |
| | 16 | 暫停 | stop |
| **handId** | 0 | 無 | zero |
| | 1 | 正拍 | forehand |
| | 2 | 反拍 | backhand |

### 2. 擊球屬性 (力道與旋轉)
| 類別 | ID | 說明 | Definition |
| :--- | :---: | :--- | :--- |
| **strengthId** | 0 | 無 | zero |
| | 1 | 強 | strong |
| | 2 | 中 | medium |
| | 3 | 弱 | slow |
| **spinId** | 0 | 無 | zero |
| | 1 | 上旋 | top spin |
| | 2 | 下旋 | back spin |
| | 3 | 不旋 | no spin |
| | 4 | 側上旋 | side top spin |
| | 5 | 側下旋 | side back spin |

### 3. 落點與站位 (Placement & Position)
| 類別 | ID | 說明 | Definition |
| :--- | :---: | :--- | :--- |
| **pointId** | 0 | 無/出界 | zero |
| | 1 | 正手短球 | forehand position near net |
| | 2 | 中間短球 | middle position near net |
| | 3 | 反手短球 | backhand position near net |
| | 4 | 正手半出台 | forehand position half-long |
| | 5 | 中路半出台 | middle position half-long |
| | 6 | 反手半出台 | backhand position half-long |
| | 7 | 正手長球 | forehand position long |
| | 8 | 中間長球 | middle position long |
| | 9 | 反手長球 | backhand position long |
| **positionId** | 0 | 無 | null |
| | 1 | 左 | left |
| | 2 | 中 | middle |
| | 3 | 右 | right |

### 4. 擊球動作類型 (Action ID)
| ID | 說明 | Definition | Action Type |
| :---: | :--- | :--- | :--- |
| 0 | 無 | zero | Zero |
| 1 | 拉球 | drive | Attack |
| 2 | 反拉 | counter drive | Attack |
| 3 | 殺球 | smash | Attack |
| 4 | 擰球 | backhand twist | Attack |
| 5 | 快帶 | fast drive | Attack |
| 6 | 推擠 | fast push | Attack |
| 7 | 挑撥 | flip | Attack |
| 8 | 拱球 | pimple’s long push | Control |
| 9 | 磕球 | pimple’s fast push | Control |
| 10 | 搓球 | long push | Control |
| 11 | 擺短 | drop shot | Control |
| 12 | 削球 | chop | Defensive |
| 13 | 擋球 | block | Defensive |
| 14 | 放高球 | lob | Defensive |
| 15 | 傳統發球 | traditional | Serve |
| 16 | 勾手發球 | hook | Serve |
| 17 | 逆旋轉發球 | reverse | Serve |
| 18 | 下蹲式發球 | squat | Serve |

---

## 評分方式

本競賽旨在評估參賽者利用前 *n−1* 拍的時序資料，對下一拍（第 *n* 拍）及當前回合（Rally）結果進行多項預測的綜合能力。為實現對所有參賽者的單一排名，競賽採用一個**綜合評分指標（Overall Score）**，由以下三項預測任務的個別表現加權平均構成。

### 三項任務

| 任務 | 預測欄位 | 評估指標 (Sᵢ) | 採用該指標的理由 |
|---|---|---|---|
| **任務一**：下一拍（第 n 拍）的球種預測 | `actionId` | **Macro F1-Score** | 球種類別樣本高度不均衡。Macro F1 給予所有類別相同權重，評估稀有球種的平均預測表現。 |
| **任務二**：下一拍（第 n 拍）的落點預測（九宮格） | `pointId` | **Macro F1-Score** | 不同落點區域的擊中頻率不一。Macro F1 在各區域間提供均衡的性能評估。 |
| **任務三**：根據此回合已進行的擊球序列，預測最終此 Rally 的發球者是否得分 | `serverGetPoint` | **AUC-ROC** | AUC-ROC 衡量模型整體區分能力，不受特定分類閾值影響，並在處理潛在類別不平衡時表現穩健。 |

### 綜合評分公式

```
Score = w₁ × S₁ + w₂ × S₂ + w₃ × S₃
```

各任務分數 *Sᵢ* 均標準化至 [0, 1] 範圍。權重設定：

| 權重 | 數值 | 對應任務 |
|---|---|---|
| w₁ | **0.4** | actionId 球種預測 |
| w₂ | **0.4** | pointId 落點預測 |
| w₃ | **0.2** | serverGetPoint 勝負預測 |

權重設計理由：
- 準確預測對方下一拍的**球種**（任務一）和**落點**（任務二）是戰術分析與應對的核心，戰術價值高，故給予較高權重 0.4。
- 預測整個回合的勝負（任務三）是模型對局勢發展與多拍連貫性理解的最終體現，但其重要性可能不及關鍵戰術元素，因此權重相對較低，設為 0.2。

