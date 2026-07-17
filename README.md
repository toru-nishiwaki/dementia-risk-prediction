# 認知症リスク予測モデル｜介護現場データを活用した介入優先度判定

訪問介護職としての現場経験を起点に、データ分析・機械学習を用いて
「認知症リスクの早期検知」と「介入優先度の判定」を検討したポートフォリオです。

## 概要

Kaggleの認知症関連データセットを用い、RandomForestClassifierによる
リスク予測モデルを構築しました。単純な予測精度の追求ではなく、

- 見逃し（False Negative）を抑えるための**閾値調整**
- 予測確率を実態に近づける**確率校正（Calibration）**
- 判断根拠を確認するための**特徴量重要度**分析
- リスクスコアを介入優先度（High / Medium / Low）に変換する**意思決定ルールの設計**

に重点を置き、現場の意思決定支援に使える形を検討しました。

## 背景・課題

介護現場では、限られた職員数の中で利用者の状態変化を早期に把握し、
適切な介入判断を行う必要があります。しかし現場の判断は経験や主観に
依存しやすく、対応のばらつきや見逃しが生じる可能性があります。

本プロジェクトでは、訪問介護の現場で感じたこの課題意識を出発点に、
リスクを数値化し、介入候補を客観的に検討する仕組みを分析しました。

## 目的

認知症リスクの早期検知と介入優先度の検討を通じて、介護現場における
意思決定を支援することを目的としています。

**注意**：本モデルは診断を行うものではなく、あくまで統計的な
「リスク予測」および「介入優先度の検討」を目的とした分析です。
医学的診断の代替となるものではありません。

## 使用データ

- Kaggle: [Alzheimer's Disease Dataset](https://www.kaggle.com/)（`alzheimers_disease_data.csv`）
- データ数：2,149件
- 特徴量：年齢、生活習慣、認知機能（MMSE、ADL、FunctionalAssessmentなど）
- 目的変数：Diagnosis（0: 健常、1: アルツハイマー）

## 分析の流れ

1. **前処理**：One-Hot Encoding、不要カラムの削除
2. **モデル①（高精度モデル）**：Kaggleデータ全体を用いたRandomForestClassifierによる分類モデルの構築
3. **交差検証**：5分割Stratified K-Foldで基本性能を確認
4. **確率校正（Calibration）**：CalibratedClassifierCVによる予測確率の補正、Brier Scoreで前後比較
5. **特徴量重要度の確認**：どの指標がリスク判定に寄与しているかを可視化
6. **閾値調整**：Recallを重視した閾値選定（ROC曲線・PR曲線・閾値ごとの評価比較）
7. **リスクスコア・リスクレベル分類**：予測確率を High / Medium / Low の3段階に変換
8. **高リスク群の特徴分析**：全体平均との比較による傾向把握
9. **介入シミュレーション**：ADL・FunctionalAssessmentを改善した場合のリスクスコア変化を検証
10. **モデル②（現場実用モデル）**：現場で取得しやすい項目に絞った簡易モデルで閾値0.5と0.4を比較

## 主な結果

| 観点 | 結果 |
|---|---|
| 交差検証（5-fold） | Accuracy 約0.93、Recall 約0.85 |
| 確率校正（Brier Score） | 校正前 0.078 → 校正後 0.054 に改善 |
| 特徴量重要度 | MMSE・ADL・FunctionalAssessmentが上位（介護現場の実感と整合） |
| 閾値調整 | Recall 0.85以上を制約に、F1が最大となる閾値を採用候補として選定（0.35/0.40帯を参考比較） |
| 介入シミュレーション | ADL・FunctionalAssessmentを仮想的に改善 → 平均リスクスコアが0.346→0.096に低下 |
| 現場実用モデル（閾値0.5→0.4） | Recall 0.57→0.63、False Negative 66件→56件に減少（False Positiveは53件→65件に増加） |

Recallを優先する場合はPrecisionとのトレードオフが生じるため、
本分析では「見逃し防止」と「誤検知の抑制」のバランスを踏まえたうえで、
MMSEとリスクスコアを組み合わせた段階的な介入優先度（優先介入／介入／非介入）を設計しています。

## ビジネス上の示唆

- **利用者側**：認知機能低下の早期検知により、介入を前倒しできる可能性
- **職員側**：リスク評価を数値で標準化することで、判断のばらつきや心理的負担を軽減できる可能性
- **経営側**：重症化予防を通じて、入院・退所リスクの低減や人的リソースの最適配分に寄与する可能性

いずれも仮説段階であり、実際の効果検証には現場データによる追加検証が必要です。

## 使用技術

- Python（pandas, numpy）
- scikit-learn（RandomForestClassifier, CalibratedClassifierCV, 各種評価指標）
- matplotlib, seaborn（可視化）
- Google Colab / Jupyter Notebook

## ファイル構成

```
dementia-risk-prediction/
├── README.md
├── notebook/
│   └── dementia_risk_prediction.ipynb   # 分析本体（Notebook形式）
├── src/
│   └── dementia_risk_prediction.py      # 分析本体（スクリプト形式）
└── docs/
    └── dementia_risk_prediction_slides.pdf  # 分析サマリー資料（スライド）
```

## Notebook・資料へのリンク

- 分析Notebook：[notebook/dementia_risk_prediction.ipynb](notebook/dementia_risk_prediction.ipynb)
- 分析スクリプト：[src/dementia_risk_prediction.py](src/dementia_risk_prediction.py)
- サマリー資料（PDF）：[docs/dementia_risk_prediction_slides.pdf](docs/dementia_risk_prediction_slides.pdf)

## 位置づけ

本プロジェクトは、訪問介護職として培った現場感覚と、データ分析・AI活用の
スキルを接続することを目的とした個人ポートフォリオです。ITコンサルタントへの
キャリアチェンジにあたり、「現場課題の理解」と「データに基づく意思決定支援の設計」の
両方に取り組めることを示すことを狙いとしています。
