# -*- coding: utf-8 -*-
"""
現場運用シミュレーション用デモデータの生成スクリプト。

架空の訪問介護利用者（U001〜U012）について、
学習済みモデル（models/calibrated_model.joblib）を用いてrisk_scoreを算出し、
それに加えて「データ取得状況」「データ参考度」「確認状況」などの
運用シミュレーション用の属性を付与したCSVを書き出す。

このスクリプトはモデルの再学習を行わない。
既にscripts/export_streamlit_assets.pyで書き出し済みのモデル・
特徴量順序・デモ用基準値をそのまま読み込んで使用する。

実行方法:
    python3 scripts/generate_operations_demo.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
DATA_DIR = REPO_ROOT / "data"

FEATURE_STATUS_OPTIONS = [
    "実測・確認済み",
    "記録から抽出・職員確認済み",
    "記録から抽出・未確認",
    "未取得",
]

MAIN_FEATURES = [
    "MMSE",
    "ADL",
    "FunctionalAssessment",
    "MemoryComplaints",
    "BehavioralProblems",
    "PhysicalActivity",
]


def compute_data_reference_level(completeness_pct: float, last_updated_days: int, confirmed_by_staff: bool) -> str:
    """データ参考度を算出する簡易ルール（デモ用の業務指標であり、医学的な信頼度指標ではない）。

    - 参考外: 重要情報が著しく不足している（充足率30%未満）
    - 高    : 充足率80%以上 かつ 更新30日以内 かつ 主要情報が職員確認済み
    - 中    : 充足率50〜79%、または更新31〜90日
    - 低    : 上記以外（充足率50%未満・更新90日超・未確認情報が多い、など）
    """
    if completeness_pct < 30:
        return "参考外"
    if completeness_pct >= 80 and last_updated_days <= 30 and confirmed_by_staff:
        return "高"
    if (50 <= completeness_pct < 80) or (31 <= last_updated_days <= 90):
        return "中"
    return "低"


def build_row(model, feature_columns, feature_defaults, main_values: dict) -> float:
    row = dict(feature_defaults)
    row.update(main_values)
    df = pd.DataFrame([row], columns=feature_columns)
    return float(model.predict_proba(df)[:, 1][0])


def classify_priority(score: float, selected_threshold: float, high_risk_threshold: float) -> str:
    if score >= high_risk_threshold:
        return "High"
    elif score >= selected_threshold:
        return "Medium"
    return "Low"


# 各利用者の「現在」「前回」の主要項目値は、実際に学習済みモデルへ入力して
# Low/Medium/Highの各区分に確実に収まることを事前に確認した組み合わせを使用している
# （RandomForestは入力の組み合わせによりスコアが非線形に変化するため、
#   値を仮決めするのではなく、モデル出力を確認したうえで採用している）。
# statuses の順序は MAIN_FEATURES と対応: [MMSE, ADL, FunctionalAssessment, MemoryComplaints, BehavioralProblems, PhysicalActivity]
USERS = [
    {
        "user_id": "U001",
        "current": {"MMSE": 27, "ADL": 8, "FunctionalAssessment": 8, "PhysicalActivity": 6, "MemoryComplaints": 0, "BehavioralProblems": 0},
        "previous": {"MMSE": 26, "ADL": 7.8, "FunctionalAssessment": 7.6, "PhysicalActivity": 5.6, "MemoryComplaints": 0, "BehavioralProblems": 0},
        "statuses": ["実測・確認済み"] * 6,
        "last_updated_days": 7, "confirmed_by_staff": True, "review_status": "確認済み",
        "main_reason": "定期確認の対象（大きな変化なし）", "main_source": "訪問介護計画書",
    },
    {
        "user_id": "U002",
        "current": {"MMSE": 18, "ADL": 6, "FunctionalAssessment": 5, "PhysicalActivity": 4, "MemoryComplaints": 1, "BehavioralProblems": 0},
        "previous": {"MMSE": 26, "ADL": 7.8, "FunctionalAssessment": 7.6, "PhysicalActivity": 5.6, "MemoryComplaints": 0, "BehavioralProblems": 0},
        "statuses": ["記録から抽出・職員確認済み", "記録から抽出・職員確認済み", "未取得",
                     "記録から抽出・職員確認済み", "記録から抽出・未確認", "未取得"],
        "last_updated_days": 45, "confirmed_by_staff": False, "review_status": "確認中",
        "main_reason": "もの忘れの訴えが新たに記録された", "main_source": "モニタリング記録",
    },
    {
        "user_id": "U003",
        "current": {"MMSE": 8, "ADL": 2, "FunctionalAssessment": 1.5, "PhysicalActivity": 2, "MemoryComplaints": 1, "BehavioralProblems": 1},
        "previous": {"MMSE": 18, "ADL": 6, "FunctionalAssessment": 5, "PhysicalActivity": 4, "MemoryComplaints": 1, "BehavioralProblems": 0},
        "statuses": ["実測・確認済み", "実測・確認済み", "実測・確認済み",
                     "実測・確認済み", "実測・確認済み", "記録から抽出・職員確認済み"],
        "last_updated_days": 4, "confirmed_by_staff": True, "review_status": "未確認",
        "main_reason": "生活機能・認知面の低下が継続", "main_source": "訪問看護報告書",
    },
    {
        "user_id": "U004",
        "current": {"MMSE": 20, "ADL": 7, "FunctionalAssessment": 1, "PhysicalActivity": 2, "MemoryComplaints": 1, "BehavioralProblems": 0},
        "previous": {"MMSE": 24, "ADL": 4, "FunctionalAssessment": 5, "PhysicalActivity": 4, "MemoryComplaints": 1, "BehavioralProblems": 1},
        "statuses": ["記録から抽出・未確認", "記録から抽出・未確認", "未取得", "未取得", "未取得", "未取得"],
        "last_updated_days": 110, "confirmed_by_staff": False, "review_status": "未確認",
        "main_reason": "行動面の変化の記載があるが未確認情報が多い", "main_source": "日々の介護記録",
    },
    {
        "user_id": "U005",
        "current": {"MMSE": 28, "ADL": 8.6, "FunctionalAssessment": 8.4, "PhysicalActivity": 6.4, "MemoryComplaints": 0, "BehavioralProblems": 0},
        "previous": {"MMSE": 28, "ADL": 8.6, "FunctionalAssessment": 8.4, "PhysicalActivity": 6.4, "MemoryComplaints": 0, "BehavioralProblems": 0},
        "statuses": ["記録から抽出・職員確認済み", "記録から抽出・未確認", "記録から抽出・未確認",
                     "記録から抽出・職員確認済み", "未取得", "未取得"],
        "last_updated_days": 70, "confirmed_by_staff": False, "review_status": "確認済み",
        "main_reason": "定期確認の対象（大きな変化なし）", "main_source": "ケアプラン",
    },
    {
        "user_id": "U006",
        "current": {"MMSE": 18, "ADL": 5, "FunctionalAssessment": 2, "PhysicalActivity": 6, "MemoryComplaints": 0, "BehavioralProblems": 0},
        "previous": {"MMSE": 18, "ADL": 5, "FunctionalAssessment": 2, "PhysicalActivity": 6, "MemoryComplaints": 0, "BehavioralProblems": 0},
        "statuses": ["未取得", "未取得", "記録から抽出・未確認", "未取得", "未取得", "未取得"],
        "last_updated_days": 150, "confirmed_by_staff": False, "review_status": "未確認",
        "main_reason": "取得情報が少なく評価保留", "main_source": "既往歴",
    },
    {
        "user_id": "U007",
        "current": {"MMSE": 14, "ADL": 8, "FunctionalAssessment": 2, "PhysicalActivity": 4, "MemoryComplaints": 1, "BehavioralProblems": 1},
        "previous": {"MMSE": 24, "ADL": 4, "FunctionalAssessment": 5, "PhysicalActivity": 4, "MemoryComplaints": 1, "BehavioralProblems": 1},
        "statuses": ["記録から抽出・職員確認済み", "記録から抽出・職員確認済み", "未取得",
                     "記録から抽出・職員確認済み", "記録から抽出・未確認", "未取得"],
        "last_updated_days": 55, "confirmed_by_staff": True, "review_status": "確認中",
        "main_reason": "行動上の課題が新たに記録された", "main_source": "サービス提供責任者記録",
    },
    {
        "user_id": "U008",
        "current": {"MMSE": 27, "ADL": 8, "FunctionalAssessment": 8, "PhysicalActivity": 6, "MemoryComplaints": 0, "BehavioralProblems": 0},
        "previous": {"MMSE": 27, "ADL": 8, "FunctionalAssessment": 8, "PhysicalActivity": 6, "MemoryComplaints": 0, "BehavioralProblems": 0},
        "statuses": ["実測・確認済み"] * 6,
        "last_updated_days": 3, "confirmed_by_staff": True, "review_status": "確認済み",
        "main_reason": "定期確認の対象（大きな変化なし）", "main_source": "訪問介護計画書",
    },
    {
        "user_id": "U009",
        "current": {"MMSE": 22, "ADL": 4, "FunctionalAssessment": 5, "PhysicalActivity": 2, "MemoryComplaints": 0, "BehavioralProblems": 0},
        "previous": {"MMSE": 12, "ADL": 4, "FunctionalAssessment": 1, "PhysicalActivity": 2, "MemoryComplaints": 0, "BehavioralProblems": 0},
        "statuses": ["記録から抽出・未確認", "未取得", "未取得", "記録から抽出・未確認", "未取得", "未取得"],
        "last_updated_days": 95, "confirmed_by_staff": False, "review_status": "未確認",
        "main_reason": "リハビリ後、生活機能がやや改善", "main_source": "モニタリング記録",
    },
    {
        "user_id": "U010",
        "current": {"MMSE": 8, "ADL": 3, "FunctionalAssessment": 2, "PhysicalActivity": 6, "MemoryComplaints": 1, "BehavioralProblems": 0},
        "previous": {"MMSE": 18, "ADL": 6, "FunctionalAssessment": 5, "PhysicalActivity": 4, "MemoryComplaints": 1, "BehavioralProblems": 0},
        "statuses": ["実測・確認済み"] * 6,
        "last_updated_days": 2, "confirmed_by_staff": True, "review_status": "未確認",
        "main_reason": "短期間で生活機能・認知面が大きく低下", "main_source": "訪問看護報告書",
    },
    {
        "user_id": "U011",
        "current": {"MMSE": 22, "ADL": 4, "FunctionalAssessment": 5, "PhysicalActivity": 2, "MemoryComplaints": 0, "BehavioralProblems": 0},
        "previous": {"MMSE": 18, "ADL": 5, "FunctionalAssessment": 2, "PhysicalActivity": 6, "MemoryComplaints": 0, "BehavioralProblems": 0},
        "statuses": ["記録から抽出・未確認", "記録から抽出・未確認", "記録から抽出・未確認",
                     "記録から抽出・未確認", "未取得", "未取得"],
        "last_updated_days": 40, "confirmed_by_staff": False, "review_status": "未確認",
        "main_reason": "もの忘れの訴えが継続して記録されている", "main_source": "日々の介護記録",
    },
    {
        "user_id": "U012",
        "current": {"MMSE": 12, "ADL": 4, "FunctionalAssessment": 1, "PhysicalActivity": 2, "MemoryComplaints": 0, "BehavioralProblems": 0},
        "previous": {"MMSE": 12, "ADL": 4, "FunctionalAssessment": 1, "PhysicalActivity": 2, "MemoryComplaints": 0, "BehavioralProblems": 0},
        "statuses": ["未取得", "未取得", "未取得", "記録から抽出・未確認", "未取得", "未取得"],
        "last_updated_days": 180, "confirmed_by_staff": False, "review_status": "未確認",
        "main_reason": "長期間データ更新なし", "main_source": "既往歴",
    },
]


def main():
    model = joblib.load(MODELS_DIR / "calibrated_model.joblib")
    with open(MODELS_DIR / "feature_columns.json", encoding="utf-8") as f:
        feature_columns = json.load(f)
    with open(MODELS_DIR / "feature_defaults.json", encoding="utf-8") as f:
        feature_defaults = json.load(f)
    with open(MODELS_DIR / "model_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    selected_threshold = metadata["thresholds"]["selected_threshold"]
    high_risk_threshold = metadata["thresholds"]["high_risk_threshold"]

    rows = []
    for u in USERS:
        risk_score = build_row(model, feature_columns, feature_defaults, u["current"])
        previous_risk_score = build_row(model, feature_columns, feature_defaults, u["previous"])
        risk_change = risk_score - previous_risk_score
        priority_level = classify_priority(risk_score, selected_threshold, high_risk_threshold)

        completeness_pct = round(
            100 * sum(1 for s in u["statuses"] if s != "未取得") / len(u["statuses"])
        )
        data_reference_level = compute_data_reference_level(
            completeness_pct, u["last_updated_days"], u["confirmed_by_staff"]
        )

        row = {
            "user_id": u["user_id"],
            "risk_score": round(risk_score, 4),
            "previous_risk_score": round(previous_risk_score, 4),
            "risk_change": round(risk_change, 4),
            "priority_level": priority_level,
            "data_completeness": completeness_pct,
            "last_updated_days": u["last_updated_days"],
            "data_reference_level": data_reference_level,
            "review_status": u["review_status"],
            "main_reason": u["main_reason"],
            "main_source": u["main_source"],
            "confirmed_by_staff": u["confirmed_by_staff"],
        }
        for feat, status in zip(MAIN_FEATURES, u["statuses"]):
            row[f"{feat}_status"] = status
        rows.append(row)

    ops_df = pd.DataFrame(rows)
    ops_df = ops_df.sort_values("risk_score", ascending=False).reset_index(drop=True)

    out_path = DATA_DIR / "operations_demo.csv"
    ops_df.to_csv(out_path, index=False)

    print("=== 現場運用シミュレーション用デモデータを書き出しました ===")
    print(out_path.relative_to(REPO_ROOT))
    print(ops_df[["user_id", "risk_score", "priority_level", "data_reference_level", "review_status"]])


if __name__ == "__main__":
    main()
