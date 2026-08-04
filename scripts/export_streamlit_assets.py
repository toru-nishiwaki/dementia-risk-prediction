# -*- coding: utf-8 -*-
"""
Streamlitデモ用アセットの書き出しスクリプト。

notebook/dementia_risk_prediction.ipynb（および src/dementia_risk_prediction.py）の
「モデル①：高精度モデル」の分析パイプラインを、既存の前処理・学習・閾値選定ロジックを
変更せずにそのまま再現し、Streamlitアプリが読み込む学習済みアセットを書き出す。

このスクリプトは既存の分析結果を変更しない。あくまで、既存Notebookと同じ手順で
モデルを再学習し、その出力（モデル本体・評価指標・特徴量重要度など）を
Streamlitアプリから読み込める形式で保存するためのものである。

実行方法:
    python3 scripts/export_streamlit_assets.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "alzheimers_disease_data.csv"
MODELS_DIR = REPO_ROOT / "models"
OUTPUTS_DIR = REPO_ROOT / "outputs"
DATA_DIR = REPO_ROOT / "data"

RANDOM_STATE = 42
RECALL_CONSTRAINT = 0.85
HIGH_RISK_THRESHOLD = 0.7

# アプリで操作可能にするメイン入力項目
MAIN_INPUT_FEATURES = [
    "MMSE",
    "ADL",
    "FunctionalAssessment",
    "PhysicalActivity",
    "MemoryComplaints",
    "BehavioralProblems",
]

# 元データにおける各主要項目の値域（範囲外入力の防止に使用）
FEATURE_RANGES = {
    "MMSE": (0, 30),
    "ADL": (0, 10),
    "FunctionalAssessment": (0, 10),
    "PhysicalActivity": (0, 10),
    "MemoryComplaints": (0, 1),
    "BehavioralProblems": (0, 1),
}

BINARY_FEATURES = [
    "Gender",
    "Smoking",
    "FamilyHistoryAlzheimers",
    "CardiovascularDisease",
    "Diabetes",
    "Depression",
    "HeadInjury",
    "Hypertension",
    "MemoryComplaints",
    "BehavioralProblems",
    "Confusion",
    "Disorientation",
    "PersonalityChanges",
    "DifficultyCompletingTasks",
    "Forgetfulness",
]

CATEGORICAL_FEATURES = ["Gender", "Ethnicity", "EducationLevel"]


def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)

    # 既存分析（src/dementia_risk_prediction.py）と同一の前処理
    df_encoded = pd.get_dummies(
        df, columns=["Gender", "Ethnicity", "EducationLevel"], drop_first=True
    )
    df_encoded = df_encoded.drop(["PatientID", "DoctorInCharge"], axis=1)

    X = df_encoded.drop(columns=["Diagnosis"])
    y = df_encoded["Diagnosis"]

    return df, X, y


def build_feature_defaults(df_raw: pd.DataFrame, feature_columns: list) -> dict:
    """デモ用基準値を作成する。

    連続値は中央値、二値項目は最頻値を用いる。Gender/Ethnicity/EducationLevelは
    One-Hot化の整合性を保つため、元データの最頻カテゴリをそのままエンコードする。
    """
    defaults_raw = {}
    for col in df_raw.columns:
        if col in ("PatientID", "DoctorInCharge", "Diagnosis"):
            continue
        if col in CATEGORICAL_FEATURES:
            defaults_raw[col] = df_raw[col].mode().iloc[0]
        elif col in BINARY_FEATURES:
            defaults_raw[col] = int(df_raw[col].mode().iloc[0])
        else:
            defaults_raw[col] = float(df_raw[col].median())

    baseline_row = pd.DataFrame([defaults_raw])
    baseline_encoded = pd.get_dummies(
        baseline_row, columns=["Gender", "Ethnicity", "EducationLevel"], drop_first=True
    )

    # 学習データの列と揃える（欠けるダミー列は0で補完）
    baseline_encoded = baseline_encoded.reindex(columns=feature_columns, fill_value=0)

    defaults = {}
    for col in feature_columns:
        val = baseline_encoded.iloc[0][col]
        if col in MAIN_INPUT_FEATURES and col not in ("MemoryComplaints", "BehavioralProblems"):
            defaults[col] = float(val)
        else:
            defaults[col] = int(val) if float(val).is_integer() else float(val)

    return defaults


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    df_raw, X, y = load_and_preprocess()
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
    )

    # === モデル①：高精度モデル（特徴量重要度の算出に使用）===
    model_rf = RandomForestClassifier(random_state=RANDOM_STATE)
    model_rf.fit(X_train, y_train)

    # === 交差検証（既存分析と同一設定）===
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        RandomForestClassifier(random_state=RANDOM_STATE),
        X_train,
        y_train,
        cv=cv,
        scoring=["accuracy", "recall", "precision", "f1", "roc_auc"],
        return_train_score=False,
    )
    cv_summary = {
        "accuracy": float(np.mean(cv_results["test_accuracy"])),
        "recall": float(np.mean(cv_results["test_recall"])),
        "precision": float(np.mean(cv_results["test_precision"])),
        "f1": float(np.mean(cv_results["test_f1"])),
        "roc_auc": float(np.mean(cv_results["test_roc_auc"])),
    }

    # === キャリブレーション済みモデル（アプリの予測に使用）===
    y_proba_raw = model_rf.predict_proba(X_test)[:, 1]

    calibrated_model = CalibratedClassifierCV(
        RandomForestClassifier(random_state=RANDOM_STATE), method="sigmoid", cv=3
    )
    calibrated_model.fit(X_train, y_train)
    y_proba = calibrated_model.predict_proba(X_test)[:, 1]

    brier_before = float(brier_score_loss(y_test, y_proba_raw))
    brier_after = float(brier_score_loss(y_test, y_proba))

    test_roc_auc = float(roc_auc_score(y_test, y_proba))

    # === 閾値ごとの評価比較（既存分析と同一ロジック）===
    threshold_list = np.arange(0.1, 0.9, 0.05)
    rows = []
    for th in threshold_list:
        y_pred_th = (y_proba >= th).astype(int)
        report = classification_report(y_test, y_pred_th, output_dict=True, zero_division=0)
        rows.append(
            {
                "threshold": round(float(th), 2),
                "accuracy": report["accuracy"],
                "precision_1": report["1"]["precision"],
                "recall_1": report["1"]["recall"],
                "f1_1": report["1"]["f1-score"],
            }
        )
    threshold_df = pd.DataFrame(rows)

    # Recall制約付き最適閾値（既存分析と同一ロジック：SELECTED_THRESHOLDの根拠）
    candidate_df = threshold_df[threshold_df["recall_1"] >= RECALL_CONSTRAINT].copy()
    best_row = candidate_df.sort_values("f1_1", ascending=False).head(1)
    selected_threshold = float(best_row["threshold"].values[0])

    # === 採用閾値での最終評価 ===
    y_pred_selected = (y_proba >= selected_threshold).astype(int)
    selected_report = classification_report(
        y_test, y_pred_selected, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred_selected)

    # === 特徴量重要度（モデル①・全体重要度）===
    importances = model_rf.feature_importances_
    feature_importance_df = pd.DataFrame(
        {"feature": feature_columns, "importance": importances}
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    # === デモ用基準値・特徴量順序 ===
    feature_defaults = build_feature_defaults(df_raw, feature_columns)

    # === デモケース ===
    # 主要項目以外はデモ用基準値（feature_defaults）に固定した状態で、
    # アプリと同じ計算方法によりリスクスコアを算出する。
    # 元データの行をそのまま抽出すると、非表示項目（隠れた特徴量）の値の違いにより
    # 「MMSEが高いのにHigh判定」のような直感に反する組み合わせが生じ得るため、
    # 3区分（Low/Medium/High）を明確に代表する主要項目の値を設計し、
    # 学習済みモデルでスコアを算出する方式を採用する。
    demo_case_inputs = [
        (
            "case_1",
            "状態が比較的安定した例",
            {"MMSE": 28, "ADL": 8.5, "FunctionalAssessment": 8.5, "PhysicalActivity": 6.0,
             "MemoryComplaints": 0, "BehavioralProblems": 0},
        ),
        (
            "case_2",
            "経過観察が必要な例",
            {"MMSE": 18, "ADL": 6.0, "FunctionalAssessment": 5.0, "PhysicalActivity": 4.0,
             "MemoryComplaints": 1, "BehavioralProblems": 0},
        ),
        (
            "case_3",
            "優先確認が必要な例",
            {"MMSE": 8, "ADL": 2.0, "FunctionalAssessment": 1.5, "PhysicalActivity": 2.0,
             "MemoryComplaints": 1, "BehavioralProblems": 1},
        ),
    ]

    def build_case_row(main_values: dict) -> pd.DataFrame:
        row = dict(feature_defaults)
        row.update(main_values)
        return pd.DataFrame([row], columns=feature_columns)

    def risk_level_of(score_value: float) -> str:
        if score_value >= HIGH_RISK_THRESHOLD:
            return "High"
        elif score_value >= selected_threshold:
            return "Medium"
        return "Low"

    demo_case_records = []
    for case_id, case_name, main_values in demo_case_inputs:
        case_row = build_case_row(main_values)
        case_score = float(calibrated_model.predict_proba(case_row)[:, 1][0])
        record = {"case_id": case_id, "case_name": case_name}
        record.update(main_values)
        record["reference_risk_score"] = round(case_score, 4)
        record["reference_risk_level"] = risk_level_of(case_score)
        demo_case_records.append(record)

    demo_cases = pd.DataFrame(demo_case_records)

    # ===================== ファイル書き出し =====================

    joblib.dump(calibrated_model, MODELS_DIR / "calibrated_model.joblib")

    with open(MODELS_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=2)

    with open(MODELS_DIR / "feature_defaults.json", "w", encoding="utf-8") as f:
        json.dump(feature_defaults, f, ensure_ascii=False, indent=2)

    metadata = {
        "model_description": {
            "ja": "RandomForestClassifierをCalibratedClassifierCV（sigmoid, cv=3）で確率校正したモデル",
            "base_estimator": "RandomForestClassifier(random_state=42)",
            "calibration_method": "sigmoid",
            "calibration_cv": 3,
        },
        "dataset": {
            "source": "Kaggle: Alzheimer's Disease Dataset (alzheimers_disease_data.csv)",
            "n_samples": int(len(df_raw)),
            "n_features_raw": int(df_raw.shape[1] - 3),  # PatientID, DoctorInCharge, Diagnosisを除く
            "n_features_encoded": int(len(feature_columns)),
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
        },
        "cross_validation_5fold": cv_summary,
        "test_set_metrics": {
            "roc_auc": test_roc_auc,
            "at_selected_threshold": {
                "threshold": selected_threshold,
                "accuracy": selected_report["accuracy"],
                "precision_1": selected_report["1"]["precision"],
                "recall_1": selected_report["1"]["recall"],
                "f1_1": selected_report["1"]["f1-score"],
            },
        },
        "calibration": {
            "brier_score_before": brier_before,
            "brier_score_after": brier_after,
        },
        "thresholds": {
            "selected_threshold": selected_threshold,
            "selection_rule": "Recall >= 0.85 の候補の中でF1-scoreが最大となる閾値",
            "high_risk_threshold": HIGH_RISK_THRESHOLD,
            "high_risk_threshold_rule": "優先対応ラインとして運用上設定した固定値",
        },
        "main_input_features": MAIN_INPUT_FEATURES,
        "feature_ranges": FEATURE_RANGES,
        "main_input_note": "このデモでは主要項目を操作し、その他の項目はデモ用基準値に固定しています。",
        "disclaimer": {
            "not_diagnostic": "医療診断を目的としたものではありません。",
            "final_decision": "最終判断は専門職が行います。",
            "data_note": "架空または公開データを使用しています。",
        },
    }
    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    feature_importance_df.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)
    threshold_df.to_csv(OUTPUTS_DIR / "threshold_metrics.csv", index=False)

    cm_df = pd.DataFrame(
        cm,
        index=["actual_0_negative", "actual_1_positive"],
        columns=["pred_0_negative", "pred_1_positive"],
    )
    cm_df.to_csv(OUTPUTS_DIR / "confusion_matrix.csv")

    demo_cases.to_csv(DATA_DIR / "demo_cases.csv", index=False)

    # 検証用データ（テスト分割のy_true / risk_score）をそのまま書き出す。
    # 再学習やデータ分割の変更は行わず、既に算出済みのy_test・y_probaを保存するのみ。
    # 「閾値と確認業務量」画面で、任意の閾値における実際の見逃し・誤検知件数を
    # 算出するために使用する。
    validation_df = pd.DataFrame({
        "y_true": y_test.values,
        "risk_score": y_proba,
    })
    validation_df.to_csv(OUTPUTS_DIR / "validation_predictions.csv", index=False)

    print("=== 書き出し完了 ===")
    print("採用閾値 (selected_threshold):", selected_threshold)
    print("高リスク閾値 (high_risk_threshold):", HIGH_RISK_THRESHOLD)
    print("ROC-AUC (test):", round(test_roc_auc, 4))
    print("Recall (selected threshold):", round(selected_report["1"]["recall"], 4))
    print("Precision (selected threshold):", round(selected_report["1"]["precision"], 4))
    print("F1 (selected threshold):", round(selected_report["1"]["f1-score"], 4))
    print("Confusion Matrix:\n", cm)
    print("\n書き出し先:")
    for p in [
        MODELS_DIR / "calibrated_model.joblib",
        MODELS_DIR / "feature_columns.json",
        MODELS_DIR / "feature_defaults.json",
        MODELS_DIR / "model_metadata.json",
        OUTPUTS_DIR / "feature_importance.csv",
        OUTPUTS_DIR / "threshold_metrics.csv",
        OUTPUTS_DIR / "confusion_matrix.csv",
        OUTPUTS_DIR / "validation_predictions.csv",
        DATA_DIR / "demo_cases.csv",
    ]:
        print(" -", p.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
