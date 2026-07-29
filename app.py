# -*- coding: utf-8 -*-
"""
介護記録から状態変化と確認優先度を整理するデモ
Care Record Review & Priority Support Demo

採用担当者向けの軽量Streamlitデモ。既存の分析（notebook/, src/）で構築した
キャリブレーション済みモデルを「補助的な分析エンジン」として残しつつ、
訪問介護計画書・日々の介護記録・モニタリングなどから状態変化と根拠文章を整理し、
確認優先度を判定する意思決定支援PoCとして再構成したもの。

このアプリは医療診断アプリではありません。
AIが診断や支援内容を確定するものではなく、抽出結果と候補を職員・専門職が
確認して判断します。
"""

import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# 基本設定
# ============================================================

st.set_page_config(
    page_title="介護記録CSVから状態変化と確認優先度を整理するPoC",
    page_icon="🧭",
    layout="wide",
)

REPO_ROOT = Path(__file__).resolve().parent
MODELS_DIR = REPO_ROOT / "models"
OUTPUTS_DIR = REPO_ROOT / "outputs"
DATA_DIR = REPO_ROOT / "data"

REQUIRED_FILES = {
    "モデル本体": MODELS_DIR / "calibrated_model.joblib",
    "特徴量の並び順": MODELS_DIR / "feature_columns.json",
    "デモ用基準値": MODELS_DIR / "feature_defaults.json",
    "モデル評価指標": MODELS_DIR / "model_metadata.json",
    "特徴量重要度": OUTPUTS_DIR / "feature_importance.csv",
    "閾値ごとの評価比較": OUTPUTS_DIR / "threshold_metrics.csv",
    "混同行列": OUTPUTS_DIR / "confusion_matrix.csv",
    "検証用予測データ": OUTPUTS_DIR / "validation_predictions.csv",
    "デモケース": DATA_DIR / "demo_cases.csv",
    "現場運用シミュレーション用データ": DATA_DIR / "operations_demo.csv",
    "介護記録デモデータ": DATA_DIR / "care_record_demo.csv",
    "列名マッピング定義": DATA_DIR / "column_aliases.json",
    "サンプルCSV（利用者基本情報）": DATA_DIR / "sample_users.csv",
    "サンプルCSV（訪問介護計画書）": DATA_DIR / "sample_care_plans.csv",
    "サンプルCSV（日々の介護記録）": DATA_DIR / "sample_daily_records.csv",
    "サンプルCSV（モニタリング記録）": DATA_DIR / "sample_monitoring_records.csv",
}

MAIN_FEATURE_LABELS = {
    "MMSE": "MMSE（認知機能検査スコア）",
    "ADL": "ADL（日常生活動作スコア）",
    "FunctionalAssessment": "FunctionalAssessment（生活機能評価）",
    "PhysicalActivity": "PhysicalActivity（身体活動量）",
    "MemoryComplaints": "MemoryComplaints（もの忘れの訴え）",
    "BehavioralProblems": "BehavioralProblems（行動上の問題）",
}

BINARY_MAIN_FEATURES = ("MemoryComplaints", "BehavioralProblems")

FEATURE_JP_MEANING = {
    "FunctionalAssessment": "生活機能評価",
    "ADL": "日常生活動作",
    "MMSE": "認知機能評価",
    "MemoryComplaints": "記憶に関する訴え",
    "BehavioralProblems": "行動上の課題",
    "PhysicalActivity": "身体活動",
}

ACTION_CANDIDATES = {
    "Low": [
        "定期的な状態確認を継続",
        "新たな生活機能の変化がないか記録を確認",
    ],
    "Medium": [
        "直近の生活機能・認知面の変化を追加確認",
        "根拠となる日々の記録を確認",
        "サービス提供責任者やケアマネジャーへの共有を検討",
    ],
    "High": [
        "専門職による優先的な確認",
        "ケアプラン、服薬、安全面、生活機能の変化を確認",
        "必要に応じて看護職・医療職を含む関係者への共有を検討",
    ],
}

PRIORITY_LABELS = {"Low": "Low（低）", "Medium": "Medium（中）", "High": "High（高）"}

# 運用上の確認区分（系統A・系統B共通で使用する3区分）
OPERATIONAL_CATEGORIES = ["優先確認", "追加情報確認", "経過観察"]
OPERATIONAL_CATEGORY_ORDER = {name: i for i, name in enumerate(OPERATIONAL_CATEGORIES)}

# 「データ参考度」という名称は使用せず、直接的な表記に置き換える表示用マップ
REFERENCE_DISPLAY_MAP = {
    "高": "情報は十分",
    "中": "一部不足",
    "低": "更新が必要",
    "参考外": "判断材料不足",
}


def compute_operational_category(risk_level: str, data_reference_level: str) -> str:
    """系統A（構造化評価値によるモデルスコア）のみで運用区分を判定するデモ用ルール。

    入力情報の状態（旧データ参考度）を踏まえ、モデルスコアだけで機械的に
    優先確認と断定しない。確認状況（未確認／確認中／確認済み）は作業の進捗を
    示す別軸であり、この判定には影響させない。医学的判断ではない。
    """
    if data_reference_level == "参考外":
        return "追加情報確認"
    if risk_level == "High":
        return "優先確認" if data_reference_level in ("高", "中") else "追加情報確認"
    return "経過観察"


LIMITATIONS = [
    "Kaggle公開データを用いた分析PoCである",
    "実在する訪問介護事業所での外部検証は未実施である",
    "実際の訪問介護現場では構造化評価値を取得しにくい",
    "モデル上のスコアは将来の発症確率を断定するものではない",
    "入力情報の状態（旧データ参考度）は医学的な信頼度ではない",
    "実運用には専門職監修、データ品質管理、追加検証が必要",
]

TEAL = "#0d9488"
TEAL_DARK = "#0f766e"


# ============================================================
# 系統B：介護記録からの観察事項抽出（キーワード・ルールベース／外部LLM APIなし）
# ============================================================

EXTRACTION_RULES = [
    {"category": "服薬管理", "keywords": [
        "薬が数日分残っ", "薬が残っ", "服薬したかどうか", "服薬したか分から",
        "服薬状況の確認が必要", "飲み忘れ",
    ]},
    {"category": "記憶に関する変化", "keywords": [
        "同じ質問を何度も", "もの忘れが増えた", "少し前の出来事を覚えていない", "思い出せない",
    ]},
    {"category": "ADL・IADLの変化", "keywords": [
        "介助が必要だった", "ふらつきが見られた", "食事の準備が難しく", "移動に介助",
    ]},
    {"category": "以前の状態との差", "keywords": ["以前は自分で", "以前は自立"]},
    {"category": "食事・水分", "keywords": ["食事量が減", "水分摂取が少な", "食欲がない"]},
    {"category": "睡眠", "keywords": ["夜間に何度も", "不眠", "眠れない", "昼夜逆転"]},
    {"category": "行動面の変化", "keywords": ["落ち着かない", "興奮", "大声を出す", "徘徊"]},
    {"category": "安全面の懸念", "keywords": [
        "転倒", "火の始末", "外出後に戻れ", "戻れなくな", "ガスの火",
    ]},
]
OBSERVATION_CATEGORIES = [r["category"] for r in EXTRACTION_RULES] + ["情報不足"]
MIN_SUBSTANTIVE_LENGTH = 15


def extract_observations_from_text(text, record_date, source_type, allow_info_insufficient: bool = True) -> list:
    """任意のテキストから観察事項候補を抽出する（ルールベース）。

    どのキーワードがどのカテゴリに対応したかを追跡できるよう、
    一致したキーワードをそのまま結果に含める。ケアプランなど「短くて当然」の
    項目についてはallow_info_insufficient=Falseとし、情報不足タグを付けない。
    """
    text = "" if text is None else str(text)
    if not text.strip() or text.strip().lower() == "nan":
        return []
    observations = []
    for rule in EXTRACTION_RULES:
        matched_keyword = next((kw for kw in rule["keywords"] if kw in text), None)
        if matched_keyword:
            observations.append(
                {
                    "category": rule["category"],
                    "matched_keyword": matched_keyword,
                    "evidence": text,
                    "record_date": record_date,
                    "source_type": source_type,
                }
            )
    if not observations and allow_info_insufficient and len(text.strip()) < MIN_SUBSTANTIVE_LENGTH:
        observations.append(
            {
                "category": "情報不足",
                "matched_keyword": None,
                "evidence": text,
                "record_date": record_date,
                "source_type": source_type,
            }
        )
    return observations


def extract_observations_from_row(row: pd.Series) -> list:
    return extract_observations_from_text(row["record_text"], row["record_date"], row["source_type"])


def extract_all_observations(user_rows: pd.DataFrame) -> list:
    results = []
    for _, row in user_rows.iterrows():
        results.extend(extract_observations_from_row(row))
    return results


def compute_record_state_level_from_texts(texts: list, approved_observations: list) -> str:
    """介護記録上の状態変化レベルを判定するデモ用の単純なルール。

    職員が「採用」または「修正して採用」とした観察事項のみを対象とする。
    医学的に検証されたルールではなく、現場運用を説明するためのものである。
    """
    non_info = [o["category"] for o in approved_observations if o["category"] != "情報不足"]
    if non_info:
        safety_or_medication = any(c in ("安全面の懸念", "服薬管理") for c in non_info)
        if safety_or_medication or len(set(non_info)) >= 2:
            return "高"
        return "中"
    substantive = [t for t in texts if len(str(t).strip()) >= MIN_SUBSTANTIVE_LENGTH]
    if not substantive:
        return "情報不足"
    return "低"


def compute_record_state_level(user_rows: pd.DataFrame, approved_observations: list) -> str:
    texts = user_rows["record_text"].astype(str).tolist()
    return compute_record_state_level_from_texts(texts, approved_observations)


def compute_operational_category_from_tracks(record_state_level: str, model_risk_level) -> str:
    """系統A（モデルスコア）と系統B（記録上の状態変化）を統合した運用区分の判定（サンプルデータ用）。"""
    if record_state_level == "高":
        return "優先確認"
    if model_risk_level == "High":
        return "優先確認"
    if record_state_level == "情報不足":
        return "追加情報確認"
    return "経過観察"


def compute_operational_category_csv(record_state_level: str, model_risk_level, input_status: str) -> str:
    """系統A・系統B・入力情報の状態（記録や計画書の充足度）を統合した運用区分の判定（CSVアップロード用）。

    安全面・服薬管理の懸念や複数カテゴリの変化が確認された場合は優先確認、
    記録や計画書自体が不足している場合は追加情報確認とする。
    医学的判断ではなく、記録確認を標準化するためのデモ用業務ルールである。
    """
    if record_state_level == "高":
        return "優先確認"
    if model_risk_level == "High":
        return "優先確認"
    if input_status == "判断材料不足":
        return "追加情報確認"
    if record_state_level == "情報不足":
        return "追加情報確認"
    return "経過観察"


# 支援・共有候補（職員確認済みの観察カテゴリに基づくルールベース候補）。
# 「次回訪問で確認すること」「事業所内で共有すること」「関係職種への共有を検討すること」の
# 3種類に分けて表示する。Random Forestの重要度とは無関係の、説明可能な固定ルールである。
SHARE_CANDIDATES_BY_CATEGORY = {
    "服薬管理": {
        "visit": ["残薬と服薬手順を確認", "本人が服薬状況をどのように把握しているか確認"],
        "office": ["サービス提供責任者へ記録内容を共有"],
        "external": ["ケアマネジャー", "医療職"],
    },
    "記憶に関する変化": {
        "visit": ["同じ質問や説明理解の変化を確認"],
        "office": ["過去記録と今回記録の差を確認"],
        "external": ["ケアマネジャー"],
    },
    "ADL・IADLの変化": {
        "visit": ["移動・排泄・更衣などの変化を確認"],
        "office": ["現在のケアプランとの整合を確認"],
        "external": ["ケアマネジャー"],
    },
    "以前の状態との差": {
        "visit": ["以前できていた動作との差を確認"],
        "office": ["過去記録と今回記録の差を確認"],
        "external": ["ケアマネジャー"],
    },
    "安全面の懸念": {
        "visit": ["転倒・火の始末・外出時の様子など安全面を優先確認"],
        "office": ["サービス提供責任者へ記録内容を共有"],
        "external": ["看護職", "医療職", "家族"],
    },
    "食事・水分": {
        "visit": ["食事量と水分摂取を確認"],
        "office": ["記録を継続的に共有"],
        "external": ["ケアマネジャー"],
    },
    "睡眠": {
        "visit": ["睡眠状況を確認"],
        "office": ["生活リズムの変化を共有"],
        "external": ["ケアマネジャー"],
    },
    "行動面の変化": {
        "visit": ["行動面の変化が見られた状況・時間帯を確認"],
        "office": ["環境変化の有無を共有"],
        "external": ["ケアマネジャー", "医療職"],
    },
    "情報不足": {
        "visit": ["次回訪問時の記録を厚くする"],
        "office": ["根拠となる情報源を追加で確認"],
        "external": [],
    },
}


def get_share_candidates_grouped(categories: list):
    seen, visit, office, external = [], [], [], []
    for c in categories:
        if c in seen:
            continue
        seen.append(c)
        d = SHARE_CANDIDATES_BY_CATEGORY.get(c, {})
        for item in d.get("visit", []):
            if item not in visit:
                visit.append(item)
        for item in d.get("office", []):
            if item not in office:
                office.append(item)
        for item in d.get("external", []):
            if item not in external:
                external.append(item)
    return visit, office, external


# やさしい日本語表示用の固定テンプレート（外部翻訳APIは使用しない）
EASY_JAPANESE_VISIT = {
    "服薬管理": "つぎの訪問で、薬が残っていないか見ます。薬をどう飲んでいるか、本人に聞きます。",
    "記憶に関する変化": "同じ質問がないか、次の訪問で見ます。日にちや予定が分かっているか、本人と話します。",
    "ADL・IADLの変化": "着替えや歩く様子、食事の準備が前とちがうか見ます。",
    "以前の状態との差": "前とくらべて、できることが変わっていないか見ます。",
    "安全面の懸念": "転ぶこと、火のあつかい、外出のことを、注意して見ます。",
    "食事・水分": "ごはんと水をどれくらい取っているか見ます。",
    "睡眠": "夜、よく眠れているか聞きます。",
    "行動面の変化": "いつもとちがう様子がないか、時間や場所もあわせて見ます。",
    "情報不足": "次の訪問で、もう少しくわしく記録します。",
}
EASY_JAPANESE_OFFICE = "じむしょの人に、この記録をつたえます。"
EASY_JAPANESE_EXTERNAL = "かんけいする人（ケアマネジャーなど）と そうだんすることを かんがえます。"


def render_share_candidates_section(approved_observations: list, key_suffix: str):
    """支援・共有候補を3区分＋やさしい日本語切替つきで表示する。"""
    approved_categories = [o["category"] for o in approved_observations]
    visit, office, external = get_share_candidates_grouped(approved_categories)

    st.subheader("支援・共有候補")
    easy_mode = st.radio(
        "表示切替", ["標準表示", "やさしい日本語"], horizontal=True, key=f"easy_toggle_{key_suffix}"
    ) == "やさしい日本語"

    if not (visit or office or external):
        st.markdown(
            '<div class="app-card">職員確認済みの観察事項がないため、現時点での支援・共有候補はありません。</div>',
            unsafe_allow_html=True,
        )
    elif easy_mode:
        seen, lines = [], []
        for c in approved_categories:
            if c in seen:
                continue
            seen.append(c)
            if c in EASY_JAPANESE_VISIT:
                lines.append(EASY_JAPANESE_VISIT[c])
        lines.append(EASY_JAPANESE_OFFICE)
        lines.append(EASY_JAPANESE_EXTERNAL)
        st.markdown('<div class="app-card">' + "<br>".join(lines) + "</div>", unsafe_allow_html=True)
        st.caption("やさしい日本語表示は、職員への説明を補助するデモ用テンプレートです。正式な翻訳や研修を代替するものではありません。")
    else:
        for title, items in [
            ("次回訪問で確認すること", visit),
            ("事業所内で共有すること", office),
            ("関係職種への共有を検討すること", external),
        ]:
            if items:
                st.markdown(f"**{title}**")
                st.markdown('<div class="app-card">' + "".join(f"・{i}<br>" for i in items) + "</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="warn-box">
        表示される候補は、記録上の観察事項をもとにした一般的な確認・共有候補です。
        個別の支援内容、訪問回数、医療対応を決定するものではありません。
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 共通スタイル
# ============================================================

def inject_css():
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: #f8fafc; }}
        .app-card {{
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
        }}
        .metric-card {{
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-top: 4px solid {TEAL};
            border-radius: 10px;
            padding: 1rem 1.1rem;
            text-align: center;
        }}
        .metric-card .label {{ font-size: 0.85rem; color: #475569; margin-bottom: 0.25rem; }}
        .metric-card .value {{ font-size: 1.6rem; font-weight: 700; color: {TEAL_DARK}; }}
        .notice-box {{
            background-color: #f0fdfa;
            border: 1px solid #99f6e4;
            border-left: 5px solid {TEAL};
            border-radius: 8px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 1rem;
            color: #134e4a;
            font-size: 0.92rem;
        }}
        .warn-box {{
            background-color: #fffbeb;
            border: 1px solid #fde68a;
            border-left: 5px solid #b45309;
            border-radius: 8px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 1rem;
            color: #78350f;
            font-size: 0.92rem;
        }}
        .risk-badge {{
            display: inline-block; padding: 0.35rem 0.9rem; border-radius: 999px;
            font-weight: 700; font-size: 1rem; border: 1px solid transparent;
        }}
        .risk-low {{ background-color: #ecfdf5; color: #065f46; border-color: #6ee7b7; }}
        .risk-medium {{ background-color: #fffbeb; color: #92400e; border-color: #fcd34d; }}
        .risk-high {{ background-color: #fef2f2; color: #991b1b; border-color: #fca5a5; }}
        .ref-badge {{
            display: inline-block; padding: 0.3rem 0.8rem; border-radius: 999px;
            font-weight: 700; font-size: 0.88rem; border: 1px solid transparent;
        }}
        .ref-高 {{ background-color: #eef2ff; color: #3730a3; border-color: #a5b4fc; }}
        .ref-中 {{ background-color: #f1f5f9; color: #334155; border-color: #cbd5e1; }}
        .ref-低 {{ background-color: #f8fafc; color: #64748b; border-color: #cbd5e1; }}
        .ref-参考外 {{ background-color: #fef2f2; color: #991b1b; border-color: #fca5a5; border-style: dashed; }}
        .op-badge {{
            display: inline-block; padding: 0.3rem 0.8rem; border-radius: 999px;
            font-weight: 700; font-size: 0.88rem; border: 1px solid transparent;
        }}
        .op-優先確認 {{ background-color: #fef2f2; color: #991b1b; border-color: #fca5a5; }}
        .op-追加情報確認 {{ background-color: #fff7ed; color: #9a3412; border-color: #fdba74; }}
        .op-経過観察 {{ background-color: #eff6ff; color: #1e40af; border-color: #93c5fd; }}
        .metric-explain {{ font-size: 0.85rem; color: #475569; margin: 0.35rem 0 1rem 0; line-height: 1.6; }}
        .threshold-callout {{
            background-color: #f0fdfa; border: 2px solid {TEAL}; border-radius: 12px;
            padding: 1.25rem 1.5rem; margin-bottom: 1.25rem;
        }}
        .threshold-callout-title {{ font-size: 1.3rem; font-weight: 800; color: {TEAL_DARK}; margin-bottom: 0.35rem; }}
        .threshold-callout-body {{ color: #134e4a; margin-bottom: 0.9rem; }}
        .role-grid {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
        .role-card {{ flex: 1 1 240px; background-color: #ffffff; border-radius: 8px; padding: 0.75rem 1rem; border: 1px solid #e2e8f0; }}
        .role-detect {{ border-left: 5px solid {TEAL_DARK}; }}
        .role-priority {{ border-left: 5px solid #b91c1c; }}
        .role-card-title {{ font-weight: 700; margin-bottom: 0.2rem; }}
        .role-detect .role-card-title {{ color: {TEAL_DARK}; }}
        .role-priority .role-card-title {{ color: #b91c1c; }}
        .role-card-desc {{ font-size: 0.88rem; color: #334155; }}
        .missed-highlight {{ background-color: #fee2e2; color: #991b1b; font-weight: 700; padding: 0.1rem 0.4rem; border-radius: 4px; }}
        .future-box {{
            background-color: #f1f5f9; border: 2px dashed #94a3b8; border-radius: 12px;
            padding: 1.25rem 1.5rem; margin-bottom: 0.75rem; color: #334155;
        }}
        .future-box ul {{ margin: 0.3rem 0 0.8rem 1.2rem; }}
        .future-badge {{
            display: inline-block; background-color: #64748b; color: #ffffff; font-size: 0.75rem;
            font-weight: 700; padding: 0.2rem 0.6rem; border-radius: 999px; margin-bottom: 0.6rem;
        }}
        .future-title {{ font-size: 1.1rem; font-weight: 700; margin: 0.3rem 0 0.6rem 0; color: #334155; }}
        .future-flow-card {{
            background-color: #f1f5f9; border: 1px dashed #94a3b8; border-radius: 8px; padding: 0.6rem;
            text-align: center; font-size: 0.82rem; color: #334155; min-height: 70px;
            display: flex; align-items: center; justify-content: center; margin-bottom: 0.75rem;
        }}
        .flow-card {{
            background-color: #ffffff; border: 1px solid #e2e8f0; border-top: 3px solid {TEAL};
            border-radius: 10px; padding: 0.8rem; text-align: center; font-size: 0.85rem;
            min-height: 80px; display: flex; align-items: center; justify-content: center; margin-bottom: 0.75rem;
        }}
        .flow-arrow {{
            display: flex; align-items: center; justify-content: center; height: 80px;
            font-size: 1.4rem; color: {TEAL_DARK}; font-weight: 700; margin-bottom: 0.75rem;
        }}
        .stat-card {{ background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1rem 1.1rem; text-align: center; }}
        .stat-card .label {{ font-size: 0.82rem; color: #475569; margin-bottom: 0.25rem; }}
        .stat-card .value {{ font-size: 1.5rem; font-weight: 800; color: #1e293b; }}
        .obs-card {{
            background-color: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid {TEAL};
            border-radius: 10px; padding: 0.9rem 1.1rem; margin-bottom: 0.6rem; font-size: 0.92rem;
        }}
        section[data-testid="stSidebar"] .sidebar-footer {{
            font-size: 0.78rem; color: #475569; border-top: 1px solid #e2e8f0;
            padding-top: 0.75rem; margin-top: 0.75rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_flow_arrows(steps: list):
    n = len(steps)
    weights = []
    for i in range(n):
        weights.append(4)
        if i < n - 1:
            weights.append(1)
    cols = st.columns(weights)
    idx = 0
    for i, step in enumerate(steps):
        cols[idx].markdown(f'<div class="flow-card">{i + 1}. {step}</div>', unsafe_allow_html=True)
        idx += 1
        if i < n - 1:
            cols[idx].markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)
            idx += 1


# ============================================================
# データ・モデル読み込み
# ============================================================

def check_required_files():
    missing = [name for name, path in REQUIRED_FILES.items() if not path.exists()]
    if missing:
        st.error(
            "デモに必要なファイルが見つかりません。\n\n"
            + "\n".join(f"- {name}" for name in missing)
            + "\n\n`python3 scripts/export_streamlit_assets.py` と "
            "`python3 scripts/generate_operations_demo.py` を実行して、"
            "必要なファイルを書き出してから再度お試しください。"
        )
        st.stop()


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODELS_DIR / "calibrated_model.joblib")


@st.cache_data(show_spinner=False)
def load_feature_columns():
    with open(MODELS_DIR / "feature_columns.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_feature_defaults():
    with open(MODELS_DIR / "feature_defaults.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_metadata():
    with open(MODELS_DIR / "model_metadata.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_feature_importance():
    return pd.read_csv(OUTPUTS_DIR / "feature_importance.csv")


@st.cache_data(show_spinner=False)
def load_threshold_metrics():
    return pd.read_csv(OUTPUTS_DIR / "threshold_metrics.csv")


@st.cache_data(show_spinner=False)
def load_confusion_matrix():
    return pd.read_csv(OUTPUTS_DIR / "confusion_matrix.csv", index_col=0)


@st.cache_data(show_spinner=False)
def load_validation_predictions():
    return pd.read_csv(OUTPUTS_DIR / "validation_predictions.csv")


@st.cache_data(show_spinner=False)
def load_demo_cases():
    return pd.read_csv(DATA_DIR / "demo_cases.csv")


@st.cache_data(show_spinner=False)
def load_operations_demo():
    return pd.read_csv(DATA_DIR / "operations_demo.csv")


@st.cache_data(show_spinner=False)
def load_care_record_demo():
    return pd.read_csv(DATA_DIR / "care_record_demo.csv")


# ============================================================
# 予測ロジック（系統A：既存モデルをそのまま使用。再学習やロジック変更は行わない）
# ============================================================

def build_input_row(main_values: dict, feature_columns: list, feature_defaults: dict) -> pd.DataFrame:
    row = dict(feature_defaults)
    row.update(main_values)
    df = pd.DataFrame([row])
    df = df.reindex(columns=feature_columns)
    if df.isnull().any().any():
        missing = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"特徴量の値が不足しています: {missing}")
    if list(df.columns) != list(feature_columns):
        raise ValueError("特徴量の並び順が学習時と一致しません。")
    return df


def predict_risk_score(model, main_values: dict, feature_columns: list, feature_defaults: dict):
    """リスクスコアを算出する。失敗した場合は (None, エラーメッセージ) を返す。"""
    try:
        row = build_input_row(main_values, feature_columns, feature_defaults)
        score = float(model.predict_proba(row)[:, 1][0])
        return score, None
    except Exception as exc:  # 予測時の例外を利用者に分かる形で表示するため捕捉
        return None, f"リスクスコアの算出中にエラーが発生しました: {exc}"


def classify_risk(score: float, selected_threshold: float, high_risk_threshold: float) -> str:
    if score >= high_risk_threshold:
        return "High"
    elif score >= selected_threshold:
        return "Medium"
    return "Low"


def render_risk_badge(level: str) -> str:
    label_map = {"Low": ("risk-low", "Low（低）"), "Medium": ("risk-medium", "Medium（中）"), "High": ("risk-high", "High（高）")}
    css_class, label = label_map[level]
    return f'<span class="risk-badge {css_class}">{label}</span>'


def render_reference_badge(level: str) -> str:
    label = REFERENCE_DISPLAY_MAP.get(level, level)
    return f'<span class="ref-badge ref-{level}">入力情報の状態：{label}</span>'


def render_operational_badge(category: str) -> str:
    return f'<span class="op-badge op-{category}">運用上の確認区分：{category}</span>'


def render_risk_score_definition():
    st.markdown(
        """
        <div class="notice-box">
        <b>モデル上のリスクスコアとは</b>：入力された特徴が、学習データ内の「該当あり」の
        特徴にどの程度近いかを0〜1で示す参考値です。数値が高いほど、モデル上は
        「該当あり」に近い特徴を持つと判定されます。ただし、将来の認知症発症確率や
        医学的診断を示すものではありません。
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_threshold_callout(metadata: dict):
    sel = metadata["thresholds"]["selected_threshold"]
    high = metadata["thresholds"]["high_risk_threshold"]
    st.markdown(
        f"""
        <div class="threshold-callout">
            <div class="threshold-callout-title">採用閾値は {sel:.2f} です</div>
            <div class="threshold-callout-body">
                リスクのある対象者の見逃しを減らすため、Recall（再現率）を重視して設定しています。
            </div>
            <div class="role-grid">
                <div class="role-card role-detect">
                    <div class="role-card-title">{sel:.2f}：検知ライン</div>
                    <div class="role-card-desc">経過観察候補を広く拾い上げるためのラインです。</div>
                </div>
                <div class="role-card role-priority">
                    <div class="role-card-title">{high:.2f}：運用ライン</div>
                    <div class="role-card-desc">専門職による優先的な確認候補を絞り込むための運用上の目安です（モデルの正式な診断閾値ではありません）。</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_action_candidates(level: str):
    st.subheader("支援・確認アクション候補")
    st.caption("優先度を確認した後、職員が次に確認・共有すべき項目を整理するための一般的な業務案内です。")
    items = ACTION_CANDIDATES[level]
    st.markdown('<div class="app-card">' + "".join(f"・{item}<br>" for item in items) + "</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="warn-box">
        表示される内容は、リスク区分に応じた一般的な確認・共有候補です。個別の支援内容や
        医療上の対応を決定するものではありません。本人の希望、ケアプラン、直近の状態、
        専門職の評価を踏まえて、職員が採否を判断します。
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_importance_section(feature_importance: pd.DataFrame):
    st.subheader("モデル全体の特徴量重要度")
    top_n = 6
    top_features = feature_importance.head(top_n).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.barh(top_features["feature"], top_features["importance"], color=TEAL)
    ax.set_xlabel("Importance Score", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title(f"Top {top_n} Features (Model-level Importance)", fontsize=10)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    meaning_lines = [
        f"・{feat}：{FEATURE_JP_MEANING[feat]}"
        for feat in feature_importance.head(top_n)["feature"]
        if feat in FEATURE_JP_MEANING
    ]
    if meaning_lines:
        st.markdown('<div class="metric-explain">' + "<br>".join(meaning_lines) + "</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="notice-box">
        このグラフは、モデル全体がどの項目を重視しているかを示します。今回の個別予測の
        原因を直接説明するものではありません。
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_comparison_section(main_inputs: dict, feature_defaults: dict):
    st.subheader("今回の入力値と基準値の比較")
    rows = []
    for feat, label in MAIN_FEATURE_LABELS.items():
        input_val = main_inputs[feat]
        default_val = feature_defaults[feat]
        rows.append(
            {
                "項目名": label,
                "今回の入力値": input_val,
                "デモ用基準値": round(float(default_val), 2) if feat not in BINARY_MAIN_FEATURES else default_val,
                "差": round(float(input_val) - float(default_val), 2),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown(
        """
        <div class="warn-box">
        表内の「差」は、今回の入力値とデモ用基準値との単純な差であり、リスクスコアへの
        寄与度（原因の説明）を示すものではありません。
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_data_gap_section():
    st.subheader("実運用に向けたデータ取得上の課題")
    st.markdown(
        """
        <div class="app-card">
        本デモでは、公開データに含まれるMMSE、ADL、FunctionalAssessmentなどの構造化評価値と、
        キーワードルールで抽出した記録情報を使用しています。実際の訪問介護現場では、
        構造化評価値が全利用者について継続的に取得されているとは限らないため、
        現場で取得可能なデータに合わせて入力項目と検証方法を再設計する必要があります。
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_future_vision_section():
    st.markdown(
        """
        <div class="future-box">
            <span class="future-badge">将来構想（現在は未実装です）</span>
            <div class="future-title">将来構想：介護記録から観察事項を構造化</div>
            <p>訪問介護計画書・ケアプラン・既往歴・日々の介護記録・モニタリング記録などから、
            認知面の変化、ADL・IADLの変化、服薬管理、安全面の懸念、行動・心理面の変化、
            以前の状態との差、発生頻度と継続性といった観察事項を抽出する構想です。
            現在のデモは、その最小構成（キーワードルールによる抽出）を先行実装したものです。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    flow_cols = st.columns(6)
    flow_steps = ["計画書・介護記録", "観察事項と根拠文章を抽出", "職員が確認・修正", "時系列で蓄積", "確認優先度を検討", "専門職が最終判断"]
    for col, step in zip(flow_cols, flow_steps):
        col.markdown(f'<div class="future-flow-card">{step}</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="warn-box">
        介護記録から正式なMMSEなどの検査点数を自動生成する構想ではありません。
        記録に記載された観察事項と根拠文章を構造化し、職員の確認後に意思決定材料として
        利用することを想定しています。<br><b>この機能は将来構想であり、現在は未実装です。</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_limitations():
    st.subheader("本デモの限界")
    st.markdown('<div class="app-card">' + "<br>".join(f"・{item}" for item in LIMITATIONS) + "</div>", unsafe_allow_html=True)


def render_demo_data_disclaimer():
    st.caption("※本デモの利用者・記録・数値はすべて架空データです。実在利用者の情報は使用していません。")


def input_status_rule_text():
    st.markdown(
        """
        <div class="notice-box">
        「入力情報の状態」は、モデルの予測精度ではありません。入力情報の充足度・更新状況・
        職員確認状況から、現在のモデルスコアを業務上どの程度参考にできるかを示す
        <b>デモ用の業務指標</b>です。
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="app-card">
        ・<b>情報は十分</b>：情報が十分で、新しく、主要情報が職員確認済み<br>
        ・<b>一部不足</b>：一部不足、または一定期間更新されていない<br>
        ・<b>更新が必要</b>：不足、古い情報、未確認情報が多い<br>
        ・<b>判断材料不足</b>：重要情報が不足し、現在のスコアを運用判断に使うべきではない
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("詳しい判定基準"):
        st.markdown(
            """
            - 情報は十分：充足率80%以上、更新30日以内、主要情報が職員確認済み
            - 一部不足：充足率50〜79%、または更新31〜90日
            - 更新が必要：充足率50%未満、更新90日超、未確認情報が多い
            - 判断材料不足：重要情報が不足し、スコアを運用判断に使うべきでない

            このルールは医学的に検証された指標ではなく、現場運用を検討するためのデモ用ルールです。
            """
        )


# ============================================================
# CSVアップロード（介護ソフトCSV連携を想定した共通フォーマットPoC）
# 特定製品との正式連携ではない。カイポケ等の専用API連携は将来構想。
# ============================================================

CSV_ENCODINGS = ["utf-8", "utf-8-sig", "cp932", "shift_jis"]

CSV_SCHEMAS = {
    "users": {
        "label": "利用者基本情報",
        "filename_hint": "users.csv",
        "standard_columns": ["user_id", "age", "care_level", "medical_history", "language_support", "notes"],
        "required_columns": ["user_id"],
        "date_columns": [],
    },
    "care_plans": {
        "label": "訪問介護計画書",
        "filename_hint": "care_plans.csv",
        "standard_columns": [
            "user_id", "plan_start_date", "plan_end_date", "goal",
            "support_content", "precautions", "medication_support", "mobility_support",
        ],
        "required_columns": ["user_id", "support_content"],
        "date_columns": ["plan_start_date", "plan_end_date"],
    },
    "daily_records": {
        "label": "日々の介護記録",
        "filename_hint": "daily_records.csv",
        "standard_columns": ["user_id", "record_date", "service_type", "record_text", "special_notes", "staff_id"],
        "required_columns": ["user_id", "record_date", "record_text"],
        "date_columns": ["record_date"],
    },
    "monitoring_records": {
        "label": "モニタリング記録",
        "filename_hint": "monitoring_records.csv",
        "standard_columns": ["user_id", "monitoring_date", "current_status", "change_from_previous", "issues", "reviewer_id"],
        "required_columns": ["user_id", "monitoring_date", "current_status"],
        "date_columns": ["monitoring_date"],
    },
}


@st.cache_data(show_spinner=False)
def load_column_aliases():
    with open(DATA_DIR / "column_aliases.json", encoding="utf-8") as f:
        return json.load(f)


def read_csv_with_encoding_fallback(uploaded_file):
    """UTF-8 / UTF-8(BOM付き) / CP932 / Shift_JISの順に読み込みを試す。

    すべて失敗した場合は、技術的なスタックトレースではなく
    利用者向けのエラーメッセージを返す。
    """
    raw = uploaded_file.getvalue()
    for enc in CSV_ENCODINGS:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc, dtype=str)
            return df, enc, None
        except Exception:
            continue
    return None, None, "CSVの文字コードまたは形式を確認できませんでした。UTF-8またはShift_JIS形式で保存して再度お試しください。"


def auto_map_columns(actual_columns: list, standard_columns: list, aliases: dict) -> dict:
    mapping = {}
    for std_col in standard_columns:
        if std_col in actual_columns:
            mapping[std_col] = std_col
            continue
        for alias in aliases.get(std_col, []):
            if alias in actual_columns:
                mapping[std_col] = alias
                break
    return mapping


def render_csv_upload_section(schema_key: str, aliases: dict, other_normalized: dict):
    """1種類のCSVについて、アップロード→文字コード判定→列名マッピング→検証を行う。

    存在しない列を推測で補うことはしない。必須列が自動認識できない場合は
    手動マッピングUIを表示する。
    """
    schema = CSV_SCHEMAS[schema_key]
    st.markdown(f"**{schema['label']}**（推奨ファイル名: {schema['filename_hint']}）")
    uploaded = st.file_uploader(
        f"{schema['label']}のCSVを選択", type=["csv"], key=f"upload_{schema_key}", label_visibility="collapsed",
    )
    if uploaded is None:
        st.caption("未アップロードです。")
        return None

    raw_df, encoding_used, err = read_csv_with_encoding_fallback(uploaded)
    if err:
        st.error(f"{schema['filename_hint']}：{err}")
        return None

    raw_df.columns = [str(c).strip() for c in raw_df.columns]
    mapping = auto_map_columns(list(raw_df.columns), schema["standard_columns"], aliases)

    missing_required = [c for c in schema["required_columns"] if c not in mapping]
    if missing_required:
        st.warning(
            f"{schema['filename_hint']}：必須項目（{', '.join(missing_required)}）に対応する列を"
            "自動認識できませんでした。該当する列を選択してください。"
        )
        options = ["（該当なし）"] + list(raw_df.columns)
        for col in missing_required:
            choice = st.selectbox(f"「{col}」に対応する列", options, key=f"manualmap_{schema_key}_{col}")
            if choice != "（該当なし）":
                mapping[col] = choice
        missing_required = [c for c in schema["required_columns"] if c not in mapping]

    if missing_required:
        st.error(f"{schema['filename_hint']}：読込不可。必須項目（{', '.join(missing_required)}）が見つかりません。")
        return None

    normalized = pd.DataFrame()
    for std_col in schema["standard_columns"]:
        normalized[std_col] = raw_df[mapping[std_col]] if std_col in mapping else None

    messages = []
    empty_uid = normalized["user_id"].isna() | (normalized["user_id"].astype(str).str.strip() == "")
    if empty_uid.any():
        messages.append(f"user_idが空の行が{int(empty_uid.sum())}件あります（除外して読み込みます）。")
        normalized = normalized[~empty_uid].reset_index(drop=True)

    for date_col in schema["date_columns"]:
        original = normalized[date_col]
        parsed = pd.to_datetime(original, errors="coerce")
        bad = parsed.isna() & original.notna() & (original.astype(str).str.strip() != "")
        if bad.any():
            messages.append(f"{date_col}を日付として読み込めない行が{int(bad.sum())}件あります。")
        normalized[date_col] = parsed

    full_dup = normalized.duplicated()
    if full_dup.any():
        messages.append(f"完全重複行が{int(full_dup.sum())}件あります（重複分は除外します）。")
        normalized = normalized[~full_dup].reset_index(drop=True)

    text_cols_to_check = [c for c in schema["required_columns"] if c in ("record_text", "support_content", "current_status")]
    for col in text_cols_to_check:
        empty_text = normalized[col].isna() | (normalized[col].astype(str).str.strip() == "")
        if empty_text.any():
            messages.append(f"{col}が空の行が{int(empty_text.sum())}件あります。")

    if schema_key == "daily_records":
        subset_dup = normalized.duplicated(subset=["user_id", "record_date", "record_text"])
        if subset_dup.any():
            messages.append(f"同じ利用者・記録日・記録内容の重複が{int(subset_dup.sum())}件あります。")
    if schema_key == "monitoring_records":
        subset_dup = normalized.duplicated(subset=["user_id", "monitoring_date", "current_status"])
        if subset_dup.any():
            messages.append(f"同じ利用者・モニタリング日・内容の重複が{int(subset_dup.sum())}件あります。")

    if schema_key != "users" and other_normalized.get("users") is not None:
        known_ids = set(other_normalized["users"]["user_id"].astype(str))
        this_ids = set(normalized["user_id"].astype(str).unique())
        unknown = sorted(this_ids - known_ids)
        if unknown:
            messages.append(f"利用者基本情報に存在しないuser_idが{len(unknown)}件あります（例: {', '.join(unknown[:5])}）。")

    st.success(f"{schema['filename_hint']}：{len(normalized)}件を読み込みました（文字コード: {encoding_used}）。")
    for msg in messages:
        st.warning(msg)

    return normalized


def compute_input_status(has_basic_info: bool, has_care_plan: bool, daily_count: int, monitoring_count: int, last_record_days) -> str:
    """入力情報の状態（旧データ参考度）を、記録・計画書の充足度から判定するデモ用ルール。

    医学的な信頼度指標ではなく、現場運用を検討するためのデモ用業務指標である。
    """
    total = daily_count + monitoring_count
    if total < 2:
        return "判断材料不足"
    points = int(has_basic_info) + int(has_care_plan)
    if last_record_days is not None and last_record_days <= 30:
        points += 1
    if total >= 3:
        points += 1
    if points >= 4:
        return "情報は十分"
    if points >= 2:
        return "一部不足"
    return "更新が必要"


def build_integration_preview(users_df, plans_df, daily_df, monitoring_df) -> pd.DataFrame:
    """利用者ID単位でCSVを統合し、データ統合プレビューを作成する。"""
    all_ids = set()
    for df in [users_df, plans_df, daily_df, monitoring_df]:
        if df is not None:
            all_ids.update(df["user_id"].astype(str).unique())

    today = pd.Timestamp.now().normalize()
    rows = []
    for uid in sorted(all_ids):
        has_basic = users_df is not None and uid in set(users_df["user_id"].astype(str))
        has_plan = plans_df is not None and uid in set(plans_df["user_id"].astype(str))
        daily_count = 0 if daily_df is None else int((daily_df["user_id"].astype(str) == uid).sum())
        monitoring_count = 0 if monitoring_df is None else int((monitoring_df["user_id"].astype(str) == uid).sum())

        dates = []
        if daily_df is not None:
            dates += daily_df.loc[daily_df["user_id"].astype(str) == uid, "record_date"].dropna().tolist()
        if monitoring_df is not None:
            dates += monitoring_df.loc[monitoring_df["user_id"].astype(str) == uid, "monitoring_date"].dropna().tolist()
        last_date = max(dates) if dates else None
        last_record_days = (today - last_date).days if last_date is not None else None

        input_status = compute_input_status(has_basic, has_plan, daily_count, monitoring_count, last_record_days)
        rows.append(
            {
                "user_id": uid,
                "has_basic_info": has_basic,
                "has_care_plan": has_plan,
                "daily_count": daily_count,
                "monitoring_count": monitoring_count,
                "last_record_date": last_date,
                "last_record_days": last_record_days,
                "input_status": input_status,
            }
        )
    return pd.DataFrame(rows)


def extract_observations_for_user_csv(user_id: str, plans_df, daily_df, monitoring_df) -> list:
    """アップロードCSVの複数列から、利用者1名分の観察事項候補をまとめて抽出する。"""
    observations = []
    if daily_df is not None:
        sub = daily_df[daily_df["user_id"].astype(str) == user_id]
        for _, r in sub.iterrows():
            date_str = r["record_date"].date().isoformat() if pd.notna(r["record_date"]) else "日付不明"
            observations += extract_observations_from_text(r.get("record_text"), date_str, "日々の介護記録")
            if str(r.get("special_notes")).strip() and str(r.get("special_notes")).lower() != "nan":
                observations += extract_observations_from_text(r.get("special_notes"), date_str, "日々の介護記録（特記事項）")
    if monitoring_df is not None:
        sub = monitoring_df[monitoring_df["user_id"].astype(str) == user_id]
        for _, r in sub.iterrows():
            date_str = r["monitoring_date"].date().isoformat() if pd.notna(r["monitoring_date"]) else "日付不明"
            for field, label in [
                ("current_status", "モニタリング記録"),
                ("change_from_previous", "モニタリング記録（前回からの変化）"),
                ("issues", "モニタリング記録（課題）"),
            ]:
                val = r.get(field)
                if val is not None and str(val).strip() and str(val).lower() != "nan":
                    observations += extract_observations_from_text(val, date_str, label)
    if plans_df is not None:
        sub = plans_df[plans_df["user_id"].astype(str) == user_id]
        for _, r in sub.iterrows():
            date_val = r.get("plan_start_date")
            date_str = date_val.date().isoformat() if pd.notna(date_val) else "日付不明"
            for field, label in [
                ("precautions", "訪問介護計画書（留意事項）"),
                ("medication_support", "訪問介護計画書（服薬支援）"),
            ]:
                val = r.get(field)
                if val is not None and str(val).strip() and str(val).lower() != "nan":
                    observations += extract_observations_from_text(val, date_str, label, allow_info_insufficient=False)
    return observations


def collect_substantive_texts_csv(user_id: str, daily_df, monitoring_df) -> list:
    texts = []
    if daily_df is not None:
        texts += daily_df.loc[daily_df["user_id"].astype(str) == user_id, "record_text"].astype(str).tolist()
    if monitoring_df is not None:
        texts += monitoring_df.loc[monitoring_df["user_id"].astype(str) == user_id, "current_status"].astype(str).tolist()
    return texts


def build_review_priority_results_df(reviewed_users: dict) -> pd.DataFrame:
    rows = []
    for uid, info in reviewed_users.items():
        rows.append(
            {
                "user_id": uid,
                "operational_priority": info.get("operational_category", ""),
                "main_change": info.get("main_change", ""),
                "source_type": info.get("main_source_type", ""),
                "source_date": info.get("main_source_date", ""),
                "input_information_status": info.get("input_status", ""),
                "review_status": info.get("review_status", ""),
            }
        )
    return pd.DataFrame(rows)


def build_confirmed_observations_df(reviewed_users: dict) -> pd.DataFrame:
    rows = []
    for uid, info in reviewed_users.items():
        for obs in info.get("approved_observations", []):
            rows.append(
                {
                    "user_id": uid,
                    "observation_category": obs.get("category", ""),
                    "confirmed_observation": obs.get("confirmed_text", obs.get("evidence", "")),
                    "evidence_text": obs.get("evidence", ""),
                    "record_date": obs.get("record_date", ""),
                    "source_type": obs.get("source_type", ""),
                    "staff_decision": obs.get("staff_decision", ""),
                }
            )
    return pd.DataFrame(rows)


def render_session_storage_notice():
    st.caption(
        "アップロードしたデータと職員確認内容は、このデモセッション内のみ保持されます。"
        "ページを再読み込みすると内容が失われる場合があります。"
    )


# ============================================================
# サイドバー
# ============================================================

PAGES = ["1. 記録から確認優先度まで", "2. 利用者一覧・詳細", "3. モデル検証・技術情報"]


def render_sidebar():
    st.sidebar.title("介護記録CSVから状態変化と確認優先度を整理するPoC")
    st.sidebar.caption("Care Record CSV Review & Priority Support PoC")
    page = st.sidebar.radio("画面を選択", PAGES, label_visibility="collapsed")
    st.sidebar.markdown(
        """
        <div class="sidebar-footer">
        本デモは医療診断を目的としたものではありません。AIが診断や支援内容を確定するものではなく、
        抽出結果と候補を職員・専門職が確認して判断します。
        </div>
        """,
        unsafe_allow_html=True,
    )
    return page


# ============================================================
# 画面1: 記録から確認優先度まで（中心画面）
# ============================================================

def _finalize_decisions(decisions: list) -> list:
    approved = []
    for obs, decision, corrected_text in decisions:
        if decision == "不採用":
            continue
        final_obs = dict(obs)
        final_obs["confirmed_text"] = corrected_text if (decision == "修正して採用" and corrected_text) else obs["evidence"]
        final_obs["staff_decision"] = decision
        approved.append(final_obs)
    return approved


def _summarize_main_change(approved: list):
    non_info = [o for o in approved if o["category"] != "情報不足"]
    if non_info:
        top = non_info[0]
        text = top.get("confirmed_text", top["evidence"])
        return f"{top['category']}：{text[:30]}", top["source_type"], str(top.get("record_date", "-"))
    if approved:
        top = approved[0]
        return "情報不足", top["source_type"], str(top.get("record_date", "-"))
    return "特に確認すべき記載なし", "-", "-"


def _render_observation_cards_and_form(extracted: list, form_key: str, key_prefix: str):
    decisions = []
    for i, obs in enumerate(extracted):
        st.markdown(
            f'<div class="obs-card"><b>観察カテゴリ</b>：{obs["category"]}<br>'
            f'<b>根拠文章</b>：{obs["evidence"]}<br>'
            f'<b>記録日</b>：{obs.get("record_date", "-")}／<b>情報源</b>：{obs.get("source_type", "-")}／'
            f'<b>一致語</b>：{obs.get("matched_keyword") or "-"}</div>',
            unsafe_allow_html=True,
        )
        decision = st.radio(
            f"扱い（{i + 1}件目）", ["採用", "修正して採用", "不採用"],
            horizontal=True, key=f"{key_prefix}_decision_{i}",
        )
        corrected_text = None
        if decision == "修正して採用":
            corrected_text = st.text_area("修正後の内容", value=obs["evidence"], key=f"{key_prefix}_correction_{i}")
        decisions.append((obs, decision, corrected_text))
    return decisions


def page_record_to_priority(model, feature_columns, feature_defaults, metadata, care_df: pd.DataFrame):
    st.title("介護記録CSVから状態変化と確認優先度を整理するPoC")
    st.caption("Care Record CSV Review & Priority Support PoC")

    st.markdown(
        """
        <div class="app-card">
        介護ソフトから出力した利用者情報、計画書、日々の記録、モニタリングのCSVを取り込み、
        状態変化と根拠を整理します。職員確認後、確認優先度と支援・共有候補を表示します。
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_flow_arrows(
        ["CSVをアップロード", "利用者単位でデータを統合", "状態変化と根拠を抽出", "職員が確認・修正", "確認優先度を判定", "支援・共有候補を提示", "結果を出力"]
    )

    st.markdown(
        '<div class="notice-box">介護ソフトCSV連携を想定した共通フォーマットPoCです。特定製品との正式連携ではありません。</div>',
        unsafe_allow_html=True,
    )

    mode = st.radio("入力方法", ["サンプルデータで試す", "CSVをアップロードする"], horizontal=True, key="input_mode")

    if mode == "サンプルデータで試す":
        render_sample_mode(model, feature_columns, feature_defaults, metadata, care_df)
    else:
        render_csv_upload_mode(model, feature_columns, feature_defaults, metadata)

    st.markdown("---")
    render_export_section()
    render_session_storage_notice()


def render_sample_mode(model, feature_columns, feature_defaults, metadata, care_df: pd.DataFrame):
    st.caption("既存の架空利用者デモ（U001〜U003）です。")
    user_ids = care_df["user_id"].unique().tolist()
    selected_user = st.selectbox("架空利用者を選択", user_ids, key="sample_user_select")
    render_demo_data_disclaimer()

    user_rows = care_df[care_df["user_id"] == selected_user].sort_values("record_date").reset_index(drop=True)
    first_row = user_rows.iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="app-card"><b>ケアプラン概要</b>：{first_row["care_plan_text"]}<br>'
            f'<b>前回の状態</b>：{first_row["previous_state"]}</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(f'<div class="app-card"><b>現在の支援内容</b>：{first_row["current_support"]}</div>', unsafe_allow_html=True)

    with st.expander(f"計画書・日々の記録・モニタリング（{len(user_rows)}件）を見る"):
        for _, r in user_rows.iterrows():
            st.markdown(f"**{r['record_date']}｜{r['source_type']}**  \n{r['record_text']}")
            st.markdown("---")

    extract_key = f"care_extracted_{selected_user}"
    approved_key = f"care_approved_{selected_user}"

    st.caption("この抽出はキーワード・ルールベースの処理です（AI解析ではありません）。抽出候補は、職員確認後に確認優先度へ反映されます。")

    if st.button("観察事項を抽出", type="primary", key=f"extract_btn_{selected_user}"):
        st.session_state[extract_key] = extract_all_observations(user_rows)
        st.session_state.pop(approved_key, None)

    extracted = st.session_state.get(extract_key)
    if extracted is None:
        st.info("「観察事項を抽出」ボタンを押すと、記録から観察事項の候補が抽出されます。")
        return

    if not extracted:
        st.markdown(
            f'<div class="notice-box">記録上、注意すべき記載は見つかりませんでした（レビューした記録：{len(user_rows)}件）。</div>',
            unsafe_allow_html=True,
        )
        st.session_state[approved_key] = []
    else:
        st.subheader("抽出結果の確認")
        with st.form(key=f"review_form_{selected_user}"):
            decisions = _render_observation_cards_and_form(extracted, f"review_form_{selected_user}", f"sample_{selected_user}")
            submitted = st.form_submit_button("確認結果を反映", type="primary")
        if submitted:
            st.session_state[approved_key] = _finalize_decisions(decisions)

    approved = st.session_state.get(approved_key)
    if approved is None:
        st.info("観察事項の採否を選択し、「確認結果を反映」を押すと、確認優先度が表示されます。")
        return

    st.markdown("---")
    st.subheader("確認優先度と支援・共有候補")

    record_state_level = compute_record_state_level(user_rows, approved)

    structured_available = bool(first_row["structured_data_available"])
    model_risk_level, model_score = None, None
    if structured_available:
        main_values = {
            "MMSE": float(first_row["confirmed_mmse"]),
            "ADL": float(first_row["confirmed_adl"]),
            "FunctionalAssessment": float(first_row["confirmed_functional_assessment"]),
        }
        model_score, error = predict_risk_score(model, main_values, feature_columns, feature_defaults)
        if error:
            st.error(error)
        else:
            model_risk_level = classify_risk(
                model_score, metadata["thresholds"]["selected_threshold"], metadata["thresholds"]["high_risk_threshold"]
            )

    operational_category = compute_operational_category_from_tracks(record_state_level, model_risk_level)

    col1, col2, col3 = st.columns(3)
    col1.markdown(
        f'<div class="stat-card"><div class="label">記録上の状態変化レベル（系統B）</div><div class="value">{record_state_level}</div></div>',
        unsafe_allow_html=True,
    )
    with col2:
        if structured_available:
            st.markdown(
                f'<div class="stat-card"><div class="label">モデル上のリスク区分（系統A）</div>'
                f'<div class="value" style="font-size:1.2rem;">{model_risk_level}</div></div>',
                unsafe_allow_html=True,
            )
            st.caption(f"モデル上のリスクスコア：{model_score:.3f}")
            st.caption("一部のモデル入力項目はデモ用基準値で補完しています。")
        else:
            st.markdown(
                '<div class="stat-card"><div class="label">モデル上のリスク区分（系統A）</div>'
                '<div class="value" style="font-size:0.85rem;">構造化評価値が未取得のため、機械学習モデルのスコアは算出していません</div></div>',
                unsafe_allow_html=True,
            )
    col3.markdown(render_operational_badge(operational_category), unsafe_allow_html=True)

    if structured_available:
        render_risk_score_definition()

    st.caption("運用上の確認区分は、系統Aと系統Bの結果から判定するデモ用の業務ルールであり、医学的判断ではありません。")

    render_share_candidates_section(approved, key_suffix=f"sample_{selected_user}")

    main_change, main_source_type, main_source_date = _summarize_main_change(approved)
    input_status = "情報は十分" if structured_available else ("判断材料不足" if record_state_level == "情報不足" else "一部不足")
    st.session_state["reviewed_users"][selected_user] = {
        "operational_category": operational_category,
        "record_state_level": record_state_level,
        "model_risk_level": model_risk_level,
        "model_score": model_score,
        "structured_available": structured_available,
        "input_status": input_status,
        "approved_observations": approved,
        "main_change": main_change,
        "main_source_type": main_source_type,
        "main_source_date": main_source_date,
        "review_status": "確認済み",
        "basic_info": {
            "ケアプラン概要": first_row["care_plan_text"],
            "前回の状態": first_row["previous_state"],
            "現在の支援内容": first_row["current_support"],
        },
    }


def render_csv_upload_mode(model, feature_columns, feature_defaults, metadata):
    aliases = load_column_aliases()

    with st.expander("サンプルCSVをダウンロード（そのまま再アップロードして試せます）"):
        st.caption("すべて架空データです。実在利用者の情報は使用していません。")
        dl_cols = st.columns(4)
        sample_files = [
            ("users", "sample_users.csv"),
            ("care_plans", "sample_care_plans.csv"),
            ("daily_records", "sample_daily_records.csv"),
            ("monitoring_records", "sample_monitoring_records.csv"),
        ]
        for col, (schema_key, fname) in zip(dl_cols, sample_files):
            path = DATA_DIR / fname
            if path.exists():
                col.download_button(
                    CSV_SCHEMAS[schema_key]["label"], data=path.read_bytes(), file_name=fname,
                    mime="text/csv", key=f"dl_{schema_key}",
                )

    st.markdown(
        '<div class="notice-box">最低限、日々の介護記録（daily_records.csv）があれば観察事項の抽出を実行できます。'
        '他のCSVが未アップロードの場合は、下の統合プレビューで不足情報として表示されます。</div>',
        unsafe_allow_html=True,
    )

    normalized = {}
    normalized["users"] = render_csv_upload_section("users", aliases, normalized)
    normalized["care_plans"] = render_csv_upload_section("care_plans", aliases, normalized)
    normalized["daily_records"] = render_csv_upload_section("daily_records", aliases, normalized)
    normalized["monitoring_records"] = render_csv_upload_section("monitoring_records", aliases, normalized)

    if normalized["daily_records"] is None:
        st.info("daily_records.csvをアップロードすると、データ統合プレビューと観察事項の抽出を行えます。")
        return

    st.markdown("---")
    st.subheader("データ統合プレビュー")
    with st.expander("「入力情報の状態」の判定基準"):
        input_status_rule_text()
    preview = build_integration_preview(
        normalized["users"], normalized["care_plans"], normalized["daily_records"], normalized["monitoring_records"]
    )
    display_preview = pd.DataFrame(
        {
            "利用者ID": preview["user_id"],
            "基本情報": preview["has_basic_info"].map({True: "あり", False: "なし"}),
            "計画書": preview["has_care_plan"].map({True: "あり", False: "なし"}),
            "日々の記録件数": preview["daily_count"],
            "モニタリング件数": preview["monitoring_count"],
            "最終記録日": preview["last_record_date"].apply(lambda d: d.date().isoformat() if pd.notna(d) else "-"),
            "入力情報の状態": preview["input_status"],
        }
    )
    st.dataframe(display_preview, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("観察事項の抽出と職員確認")
    st.caption("この抽出はキーワード・ルールベースの処理です（AI解析ではありません）。抽出候補は、職員確認後に確認優先度へ反映されます。")

    selected_user = st.selectbox("処理する利用者を選択", preview["user_id"].tolist(), key="csv_user_select")
    user_preview_row = preview[preview["user_id"] == selected_user].iloc[0]

    extract_key = f"csv_extracted_{selected_user}"
    approved_key = f"csv_approved_{selected_user}"

    if st.button("観察事項を抽出", type="primary", key=f"csv_extract_btn_{selected_user}"):
        st.session_state[extract_key] = extract_observations_for_user_csv(
            selected_user, normalized["care_plans"], normalized["daily_records"], normalized["monitoring_records"]
        )
        st.session_state.pop(approved_key, None)

    extracted = st.session_state.get(extract_key)
    if extracted is None:
        st.info("「観察事項を抽出」ボタンを押すと、この利用者の記録から観察事項の候補が抽出されます。")
        return

    if not extracted:
        st.markdown('<div class="notice-box">記録上、注意すべき記載は見つかりませんでした。</div>', unsafe_allow_html=True)
        st.session_state[approved_key] = []
    else:
        with st.form(key=f"csv_review_form_{selected_user}"):
            decisions = _render_observation_cards_and_form(extracted, f"csv_review_form_{selected_user}", f"csv_{selected_user}")
            submitted = st.form_submit_button("確認結果を反映", type="primary")
        if submitted:
            st.session_state[approved_key] = _finalize_decisions(decisions)

    approved = st.session_state.get(approved_key)
    if approved is None:
        st.info("観察事項の採否を選択し、「確認結果を反映」を押すと、確認優先度が表示されます。")
        return

    st.markdown("---")
    st.subheader("確認優先度と支援・共有候補")

    texts = collect_substantive_texts_csv(selected_user, normalized["daily_records"], normalized["monitoring_records"])
    record_state_level = compute_record_state_level_from_texts(texts, approved)
    input_status = user_preview_row["input_status"]

    # 系統A：標準CSV列にはMMSE/ADL/FunctionalAssessmentを含めていないため、
    # CSVアップロード経由で系統A（既存モデル）が自動適用されることはない。
    # 推測・生成は行わず、モデルスコアは常に「未算出」として扱う。
    operational_category = compute_operational_category_csv(record_state_level, None, input_status)

    col1, col2, col3 = st.columns(3)
    col1.markdown(
        f'<div class="stat-card"><div class="label">記録上の状態変化レベル（系統B）</div><div class="value">{record_state_level}</div></div>',
        unsafe_allow_html=True,
    )
    col2.markdown(
        f'<div class="stat-card"><div class="label">入力情報の状態</div><div class="value" style="font-size:1.1rem;">{input_status}</div></div>',
        unsafe_allow_html=True,
    )
    col3.markdown(render_operational_badge(operational_category), unsafe_allow_html=True)

    st.markdown(
        '<div class="notice-box">構造化評価値が未取得のため、機械学習モデルのスコアは算出していません。'
        'この確認優先度は記録上の観察事項と入力情報の状態から判定しています。</div>',
        unsafe_allow_html=True,
    )
    st.caption("確認優先度は、職員確認済みの観察事項・記録の継続性・安全面や服薬管理の懸念・入力情報の状態から判定するデモ用の業務ルールであり、医学的判断ではありません。")

    render_share_candidates_section(approved, key_suffix=f"csv_{selected_user}")

    main_change, main_source_type, main_source_date = _summarize_main_change(approved)
    st.session_state["reviewed_users"][selected_user] = {
        "operational_category": operational_category,
        "record_state_level": record_state_level,
        "model_risk_level": None,
        "model_score": None,
        "structured_available": False,
        "input_status": input_status,
        "approved_observations": approved,
        "main_change": main_change,
        "main_source_type": main_source_type,
        "main_source_date": main_source_date,
        "review_status": "確認済み",
        "basic_info": {
            "基本情報": "あり" if user_preview_row["has_basic_info"] else "なし",
            "計画書": "あり" if user_preview_row["has_care_plan"] else "なし",
            "日々の記録件数": str(user_preview_row["daily_count"]),
            "モニタリング件数": str(user_preview_row["monitoring_count"]),
        },
    }


def render_export_section():
    st.subheader("結果の出力")
    reviewed = st.session_state.get("reviewed_users", {})
    if not reviewed:
        st.info("利用者を確認すると、ここから結果CSVをダウンロードできます。")
        return

    st.caption(
        "処理結果をCSVでダウンロードできます。個人名や生の記録内容が含まれる可能性があるため、"
        "実運用時には個人情報管理（アクセス制限・保管ルールなど）が必要です。"
    )

    priority_df = build_review_priority_results_df(reviewed)
    obs_df = build_confirmed_observations_df(reviewed)

    col1, col2 = st.columns(2)
    col1.download_button(
        "review_priority_results.csv をダウンロード",
        data=priority_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="review_priority_results.csv", mime="text/csv", key="dl_priority_results",
    )
    col2.download_button(
        "confirmed_observations.csv をダウンロード",
        data=obs_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="confirmed_observations.csv", mime="text/csv", key="dl_confirmed_observations",
    )


# ============================================================
# 画面2: 利用者一覧・詳細
# ============================================================

def page_user_list():
    st.title("利用者一覧・詳細")
    st.caption("「記録から確認優先度まで」画面で処理した利用者（サンプルまたはアップロードCSV）の一覧です。")

    reviewed = st.session_state.get("reviewed_users", {})
    if not reviewed:
        st.info("まだ処理済みの利用者がいません。「1. 記録から確認優先度まで」画面で利用者を確認してください。")
        return

    rows = []
    for uid, info in reviewed.items():
        rows.append(
            {
                "user_id": uid,
                "operational_category": info.get("operational_category", "経過観察"),
                "main_change": info.get("main_change", ""),
                "main_source_type": info.get("main_source_type", "-"),
                "input_status": info.get("input_status", ""),
                "review_status": info.get("review_status", "未確認"),
            }
        )
    summary_df = pd.DataFrame(rows)
    summary_df["_cat_order"] = summary_df["operational_category"].map(OPERATIONAL_CATEGORY_ORDER)
    summary_df = summary_df.sort_values("_cat_order")

    st.subheader("利用者一覧")
    display_df = pd.DataFrame(
        {
            "利用者ID": summary_df["user_id"],
            "運用上の確認区分": summary_df["operational_category"],
            "主な状態変化": summary_df["main_change"],
            "根拠となる情報源": summary_df["main_source_type"],
            "入力情報の状態": summary_df["input_status"],
            "確認状況": summary_df["review_status"],
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.caption(f"表示件数: {len(summary_df)}件（優先確認 → 追加情報確認 → 経過観察の順に表示）")

    st.markdown("---")
    st.subheader("利用者詳細")
    selected_user = st.selectbox("利用者を選択", summary_df["user_id"].tolist())
    info = reviewed[selected_user]

    col1, col2 = st.columns([1, 1])
    with col1:
        basic = info.get("basic_info", {})
        basic_lines = "<br>".join(f"<b>{k}</b>：{v}" for k, v in basic.items()) if basic else "登録情報なし"
        st.markdown(f'<div class="app-card">{basic_lines}</div>', unsafe_allow_html=True)
        st.markdown(render_operational_badge(info["operational_category"]), unsafe_allow_html=True)
        st.caption(f"確認状況：{info.get('review_status', '未確認')}")
    with col2:
        if info.get("structured_available") and info.get("model_risk_level"):
            st.markdown(render_risk_badge(info["model_risk_level"]), unsafe_allow_html=True)
            st.caption(f"モデル上のリスクスコア：{info['model_score']:.3f}")
            render_risk_score_definition()
        else:
            st.markdown(
                '<div class="app-card">構造化評価値が未取得のため、機械学習モデルのスコアは算出していません。</div>',
                unsafe_allow_html=True,
            )

    st.markdown("##### 抽出された観察事項（職員確認済み）")
    obs = info.get("approved_observations", [])
    if obs:
        obs_display = pd.DataFrame(
            [
                {
                    "観察カテゴリ": o["category"],
                    "確認済み内容": o.get("confirmed_text", o["evidence"]),
                    "記録日": str(o.get("record_date", "-")),
                    "情報源": o.get("source_type", "-"),
                    "職員判断": o.get("staff_decision", "-"),
                }
                for o in obs
            ]
        )
        st.dataframe(obs_display, use_container_width=True, hide_index=True)
    else:
        st.caption("職員確認済みの観察事項はありません。")

    st.markdown("---")
    render_share_candidates_section(obs, key_suffix=f"list_{selected_user}")


# ============================================================
# 画面3: モデル検証・技術情報
# ============================================================

def page_technical(
    model, feature_columns, feature_defaults, metadata, demo_cases, feature_importance,
    threshold_metrics, confusion_df, validation_df,
):
    st.title("モデル検証・技術情報")
    st.caption("既存の機械学習モデル（補助的な分析エンジン）の検証デモと技術的な評価指標を確認します。")

    tn = int(confusion_df.loc["actual_0_negative", "pred_0_negative"])
    fp = int(confusion_df.loc["actual_0_negative", "pred_1_positive"])
    fn = int(confusion_df.loc["actual_1_positive", "pred_0_negative"])
    tp = int(confusion_df.loc["actual_1_positive", "pred_1_positive"])
    n_positive = tp + fn
    st.markdown(
        f'<div class="app-card">実際に該当した{n_positive}件のうち{tp}件を検知しました。'
        f'見逃し{fn}件、誤って確認対象としたもの{fp}件です。</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["モデル検証デモ", "評価指標・限界・将来構想", "閾値と確認対象のバランス"])

    with tab1:
        render_model_verification_tab(model, feature_columns, feature_defaults, metadata, demo_cases, feature_importance)
    with tab2:
        render_evaluation_tab(model, feature_columns, feature_defaults, metadata, threshold_metrics, confusion_df)
    with tab3:
        render_threshold_workload_tab(metadata, validation_df)


def render_model_verification_tab(model, feature_columns, feature_defaults, metadata, demo_cases, feature_importance):
    st.markdown(
        """
        <div class="notice-box">
        この画面は、公開データを用いてモデル上のスコア算出の仕組みを確認する技術デモです。
        実際の運用では、職員が毎回すべての項目を手入力する想定ではありません。
        </div>
        """,
        unsafe_allow_html=True,
    )

    case_names = ["（デモケースを選択しない）"] + demo_cases["case_name"].tolist()
    selected_case_name = st.selectbox("デモケースを選択", case_names)

    if selected_case_name in demo_cases["case_name"].values:
        case_row = demo_cases[demo_cases["case_name"] == selected_case_name].iloc[0]
        case_key = case_row["case_id"]
        default_values = {feat: case_row[feat] for feat in MAIN_FEATURE_LABELS}
    else:
        case_key = "manual"
        default_values = {feat: feature_defaults[feat] for feat in MAIN_FEATURE_LABELS}

    ranges = metadata["feature_ranges"]

    with st.form(key="risk_form"):
        col1, col2 = st.columns(2)
        main_values = {}
        with col1:
            main_values["MMSE"] = st.slider(
                MAIN_FEATURE_LABELS["MMSE"], min_value=float(ranges["MMSE"][0]), max_value=float(ranges["MMSE"][1]),
                value=float(default_values["MMSE"]), step=1.0, key=f"mmse_{case_key}",
            )
            main_values["ADL"] = st.slider(
                MAIN_FEATURE_LABELS["ADL"], min_value=float(ranges["ADL"][0]), max_value=float(ranges["ADL"][1]),
                value=float(default_values["ADL"]), step=0.1, key=f"adl_{case_key}",
            )
            main_values["FunctionalAssessment"] = st.slider(
                MAIN_FEATURE_LABELS["FunctionalAssessment"], min_value=float(ranges["FunctionalAssessment"][0]),
                max_value=float(ranges["FunctionalAssessment"][1]), value=float(default_values["FunctionalAssessment"]),
                step=0.1, key=f"fa_{case_key}",
            )
        with col2:
            main_values["PhysicalActivity"] = st.slider(
                MAIN_FEATURE_LABELS["PhysicalActivity"], min_value=float(ranges["PhysicalActivity"][0]),
                max_value=float(ranges["PhysicalActivity"][1]), value=float(default_values["PhysicalActivity"]),
                step=0.1, key=f"pa_{case_key}",
            )
            main_values["MemoryComplaints"] = st.selectbox(
                MAIN_FEATURE_LABELS["MemoryComplaints"], options=[0, 1], index=int(default_values["MemoryComplaints"]),
                format_func=lambda v: "あり" if v == 1 else "なし", key=f"mc_{case_key}",
            )
            main_values["BehavioralProblems"] = st.selectbox(
                MAIN_FEATURE_LABELS["BehavioralProblems"], options=[0, 1], index=int(default_values["BehavioralProblems"]),
                format_func=lambda v: "あり" if v == 1 else "なし", key=f"bp_{case_key}",
            )
        submitted = st.form_submit_button("リスクを算出", type="primary")

    if submitted:
        score, error = predict_risk_score(model, main_values, feature_columns, feature_defaults)
        if error:
            st.error(error)
        else:
            st.session_state["last_prediction"] = {
                "main_inputs": main_values,
                "risk_score": score,
                "risk_level": classify_risk(
                    score, metadata["thresholds"]["selected_threshold"], metadata["thresholds"]["high_risk_threshold"]
                ),
            }

    result = st.session_state.get("last_prediction")
    if result:
        render_verification_result(result, metadata, feature_importance, feature_defaults)
    else:
        st.info("入力項目を確認し、「リスクを算出」ボタンを押すと結果が表示されます。")


def render_verification_result(result, metadata, feature_importance, feature_defaults):
    score, level = result["risk_score"], result["risk_level"]
    sel, high = metadata["thresholds"]["selected_threshold"], metadata["thresholds"]["high_risk_threshold"]

    st.markdown("---")
    st.subheader("算出結果")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("モデル上のリスクスコア", f"{score:.3f}")
        st.markdown(render_risk_badge(level), unsafe_allow_html=True)
    with col2:
        st.progress(min(max(score, 0.0), 1.0))
        st.caption(
            f"検知ライン（{sel:.2f}）との関係: " + ("検知ライン以上" if score >= sel else "検知ライン未満")
            + f" ／ 優先対応ライン（{high:.2f}）との関係: " + ("優先対応ライン以上" if score >= high else "優先対応ライン未満")
        )

    narrative = {
        "Low": "現在の入力条件では、優先度は比較的低いと考えられます。定期的な状態確認を継続する想定です。",
        "Medium": "経過観察や追加確認の候補です。単独のスコアで判断せず、直近の状態変化や専門職の評価と合わせて確認します。",
        "High": "専門職による優先的な確認候補です。医療診断を示すものではなく、追加評価や多職種での確認を検討するための目安です。",
    }
    st.markdown(f'<div class="app-card">{narrative[level]}</div>', unsafe_allow_html=True)

    render_action_candidates(level)
    st.markdown("---")
    render_feature_importance_section(feature_importance)
    st.markdown("---")
    render_input_comparison_section(result["main_inputs"], feature_defaults)


def render_evaluation_tab(model, feature_columns, feature_defaults, metadata, threshold_metrics, confusion_df):
    cv = metadata["cross_validation_5fold"]
    sel, high = metadata["thresholds"]["selected_threshold"], metadata["thresholds"]["high_risk_threshold"]
    test_metrics = metadata["test_set_metrics"]["at_selected_threshold"]

    col1, col2, col3, col4 = st.columns(4)
    for col, label, value in [
        (col1, "ROC-AUC（識別性能）", f"{metadata['test_set_metrics']['roc_auc']:.3f}"),
        (col2, "Recall（再現率・見逃し防止）", f"{test_metrics['recall_1']:.3f}"),
        (col3, "Precision（適合率）", f"{test_metrics['precision_1']:.3f}"),
        (col4, "F1-score（再現率と適合率のバランス）", f"{test_metrics['f1_1']:.3f}"),
    ]:
        col.markdown(f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="notice-box">
        ROC-AUCは、モデルが該当ありと該当なしを順位付けできる性能を示す技術指標です。
        ただし、この数値だけで現場での使いやすさや支援効果が決まるものではありません。
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("交差検証（5-fold）による基本性能を見る"):
        st.write(
            f"- Accuracy: {cv['accuracy']:.3f}\n- Recall: {cv['recall']:.3f}\n- Precision: {cv['precision']:.3f}\n"
            f"- F1-score: {cv['f1']:.3f}\n- ROC-AUC: {cv['roc_auc']:.3f}"
        )
        st.caption("学習用データに対する5分割Stratified K-Foldの平均値です。")

    render_threshold_callout(metadata)

    st.subheader("混同行列（採用閾値時点）")
    display_cm = confusion_df.rename(
        index={"actual_0_negative": "実際：該当なし", "actual_1_positive": "実際：該当あり"},
        columns={"pred_0_negative": "予測：該当なし", "pred_1_positive": "予測：該当あり"},
    )
    st.dataframe(display_cm, use_container_width=True)
    st.caption("行は実際の結果、列はモデルの予測結果を示しています。")

    st.subheader("閾値ごとの評価比較")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(threshold_metrics["threshold"], threshold_metrics["recall_1"], marker="o", label="Recall", markersize=4)
    ax.plot(threshold_metrics["threshold"], threshold_metrics["precision_1"], marker="o", label="Precision", markersize=4)
    ax.plot(threshold_metrics["threshold"], threshold_metrics["f1_1"], marker="o", label="F1-score", markersize=4)
    ax.axvline(sel, linestyle="--", color=TEAL_DARK, label=f"Detection line ({sel:.2f})")
    ax.axvline(high, linestyle="--", color="#b91c1c", label=f"Priority line ({high:.2f})")
    ax.set_xlabel("Threshold", fontsize=9)
    ax.set_ylabel("Score", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title("Threshold vs Evaluation Metrics", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    with st.expander("確率キャリブレーションについて"):
        cal = metadata["calibration"]
        st.markdown(
            f"""
            RandomForestが出力する予測確率は、そのままでは実際の発生率と一致しない場合があります。
            本モデルでは CalibratedClassifierCV（sigmoid法）を用いて補正しています。

            - 補正前 Brier Score: {cal['brier_score_before']:.3f}
            - 補正後 Brier Score: {cal['brier_score_after']:.3f}
            """
        )

    with st.expander(f"{sel:.2f}を採用した理由"):
        st.markdown(
            f"""
            介護・医療の現場では、リスクのある方を見逃す（False Negative）ことの影響が大きいため、
            本分析では「見逃しを防ぐこと（Recallを重視すること）」を優先しました。そのうえで、
            誤って高リスクと判定してしまう件数（False Positive）が増えすぎないよう、
            Precision・F1-scoreとのバランスも確認したうえで、閾値 {sel:.2f} を検知ラインとして採用しています。

            **{sel:.2f}と{high:.2f}は役割が異なります。** {sel:.2f}は経過観察候補を広く拾い上げる検知ライン、
            {high:.2f}は専門職による優先的な確認候補を絞り込む運用ラインです。
            """
        )

    st.markdown("---")
    st.subheader("状態変化シミュレーション")
    render_state_change_simulation(model, feature_columns, feature_defaults, metadata)

    st.markdown("---")
    render_data_gap_section()
    render_future_vision_section()
    st.markdown("---")
    render_limitations()


def render_state_change_simulation(model, feature_columns, feature_defaults, metadata):
    result = st.session_state.get("last_prediction")
    if not result:
        st.info("「モデル検証デモ」タブでリスクを算出すると、その入力条件をもとに状態変化シミュレーションを行えます。")
        return

    base_inputs = result["main_inputs"]
    sel, high = metadata["thresholds"]["selected_threshold"], metadata["thresholds"]["high_risk_threshold"]
    ranges = metadata["feature_ranges"]

    st.markdown(
        f"""
        <div class="app-card">
        現在の入力条件（モデル検証デモで算出したもの）を基準に、ADLおよびFunctionalAssessmentを
        仮に変更した場合のスコア変化を確認します。現在のリスクスコア: <b>{result['risk_score']:.3f}</b>（{result['risk_level']}）
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form(key="simulation_form"):
        col1, col2 = st.columns(2)
        with col1:
            new_adl = st.slider("変更後のADL", min_value=float(ranges["ADL"][0]), max_value=float(ranges["ADL"][1]), value=float(base_inputs["ADL"]), step=0.1)
        with col2:
            new_fa = st.slider(
                "変更後のFunctionalAssessment", min_value=float(ranges["FunctionalAssessment"][0]),
                max_value=float(ranges["FunctionalAssessment"][1]), value=float(base_inputs["FunctionalAssessment"]), step=0.1,
            )
        run_sim = st.form_submit_button("状態変化シミュレーションを実行", type="primary")

    st.markdown(
        """
        <div class="warn-box">
        このシミュレーションは、入力値を仮に変更した場合のモデル出力変化を確認するものです。
        実際の支援・介入による改善効果や因果関係を示すものではありません。
        </div>
        """,
        unsafe_allow_html=True,
    )

    if run_sim:
        new_inputs = dict(base_inputs)
        new_inputs["ADL"] = new_adl
        new_inputs["FunctionalAssessment"] = new_fa
        new_score, error = predict_risk_score(model, new_inputs, feature_columns, feature_defaults)
        if error:
            st.error(error)
        else:
            new_level = classify_risk(new_score, sel, high)
            diff = new_score - result["risk_score"]
            st.subheader("シミュレーション結果")
            c1, c2, c3 = st.columns(3)
            c1.metric("変更前のスコア", f"{result['risk_score']:.3f}", help=result["risk_level"])
            c2.metric("変更後のスコア", f"{new_score:.3f}", delta=f"{diff:+.3f}")
            c3.markdown(
                f'<div class="metric-card"><div class="label">区分の変化</div>'
                f'<div class="value" style="font-size:1.1rem;">{result["risk_level"]} → {new_level}</div></div>',
                unsafe_allow_html=True,
            )
            change_df = pd.DataFrame(
                {"項目名": ["ADL", "FunctionalAssessment"], "変更前": [base_inputs["ADL"], base_inputs["FunctionalAssessment"]], "変更後": [new_adl, new_fa]}
            )
            st.dataframe(change_df, use_container_width=True, hide_index=True)


def render_threshold_workload_tab(metadata: dict, validation_df: pd.DataFrame):
    st.markdown(
        """
        <div class="app-card">
        閾値を低くすると、見逃しは減りますが確認対象が増えます。閾値を高くすると、確認対象は
        減りますが、見逃しが増える可能性があります。
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_threshold_callout(metadata)

    sel_default = metadata["thresholds"]["selected_threshold"]
    y_true = validation_df["y_true"].values
    scores = validation_df["risk_score"].values
    n_eval = len(y_true)
    n_eval_positive = int((y_true == 1).sum())

    col1, col2 = st.columns(2)
    with col1:
        facility_size = st.number_input("事業所の利用者数（人）", min_value=1, max_value=1000, value=30, step=1)
    with col2:
        threshold = st.slider("検知閾値", min_value=0.05, max_value=0.95, value=float(sel_default), step=0.05)

    pred_positive_mask = scores >= threshold
    tp_eval = int(((y_true == 1) & pred_positive_mask).sum())
    fn_eval = int(((y_true == 1) & (~pred_positive_mask)).sum())
    fp_eval = int(((y_true == 0) & pred_positive_mask).sum())
    total_positive_pred = tp_eval + fp_eval
    predicted_positive_rate = float(pred_positive_mask.mean())

    st.subheader(f"評価データ{n_eval}件での実測値")
    d1, d2, d3, d4 = st.columns(4)
    for col, label, value in [
        (d1, "検知できた件数", f"{tp_eval}件"),
        (d2, "見逃し件数", f"{fn_eval}件"),
        (d3, "誤って確認対象とした件数", f"{fp_eval}件"),
        (d4, "確認対象となった総数", f"{total_positive_pred}件"),
    ]:
        col.markdown(f'<div class="stat-card"><div class="label">{label}</div><div class="value">{value}</div></div>', unsafe_allow_html=True)
    st.caption(f"評価データ（実際に該当あり{n_eval_positive}件を含む計{n_eval}件）に、閾値{threshold:.2f}を適用した場合の実測件数です。")

    st.subheader("架空事業所での仮説試算")
    st.markdown(
        """
        <div class="warn-box">
        以下は評価データの比率を事業所の利用者数に当てはめたデモ用仮説であり、
        実証結果や実際の事業所データではありません。
        </div>
        """,
        unsafe_allow_html=True,
    )
    review_count_sim = round(predicted_positive_rate * facility_size)
    ratio_pct = (review_count_sim / facility_size * 100) if facility_size else 0
    h1, h2, h3 = st.columns(3)
    for col, label, value in [
        (h1, "入力した事業所利用者数", f"{facility_size}人"),
        (h2, "想定される確認候補人数", f"{review_count_sim}人"),
        (h3, "全利用者に占める割合", f"{ratio_pct:.0f}%"),
    ]:
        col.markdown(f'<div class="stat-card"><div class="label">{label}</div><div class="value">{value}</div></div>', unsafe_allow_html=True)


# ============================================================
# メイン
# ============================================================

def main():
    inject_css()
    check_required_files()
    st.session_state.setdefault("reviewed_users", {})

    model = load_model()
    feature_columns = load_feature_columns()
    feature_defaults = load_feature_defaults()
    metadata = load_metadata()
    feature_importance = load_feature_importance()
    threshold_metrics = load_threshold_metrics()
    confusion_df = load_confusion_matrix()
    validation_df = load_validation_predictions()
    demo_cases = load_demo_cases()
    care_df = load_care_record_demo()

    page = render_sidebar()

    if page == PAGES[0]:
        page_record_to_priority(model, feature_columns, feature_defaults, metadata, care_df)
    elif page == PAGES[1]:
        page_user_list()
    elif page == PAGES[2]:
        page_technical(
            model, feature_columns, feature_defaults, metadata, demo_cases, feature_importance,
            threshold_metrics, confusion_df, validation_df,
        )


if __name__ == "__main__":
    main()
