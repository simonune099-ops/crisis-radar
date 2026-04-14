# scorer.py
# Loughran-McDonald NLP sentiment scoring + Crisis Exposure Score
# Corporate Disclosure Crisis Radar — AC820/BA870
# 中文: LM 词典情绪打分 + 危机暴露分计算模块

import re
import pandas as pd
from pathlib import Path

# ── Default word sets (development / fallback) ──────────────────────────────
# 中文: 简化版词集，用于开发测试。上传真实 LM 词典 CSV 后自动替换。
_FALLBACK_DICT = {
    "positive":     {"STRONG", "GROWTH", "IMPROVED", "RECORD", "EXCEEDED",
                     "SUCCESSFUL", "COMMITTED", "CONFIDENT", "INCREASE",
                     "GAIN", "PROFITABLE", "ENHANCE", "ACHIEVE", "BENEFIT"},
    "negative":     {"DECLINE", "LOSS", "RISK", "ADVERSE", "IMPAIR",
                     "DIFFICULT", "CHALLENGING", "DECREASE", "FAILED",
                     "CONCERN", "WEAK", "DEFICIT", "SHORTFALL", "WRITE"},
    "uncertainty":  {"MAY", "MIGHT", "COULD", "UNCERTAIN", "POSSIBLE",
                     "POTENTIAL", "SUBJECT", "APPROXIMATELY", "EXPECTED",
                     "LIKELY", "UNCLEAR", "DEPEND", "CONTINGENT", "VARIABLE"},
    "litigious":    {"LITIGATION", "LEGAL", "REGULATORY", "INVESTIGATION",
                     "LAWSUIT", "CLAIM", "PROCEEDING", "ALLEGED", "DISPUTE",
                     "PENALTY", "VIOLATION", "ENFORCEMENT", "COMPLAINT"},
    "weak_modal":   {"POSSIBLY", "PERHAPS", "SOMETIMES", "GENERALLY",
                     "APPEAR", "SEEM", "SUGGEST", "BELIEVE", "AROUND",
                     "ROUGHLY", "TYPICALLY", "ORDINARILY"},
    "constraining": {"REQUIRED", "OBLIGATED", "RESTRICTED", "CONSTRAINED",
                     "CANNOT", "SHALL", "PROHIBITED", "MUST", "COVENANT",
                     "MANDATE", "COMPELLED", "OBLIGED"},
    "strong_modal": {"WILL", "DEFINITELY", "CERTAINLY", "CLEARLY",
                     "UNDOUBTEDLY", "ASSURE", "GUARANTEE", "COMMIT",
                     "ENSURE", "AFFIRM"},
}

# Crisis Exposure Score weights (grounded in Loughran & McDonald 2011)
# 中文: 危机暴露分权重，依据 LM 2011 年 Journal of Finance 研究结果设定
_CRISIS_WEIGHTS = {
    "uncertainty_pct":  0.30,   # 79% increase in litigation odds
    "litigious_pct":    0.25,   # 38% change in litigation odds
    "negative_pct":     0.20,   # core adverse signal
    "weak_modal_pct":   0.15,   # management conviction proxy
    "constraining_pct": 0.10,   # regulatory pressure signal
}


def load_lm_dictionary(csv_path: str = "data/LM_MasterDictionary.csv") -> dict:
    """
    Load the real Loughran-McDonald Master Dictionary from CSV.
    Falls back to the built-in simplified word sets if the file is not found.

    Download the CSV from: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
    Place it at: data/LM_MasterDictionary.csv

    中文: 加载真实 LM 词典。文件不存在时自动用简化版词集。
    """
    path = Path(csv_path)
    if not path.exists():
        print(f"[scorer] LM Dictionary not found at '{csv_path}'. "
              f"Using fallback word sets.")
        return _FALLBACK_DICT

    df = pd.read_csv(path)
    df["Word"] = df["Word"].str.upper()

    word_sets = {
        "positive":     set(df[df["Positive"]     != 0]["Word"]),
        "negative":     set(df[df["Negative"]     != 0]["Word"]),
        "uncertainty":  set(df[df["Uncertainty"]  != 0]["Word"]),
        "litigious":    set(df[df["Litigious"]    != 0]["Word"]),
        "weak_modal":   set(df[df["Weak_Modal"]   != 0]["Word"]),
        "constraining": set(df[df["Constraining"] != 0]["Word"]),
        "strong_modal": set(df[df["Strong_Modal"] != 0]["Word"]),
    }
    total = sum(len(v) for v in word_sets.values())
    print(f"[scorer] LM Dictionary loaded: {total:,} entries across 7 dimensions.")
    return word_sets


def tokenize(text: str) -> list[str]:
    """
    Tokenize text into uppercase words, stripping numbers and punctuation.
    中文: 将文本拆分为大写单词列表，去除数字和标点
    """
    return [w.upper() for w in re.findall(r"[A-Za-z]+", text)]


def score_text(text: str, word_sets: dict) -> dict:
    """
    Score a filing text across all LM dimensions.

    Returns a dict containing:
      - total_words            : total word count
      - {dim}_count            : raw match count per dimension
      - {dim}_pct              : matches per 1,000 words (comparable across filings)
      - net_sentiment          : (positive - negative) / total * 100
      - crisis_score           : weighted Crisis Exposure Score (higher = more risk)

    中文: 对文本打分，返回各维度词频（每千词）和综合危机暴露分
    """
    tokens = tokenize(text)
    total  = len(tokens)
    if total == 0:
        return {}

    counts = {
        dim: sum(1 for w in tokens if w in words)
        for dim, words in word_sets.items()
    }

    results = {"total_words": total}
    for dim, count in counts.items():
        results[f"{dim}_count"] = count
        results[f"{dim}_pct"]   = round(count / total * 1000, 2)

    # Net sentiment: positive dominance over negative
    # 中文: 净情绪分，正数=偏乐观，负数=偏谨慎
    pos = counts.get("positive", 0)
    neg = counts.get("negative", 0)
    results["net_sentiment"] = round((pos - neg) / total * 100, 4)

    # Crisis Exposure Score
    # 中文: 危机暴露分，综合 5 个风险维度的加权总分
    results["crisis_score"] = round(
        sum(results.get(key, 0) * weight
            for key, weight in _CRISIS_WEIGHTS.items()),
        4
    )
    return results


def assign_rating(crisis_score: float) -> tuple[str, str, str]:
    """
    Convert a Crisis Exposure Score to an A–D PR risk rating.
    Returns (letter, hex_color, description).

    Thresholds will be recalibrated with historical data in Phase 2
    using Z-score normalization against sector benchmarks.
    中文: 把危机分转为 A-D 评级，Phase 2 用行业历史数据校准阈值
    """
    if   crisis_score < 2.0: return "A", "#2E7D32", "Low Risk — Transparent, consistent disclosure"
    elif crisis_score < 4.0: return "B", "#F9A825", "Moderate Risk — Monitor selected signals"
    elif crisis_score < 6.5: return "C", "#E65100", "Elevated Risk — Multiple indicators triggered"
    else:                    return "D", "#B71C1C", "High Risk — Strong crisis signals detected"


def get_scct_guidance(scores: dict, pr_divergence: float | None = None) -> dict:
    """
    Map scoring results to the SCCT crisis communication framework.
    (Coombs 2007: Victim / Accidental / Intentional × Deny / Diminish / Rebuild)

    Optionally incorporates PR tone divergence from Access Newswire.
    中文: 把评分结果映射到 SCCT 危机沟通框架，给出应对策略建议
    """
    lit  = scores.get("litigious_pct",   0)
    unc  = scores.get("uncertainty_pct", 0)
    neg  = scores.get("negative_pct",    0)
    weak = scores.get("weak_modal_pct",  0)

    if lit > 3.0:
        guidance = {
            "crisis_type": "Intentional",
            "cluster":     "Preventable",
            "strategy":    "Rebuild — Accept responsibility",
            "action": (
                "Litigious language is significantly elevated. "
                "Issue a formal public statement with concrete corrective actions. "
                "Avoid ambiguous language. LP/investor trust is at high risk."
            ),
            "lp_signal": "HIGH — Proactive LP communication required before next board meeting.",
            "color":     "#B71C1C",
        }
    elif unc > 5.0 and weak > 2.0:
        guidance = {
            "crisis_type": "Accidental",
            "cluster":     "Accidental",
            "strategy":    "Diminish — Reduce responsibility attribution",
            "action": (
                "Management uncertainty language is disproportionately high. "
                "Strengthen next disclosure with specific metrics and timelines. "
                'Reduce hedge words: "may", "might", "possibly", "could".'
            ),
            "lp_signal": "MODERATE — Increase update cadence; LPs may seek reassurance.",
            "color":     "#E65100",
        }
    elif neg > 4.0:
        guidance = {
            "crisis_type": "Victim",
            "cluster":     "Victim",
            "strategy":    "Deny / Clarify — Reframe external attribution",
            "action": (
                "Negative language is elevated but the firm is not the primary actor. "
                "Proactively contextualize external conditions in IR communications "
                "to prevent adverse media framing."
            ),
            "lp_signal": "MODERATE — Narrative framing is the key risk variable.",
            "color":     "#F9A825",
        }
    else:
        guidance = {
            "crisis_type": "No significant crisis signals",
            "cluster":     "N/A",
            "strategy":    "Bolstering — Build reputational capital",
            "action": (
                "Disclosure language risk is low. "
                "Use this window to proactively strengthen LP/investor relationships "
                "and build reputational reserves before any future turbulence."
            ),
            "lp_signal": "LOW — Favorable window for new fundraising communications.",
            "color":     "#2E7D32",
        }

    # Append divergence warning if applicable
    # 中文: 如果 SEC 文件和新闻稿语气不一致，追加警告
    if pr_divergence is not None and pr_divergence > 4.0:
        guidance["action"] += (
            f" ⚠️ Messaging inconsistency detected: tone divergence score "
            f"{pr_divergence:.2f} between SEC filing and press releases — "
            f"this is an independent PR risk signal."
        )

    return guidance
