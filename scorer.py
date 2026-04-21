# scorer.py
# Loughran-McDonald NLP sentiment scoring + Crisis Exposure Score
# Corporate Disclosure Crisis Radar — AC820/BA870
# 中文: LM 词典情绪打分 + 危机暴露分计算模块
# Enhanced with: Lerbinger Crisis Types, Crisis Lifecycle, Victim Recovery Cycle,
#                Proactive Checklist, Issues Management Triage (Crisis Ready logic)

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


# ════════════════════════════════════════════════════════════════════════════
# LERBINGER CRISIS TYPE CLASSIFICATION
# Source: Lerbinger (1997), adapted from AC820 Week 2 materials
# 中文: Lerbinger 危机类型分类（来自 Week 2 课件）
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# TOP TRIGGER SENTENCES
# Surfaces the actual filing sentences driving the high score,
# so users can see exactly what text is causing the rating.
# Professor feedback: "it'd be interesting to know what exactly within a
# 10-K or news is driving a result."
# 中文: 提取导致高评分的具体句子，回应教授"是什么文字驱动了评级"的问题
# ════════════════════════════════════════════════════════════════════════════

def extract_top_trigger_sentences(text: str, word_sets: dict, n: int = 6) -> list[dict]:
    """
    Split the filing text into sentences and score each one.
    Returns the top-n sentences ranked by crisis exposure score,
    along with the flagged words and dimensions detected.

    Each result: {sentence, crisis_score, flagged_words, dimensions}
    中文: 对每个句子打分，返回危机分最高的 n 句，并标注触发词
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.split()) >= 6]  # skip fragments

    scored = []
    risk_dims = ["negative", "uncertainty", "litigious", "weak_modal", "constraining"]

    for sent in sentences:
        tokens = tokenize(sent)
        if not tokens:
            continue
        total = len(tokens)

        flagged_words = []
        dim_hits = {}
        for dim in risk_dims:
            words_in_dim = word_sets.get(dim, set())
            hits = [w for w in tokens if w in words_in_dim]
            if hits:
                dim_hits[dim] = hits
                flagged_words.extend(hits)

        if not flagged_words:
            continue

        # Crisis score for this sentence
        crisis = sum(
            (len(dim_hits.get(d, [])) / total * 1000) * w
            for d, w in _CRISIS_WEIGHTS.items()
            if d.replace("_pct", "") in dim_hits or d.replace("_pct","") in dim_hits
        )
        # Simpler direct calculation
        crisis_score = sum(
            (len(dim_hits.get(dim, [])) / total * 1000) * weight
            for dim, weight in {
                "uncertainty": 0.30, "litigious": 0.25, "negative": 0.20,
                "weak_modal": 0.15, "constraining": 0.10,
            }.items()
        )

        scored.append({
            "sentence":    sent[:300],   # truncate very long sentences
            "crisis_score": round(crisis_score, 3),
            "flagged_words": list(set(flagged_words))[:8],
            "dimensions":   list(dim_hits.keys()),
        })

    # Return top-n by crisis score
    scored.sort(key=lambda x: x["crisis_score"], reverse=True)
    return scored[:n]


def detect_short_seller_signal(news_df) -> dict:
    """
    Scan Yahoo Finance news headlines for mentions of known activist
    short sellers (Hindenburg, Muddy Waters, Citron, etc.).
    Returns a dict with: detected (bool), firms_mentioned, headlines

    Professor feedback: "you may want to split NLP signals based on the
    source — whether it comes from short sellers"
    中文: 从新闻标题检测知名做空机构的报告信号
    做空机构报告是独立于公司官方披露之外的最强负面外部信号
    """
    # Known activist short seller firms (Willkie 2024 report + major others)
    # 中文: 主要做空机构名单（来自 Willkie Farr 2024 报告）
    SHORT_SELLERS = [
        "hindenburg", "muddy waters", "citron", "grizzly research",
        "iceberg research", "viceroy", "blue orca", "jehoshaphat",
        "short seller", "short report", "short position", "short attack",
        "activist short", "fraud allegations", "accounting fraud",
        "overstated revenue", "fabricated", "whistleblower",
    ]

    if news_df is None or len(news_df) == 0:
        return {"detected": False, "firms_mentioned": [], "headlines": []}

    titles = news_df.get("title", news_df.get("Title", []))
    if hasattr(titles, "tolist"):
        titles = titles.tolist()

    detected_firms = []
    alert_headlines = []

    for title in titles:
        title_lower = str(title).lower()
        for firm in SHORT_SELLERS:
            if firm in title_lower:
                if firm not in detected_firms:
                    detected_firms.append(firm)
                if title not in alert_headlines:
                    alert_headlines.append(title)
                break

    return {
        "detected":        len(detected_firms) > 0,
        "firms_mentioned": detected_firms,
        "headlines":       alert_headlines[:5],
        "severity":        "🔴 CRITICAL" if any(f in detected_firms for f in
                           ["hindenburg","muddy waters","citron","grizzly research",
                            "viceroy","blue orca"]) else
                           "🟠 HIGH" if detected_firms else "🟢 None",
    }


def classify_lerbinger_type(scores: dict) -> dict:
    """
    Classify the filing into Lerbinger's crisis typology based on LM scores.

    Four types (mapped to LM signal patterns):
      1. Mismanagement / Misconduct  — high litigious + high constraining
      2. Stakeholder Confrontation   — high litigious + high negative (external framing)
      3. Technological Failure       — high uncertainty + high constraining (technical unknowns)
      4. Environment & Sustainability — high uncertainty + high negative (external shocks)

    Returns a dict with: type_name, description, root_vulnerability,
                         exacerbating_factors, lm_signal_basis

    中文: 根据 LM 维度打分，判断属于哪种 Lerbinger 危机类型
    Lerbinger 四类：管理失当、利益相关者对抗、技术失败、环境/可持续性
    """
    lit  = scores.get("litigious_pct",    0)
    con  = scores.get("constraining_pct", 0)
    neg  = scores.get("negative_pct",     0)
    unc  = scores.get("uncertainty_pct",  0)
    weak = scores.get("weak_modal_pct",   0)

    # Score each type
    # 中文: 为每种危机类型计算匹配分
    type_scores = {
        "Mismanagement / Misconduct":   lit * 1.5 + con * 1.2,
        "Stakeholder Confrontation":    lit * 1.2 + neg * 1.0,
        "Technological Failure":        unc * 1.2 + con * 1.0 + weak * 0.5,
        "Environment & Sustainability": unc * 1.0 + neg * 1.2,
    }

    best_type = max(type_scores, key=type_scores.get)
    best_score = type_scores[best_type]

    type_details = {
        "Mismanagement / Misconduct": {
            "icon": "⚖️",
            "description": (
                "Filing language suggests regulatory scrutiny or management conduct issues. "
                "High litigious and constraining language signals legal exposure "
                "and possible governance failures."
            ),
            "root_vulnerability": "Internal — Management & Governance",
            "exacerbating_factors": ["Social media amplification", "Regulatory enforcement pressure",
                                     "Decision-making bias (optimism bias)"],
            "lm_signal_basis": f"Litigious: {lit:.1f}/1000 words | Constraining: {con:.1f}/1000 words",
            "stealing_thunder_tip": (
                "Consider proactively disclosing corrective governance actions before "
                "regulators announce — 'Stealing Thunder' reduces reputational damage by "
                "up to 35% (Arpan & Roskos-Ewoldsen, 2005)."
            ),
        },
        "Stakeholder Confrontation": {
            "icon": "🤝",
            "description": (
                "Filing signals external stakeholder pressure — regulators, activists, "
                "or institutional investors may be in adversarial positions. "
                "High negative language combined with litigation signals an external conflict."
            ),
            "root_vulnerability": "External — Stakeholder & Political Forces",
            "exacerbating_factors": ["Globalization (cross-border regulatory complexity)",
                                     "Social forces (activist movements, ESG scrutiny)",
                                     "Media amplification"],
            "lm_signal_basis": f"Litigious: {lit:.1f}/1000 words | Negative: {neg:.1f}/1000 words",
            "stealing_thunder_tip": (
                "Engage key stakeholders directly before public escalation. "
                "Prepare a stakeholder communication map (identify, prioritize by impact, "
                "assign communication leads)."
            ),
        },
        "Technological Failure": {
            "icon": "⚙️",
            "description": (
                "Elevated uncertainty and constraining language suggests operational or "
                "technical unknowns — system failures, supply chain disruptions, or "
                "product/process deficiencies that management cannot fully characterize yet."
            ),
            "root_vulnerability": "Internal — Operations & Technology",
            "exacerbating_factors": ["Supply chain complexity", "Human error in decision-making",
                                     "Globalization (cross-border operational risk)"],
            "lm_signal_basis": f"Uncertainty: {unc:.1f}/1000 words | Constraining: {con:.1f}/1000 words",
            "stealing_thunder_tip": (
                "Disclose technical issues with specific remediation timelines. "
                "Vague language ('we are monitoring') significantly increases investor anxiety — "
                "replace with concrete metrics and milestones."
            ),
        },
        "Environment & Sustainability": {
            "icon": "🌍",
            "description": (
                "The filing reflects exposure to external environmental or macro shocks — "
                "climate events, regulatory ESG mandates, or economic force majeure. "
                "The organization is positioned as a victim of external forces."
            ),
            "root_vulnerability": "External — Environment & Economic Forces",
            "exacerbating_factors": ["Climate impact", "Political/legal forces",
                                     "Economic forces (recession, inflation, supply disruption)"],
            "lm_signal_basis": f"Uncertainty: {unc:.1f}/1000 words | Negative: {neg:.1f}/1000 words",
            "stealing_thunder_tip": (
                "Proactively frame external attribution with evidence — avoid being perceived "
                "as deflecting. Contextualize macro impacts with specific company response actions."
            ),
        },
    }

    result = type_details[best_type].copy()
    result["type_name"] = best_type
    result["confidence"] = round(best_score, 2)

    # Secondary type if close
    # 中文: 如果第二高类型分数接近，列出作为次要警示
    sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_types) > 1 and sorted_types[1][1] > best_score * 0.7:
        result["secondary_type"] = sorted_types[1][0]
    else:
        result["secondary_type"] = None

    return result


# ════════════════════════════════════════════════════════════════════════════
# CRISIS LIFECYCLE STAGE
# Source: Crandall (2013) lifecycle model, AC820 Week 1 & Week 3
# Preconditions → Trigger Event → Crisis → Post-Crisis
# 中文: 危机生命周期阶段判断（Week 1 课件）
# ════════════════════════════════════════════════════════════════════════════

def get_lifecycle_stage(scores: dict, rating: str) -> dict:
    """
    Infer where this filing likely sits in the crisis lifecycle.

    Preconditions: low-level signals building up (Issues Management territory)
    Trigger Event: sharp spike in key signals — the tipping point
    Active Crisis: multiple dimensions simultaneously elevated
    Post-Crisis:   high constraining + some positive recovery language

    中文: 根据评分推断公司目前处于危机生命周期的哪个阶段
    """
    lit  = scores.get("litigious_pct",    0)
    unc  = scores.get("uncertainty_pct",  0)
    neg  = scores.get("negative_pct",     0)
    con  = scores.get("constraining_pct", 0)
    pos  = scores.get("positive_pct",     0)
    weak = scores.get("weak_modal_pct",   0)

    # Count how many dimensions are above baseline
    # 中文: 统计多少维度同时超出基准线
    elevated_dims = sum([
        lit > 2.5, unc > 5.0, neg > 3.5, con > 1.5, weak > 2.0
    ])

    if rating == "A":
        stage = "Preconditions / Business as Usual"
        icon  = "🟢"
        description = (
            "Disclosure language is within normal parameters. "
            "This is the optimal window for preventive action — "
            "conduct vulnerability audits and stress-test crisis plans "
            "while reputational capital is intact."
        )
        action_phase = "BEFORE"
        urgency = "Proactive"

    elif rating == "B":
        stage = "Issues Management Territory"
        icon  = "🟡"
        description = (
            "Signals are elevated but have not reached crisis threshold. "
            "This is the Issues Management zone — early intervention now can "
            "prevent escalation to a full crisis. "
            "Apply the Crisis Ready® triage: Is it emotionally charged? "
            "Does it risk relationships or bottom line?"
        )
        action_phase = "BEFORE / EARLY"
        urgency = "Monitor & Intervene"

    elif rating == "C" and elevated_dims <= 2:
        stage = "Trigger Event — Escalation Risk"
        icon  = "🟠"
        description = (
            "Multiple risk dimensions are simultaneously elevated — "
            "this filing may represent a trigger event. "
            "The window between trigger and full crisis is narrow. "
            "Convene the Crisis Management Team (CMT) for assessment. "
            "Do not wait for media coverage to force a response."
        )
        action_phase = "DURING (Early)"
        urgency = "Immediate CMT Activation"

    elif rating in ("C", "D"):
        stage = "Active Crisis / Crisis Management"
        icon  = "🔴"
        description = (
            "Filing language profile is consistent with an active or escalating crisis. "
            f"{elevated_dims} risk dimensions are simultaneously elevated. "
            "The Crisis Management Team (CMT) should be fully activated. "
            "Prioritize: (1) internal stakeholders first, (2) external stakeholders, "
            "(3) consistent single-voice communications, (4) hourly status updates."
        )
        action_phase = "DURING"
        urgency = "Full Crisis Protocol"

    else:
        # High constraining but recovering positive — post-crisis pattern
        if con > 2.0 and pos > 3.0:
            stage = "Post-Crisis — Recovery & Learning"
            icon  = "🔵"
            description = (
                "Language pattern suggests the organization is in recovery mode — "
                "constraining language reflects regulatory remediation while "
                "positive language indicates recovery efforts. "
                "Focus: post-crisis debrief, organizational learning, "
                "and rebuilding stakeholder trust."
            )
            action_phase = "AFTER"
            urgency = "Stabilize & Learn"
        else:
            stage = "Issues Management Territory"
            icon  = "🟡"
            description = (
                "Signals are elevated but not at crisis threshold. "
                "Apply issues management protocols now."
            )
            action_phase = "BEFORE / EARLY"
            urgency = "Monitor & Intervene"

    return {
        "stage": stage,
        "icon":  icon,
        "description": description,
        "action_phase": action_phase,
        "urgency": urgency,
        "elevated_dimensions": elevated_dims,
    }


# ════════════════════════════════════════════════════════════════════════════
# VICTIM RECOVERY CYCLE GUIDANCE
# Source: Coombs SCCT (2007) + Victim Recovery Cycle, AC820 Week 1 & Week 6
# Feelings → Seeking Retribution → Search for Healing → Victim's Needs
# 中文: 受害者恢复循环 + SCCT 整合框架（Week 1 + Week 6）
# ════════════════════════════════════════════════════════════════════════════

def get_scct_guidance(scores: dict, pr_divergence: float | None = None) -> dict:
    """
    Map scoring results to the SCCT crisis communication framework.
    Enhanced with Victim Recovery Cycle (Feelings → Retribution → Healing → Needs)
    and Lerbinger crisis type integration.
    (Coombs 2007: Victim / Accidental / Intentional × Deny / Diminish / Rebuild)

    中文: 把评分结果映射到 SCCT 框架 + 受害者恢复循环，给出分层应对策略
    """
    lit  = scores.get("litigious_pct",   0)
    unc  = scores.get("uncertainty_pct", 0)
    neg  = scores.get("negative_pct",    0)
    weak = scores.get("weak_modal_pct",  0)
    con  = scores.get("constraining_pct", 0)

    if lit > 3.0:
        guidance = {
            "crisis_type":   "Intentional / Preventable",
            "cluster":       "Preventable",
            "strategy":      "Rebuild — Accept responsibility, take corrective action",
            "advocacy_accommodation": "High Accommodation",
            "action": (
                "Litigious language is significantly elevated — regulatory or legal exposure is high. "
                "Issue a formal public statement with concrete corrective actions and timelines. "
                "Avoid ambiguous or hedging language. LP/investor trust is at high risk."
            ),
            "victim_recovery": {
                "stage_1_feelings": (
                    "Stakeholders are likely experiencing anger and distrust. "
                    "Acknowledge their concerns explicitly — do NOT lead with facts or defenses."
                ),
                "stage_2_retribution": (
                    "Expect stakeholders to seek accountability. "
                    "Proactively identify who is responsible and what consequences follow."
                ),
                "stage_3_healing": (
                    "Offer concrete remediation: financial compensation, operational changes, "
                    "or structural governance reforms."
                ),
                "stage_4_needs": (
                    "Ultimate stakeholder needs: Safety, Dignity, Compensation, Truth, "
                    "and Non-recurrence. Address each explicitly in communications."
                ),
            },
            "lp_signal": "🔴 HIGH — Proactive LP communication required before next board meeting.",
            "color":     "#B71C1C",
        }
    elif unc > 5.0 and weak > 2.0:
        guidance = {
            "crisis_type":   "Accidental",
            "cluster":       "Accidental",
            "strategy":      "Diminish — Reduce responsibility attribution, provide context",
            "advocacy_accommodation": "Moderate Accommodation",
            "action": (
                "Management uncertainty language is disproportionately high. "
                "Strengthen next disclosure with specific metrics and timelines. "
                'Reduce hedge words: "may," "might," "possibly," "could." '
                "Replace with specific milestones: 'We expect X by Q3.'"
            ),
            "victim_recovery": {
                "stage_1_feelings": (
                    "Stakeholders feel anxious and uncertain about the company's direction. "
                    "Lead with reassurance — demonstrate command of the situation."
                ),
                "stage_2_retribution": (
                    "Stakeholders may blame management for poor foresight. "
                    "Provide clear evidence of early warning systems and response protocols."
                ),
                "stage_3_healing": (
                    "Offer specific timelines and metrics for resolution. "
                    "Establish a regular update cadence — uncertainty is the enemy of trust."
                ),
                "stage_4_needs": (
                    "Stakeholders need clarity, predictability, and evidence that leadership "
                    "is in control. Avoid 'we are monitoring the situation' language."
                ),
            },
            "lp_signal": "🟠 MODERATE — Increase update cadence; LPs may seek reassurance.",
            "color":     "#E65100",
        }
    elif neg > 4.0:
        guidance = {
            "crisis_type":   "Victim",
            "cluster":       "Victim",
            "strategy":      "Deny / Clarify — Reframe external attribution",
            "advocacy_accommodation": "Low-Moderate Accommodation",
            "action": (
                "Negative language is elevated but the firm appears to be reacting to external forces. "
                "Proactively contextualize external conditions in IR communications "
                "to prevent adverse media framing. Lead with what the company controls."
            ),
            "victim_recovery": {
                "stage_1_feelings": (
                    "Stakeholders are concerned but may not yet attribute blame to management. "
                    "Acknowledge the external impact clearly and with empathy."
                ),
                "stage_2_retribution": (
                    "Risk is lower than intentional crises — but stakeholders will still "
                    "ask 'why weren't you better prepared?' Have a readiness answer ready."
                ),
                "stage_3_healing": (
                    "Demonstrate the company's resilience plan. "
                    "What buffering mechanisms exist? (inventory, insurance, alternate suppliers?)"
                ),
                "stage_4_needs": (
                    "Stakeholders need to know you have a plan. "
                    "Focus communications on your response, not the external event itself."
                ),
            },
            "lp_signal": "🟡 MODERATE — Narrative framing is the key risk variable.",
            "color":     "#F9A825",
        }
    else:
        guidance = {
            "crisis_type":   "No significant crisis signals",
            "cluster":       "N/A",
            "strategy":      "Bolstering — Build reputational capital proactively",
            "advocacy_accommodation": "Advocacy Mode",
            "action": (
                "Disclosure language risk is low. "
                "This is your window to build reputational reserves. "
                "Proactively strengthen LP/investor relationships, "
                "publish thought leadership, and conduct vulnerability audits "
                "before the next potential turbulence."
            ),
            "victim_recovery": {
                "stage_1_feelings": "N/A — No active stakeholder distress signals detected.",
                "stage_2_retribution": "N/A — No active accountability pressure.",
                "stage_3_healing": "Focus on preventive relationship-building with all stakeholder groups.",
                "stage_4_needs": (
                    "Invest in ongoing good PR: 'Good PR is the best crisis insurance.' "
                    "(PR Page Principle: Do the right thing. Listen. Fix small issues early.)"
                ),
            },
            "lp_signal": "🟢 LOW — Favorable window for new fundraising communications.",
            "color":     "#2E7D32",
        }

    # Append divergence warning if applicable
    # 中文: 如果 SEC 文件和新闻稿语气不一致，追加警告
    if pr_divergence is not None and pr_divergence > 4.0:
        guidance["action"] += (
            f" ⚠️ Messaging inconsistency detected: tone divergence score "
            f"{pr_divergence:.2f} between SEC filing and 8-K press releases — "
            f"this is an independent PR risk signal (Stealing Thunder principle: "
            f"your public narrative should lead, not lag, your SEC disclosures)."
        )

    return guidance


# ════════════════════════════════════════════════════════════════════════════
# PROACTIVE ACTION CHECKLIST
# Source: AC820 Week 3 "Proactive Approach" + Week 2 External Scanning logic
# Vulnerability Audit, Process Improvement, Stealing Thunder,
# Leaders Ready, Monitor Your Radar Screen
# 中文: 主动预防行动清单（Week 3 课件）
# ════════════════════════════════════════════════════════════════════════════

def get_proactive_checklist(scores: dict, rating: str) -> list[dict]:
    """
    Generate a prioritized proactive action checklist based on the
    scoring profile, following the AC820 Proactive Approach framework.

    Each item: {priority, action, rationale, timing}
    Priority: 🔴 Immediate | 🟠 This Quarter | 🟡 This Year | 🟢 Ongoing

    中文: 根据评分生成优先级行动清单，基于课件"主动策略"框架
    """
    lit  = scores.get("litigious_pct",    0)
    unc  = scores.get("uncertainty_pct",  0)
    neg  = scores.get("negative_pct",     0)
    con  = scores.get("constraining_pct", 0)
    weak = scores.get("weak_modal_pct",   0)
    pos  = scores.get("positive_pct",     0)

    checklist = []

    # ── 1. Steal Thunder (always relevant if any risk) ──────────────────────
    if rating in ("C", "D"):
        checklist.append({
            "priority": "🔴 Immediate",
            "action": "Steal Thunder — Get ahead of the story",
            "rationale": (
                "Companies that proactively disclose bad news before media coverage "
                "suffer 35% less reputational damage (Arpan & Roskos-Ewoldsen, 2005). "
                "Draft a proactive stakeholder statement now."
            ),
            "timing": "Within 48 hours",
            "framework": "Proactive Approach",
        })

    # ── 2. Vulnerability Audit ───────────────────────────────────────────────
    checklist.append({
        "priority": "🟠 This Quarter" if rating in ("B", "C") else "🟡 This Year",
        "action": "Conduct a Vulnerability Audit (Internal + External)",
        "rationale": (
            "Audit all organizational aspects: supply chain, global operations, "
            "stakeholder relationships, management conduct, and industry-specific risks. "
            "Ask: 'Where are we most vulnerable? Are we self-creating our own crises?'"
        ),
        "timing": "Before next board meeting",
        "framework": "Vulnerability Audit (AC820 Week 3)",
    })

    # ── 3. Litigation / Legal exposure ──────────────────────────────────────
    if lit > 2.0:
        checklist.append({
            "priority": "🔴 Immediate" if lit > 3.5 else "🟠 This Quarter",
            "action": "Legal & Compliance Disclosure Review",
            "rationale": (
                f"Litigious language score of {lit:.1f}/1000 words exceeds baseline. "
                "Engage legal counsel to review all pending and potential claims. "
                "Ensure disclosures are complete — incomplete disclosure is a "
                "compounding risk (Mismanagement crisis type)."
            ),
            "timing": "Before next 10-K/Q filing",
            "framework": "Lerbinger: Mismanagement/Misconduct type",
        })

    # ── 4. Uncertainty / Hedge language reduction ────────────────────────────
    if unc > 4.0 or weak > 2.0:
        checklist.append({
            "priority": "🟠 This Quarter",
            "action": "Disclosure Language Quality Review",
            "rationale": (
                f"Uncertainty score {unc:.1f}/1000 + Weak Modal {weak:.1f}/1000 are elevated. "
                "Audit next filing draft for hedge words ('may', 'might', 'could', 'possibly'). "
                "Replace with specific metrics, timelines, and commitments where possible. "
                "Management conviction language directly affects investor confidence."
            ),
            "timing": "Pre-filing editorial review",
            "framework": "LM Dictionary signal: Uncertainty + Weak Modal",
        })

    # ── 5. Crisis Management Team (CMT) Readiness ───────────────────────────
    if rating in ("C", "D"):
        checklist.append({
            "priority": "🔴 Immediate",
            "action": "Convene Crisis Management Team (CMT)",
            "rationale": (
                "CMT composition: CEO/COO, PR Lead, Legal, HR, Finance, "
                "and affected business unit lead. "
                "Assign: (1) Spokesperson, (2) Command Center, (3) Stakeholder list, "
                "(4) Communication cadence. "
                "The CMT is the 'nerve center' — activate before media forces your hand."
            ),
            "timing": "Today",
            "framework": "Crisis Management Team (AC820 Week 3)",
        })
    else:
        checklist.append({
            "priority": "🟡 This Year",
            "action": "Test & Update Crisis Management Team (CMT) Plan",
            "rationale": (
                "Review CMT membership, contact lists, and response plans annually. "
                "Conduct a tabletop simulation — 'Why the best plans fail: "
                "no one has practiced them.' (AC820 Week 4)"
            ),
            "timing": "Annual review",
            "framework": "CMT Planning (AC820 Week 3)",
        })

    # ── 6. External Scanning (Meltwater logic — replicated without API) ──────
    checklist.append({
        "priority": "🟢 Ongoing",
        "action": "Monitor Radar Screen — External Scanning",
        "rationale": (
            "Track: (1) Media/social early signals on this company, "
            "(2) Industry regulatory changes, "
            "(3) Competitor crisis patterns as leading indicators. "
            "External scanning logic: Policy → Economic → Social → Environmental forces "
            "are the most common crisis triggers for public companies."
        ),
        "timing": "Weekly",
        "framework": "External Scanning (AC820 Week 2 — Meltwater logic)",
    })

    # ── 7. Leaders Ready ─────────────────────────────────────────────────────
    checklist.append({
        "priority": "🟡 This Year",
        "action": "Leaders Ready — Board & Executive Crisis Briefing",
        "rationale": (
            "Board involvement is a crisis preparedness prerequisite. "
            "Ensure: (1) Executives willing to recognize risks (not optimism bias), "
            "(2) Board has reviewed crisis plan, "
            "(3) Designated spokesperson is trained and media-ready."
        ),
        "timing": "Next board meeting",
        "framework": "Leadership in Crisis Planning (AC820 Week 3)",
    })

    # ── 8. Process Improvement (if constraining signals) ────────────────────
    if con > 1.5:
        checklist.append({
            "priority": "🟠 This Quarter",
            "action": "Process Improvement — Pre-build Remediation",
            "rationale": (
                f"Constraining language score of {con:.1f}/1000 words indicates "
                "regulatory obligations or operational restrictions. "
                "Ensure remediation processes are already in motion before "
                "regulators or media escalate — remediation announced proactively "
                "is far more credible than remediation announced reactively."
            ),
            "timing": "Q-over-Q review",
            "framework": "Process Improvement (AC820 Week 3 Proactive Approach)",
        })

    # ── 9. Good Ongoing PR (always) ─────────────────────────────────────────
    checklist.append({
        "priority": "🟢 Ongoing",
        "action": "Good Ongoing PR — Build Reputational Capital",
        "rationale": (
            "'Good PR is the best crisis insurance.' (PR Page Principle). "
            "Maintain regular, transparent stakeholder communications even in "
            "non-crisis periods. Reputational capital built today reduces "
            "crisis damage tomorrow. "
            "Publish ESG updates, earnings narratives, and community engagement stories."
        ),
        "timing": "Quarterly",
        "framework": "PR Page Principle (AC820 Week 1)",
    })

    # Sort by priority
    priority_order = {"🔴 Immediate": 0, "🟠 This Quarter": 1,
                      "🟡 This Year": 2, "🟢 Ongoing": 3}
    checklist.sort(key=lambda x: priority_order.get(x["priority"], 9))

    return checklist


# ════════════════════════════════════════════════════════════════════════════
# ISSUES MANAGEMENT TRIAGE
# Source: Crisis Ready® Flowchart + AC820 Week 1 Issues Management Process
# Identify → Listen → Investigate → React → Respond → Communicate → Debrief
# Crisis Spectrum: Mild (BAU) → Issues Management → Crisis Management
# 中文: Issues Management 分级判断（Crisis Ready 决策树逻辑）
# ════════════════════════════════════════════════════════════════════════════

def triage_issue_severity(scores: dict, rating: str) -> dict:
    """
    Apply the Crisis Ready® triage logic to classify severity and
    recommend the appropriate response level.

    Spectrum: Business as Usual → Issues Management → Crisis Management
    Key triage questions (from Crisis Ready® Flowchart):
      - Is it emotionally charged / garnering negative attention?
      - Can we correct and regain control easily?
      - Does it risk relationships, reputation, or bottom line long-term?

    中文: 应用 Crisis Ready 决策树判断事件严重程度和应对级别
    """
    lit  = scores.get("litigious_pct",   0)
    unc  = scores.get("uncertainty_pct", 0)
    neg  = scores.get("negative_pct",    0)
    weak = scores.get("weak_modal_pct",  0)

    # Crisis Ready® triage questions mapped to LM signals
    # 中文: 把 LM 信号映射到 Crisis Ready 判断问题
    emotionally_charged    = neg > 3.5 or lit > 2.0          # negative/litigious = emotional signal
    can_regain_control     = weak < 2.0 and unc < 4.0        # low uncertainty = controllable
    long_term_relationship_risk = lit > 2.5 or unc > 5.0    # litigation/uncertainty = long-term risk

    if rating == "A" and not emotionally_charged:
        level    = "Business as Usual — Monitor"
        color    = "#2E7D32"
        response = (
            "Situation is within normal parameters. "
            "Monitor routine — no escalation required. "
            "Continue good ongoing PR and stakeholder engagement."
        )
        escalate = False

    elif rating == "B" or (emotionally_charged and can_regain_control):
        level    = "Issues Management — Respond Accordingly"
        color    = "#F9A825"
        response = (
            "This is an Issues Management situation — do NOT let it escalate. "
            "Publish a sincere, emotionally intelligent response. "
            "If apology is due, apologize clearly and focus on "
            "strengthening relationships. Monitor closely for escalation."
        )
        escalate = long_term_relationship_risk

    else:
        level    = "Crisis Management — Escalate to CMT"
        color    = "#B71C1C"
        response = (
            "Escalate to Crisis Management Team immediately. "
            "Multiple crisis signals are simultaneously elevated. "
            "Ensure an emotionally intelligent response is published "
            "with minimum delay. Focus on learning, corrective actions, "
            "and stakeholder relationship repair."
        )
        escalate = True

    return {
        "level":                     level,
        "color":                     color,
        "response_guidance":         response,
        "escalate_to_cmt":           escalate,
        "triage_emotionally_charged": emotionally_charged,
        "triage_controllable":        can_regain_control,
        "triage_long_term_risk":      long_term_relationship_risk,
    }
