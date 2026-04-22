import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

import edgar
import scorer
import models

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Corporate Disclosure Crisis Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
def init_state():
    defaults = {
        "word_sets": None,
        "crisis_model": None,
        "results": {},
        "peer_results": [],
        "setup_complete": False,
        "active_step": "Setup",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()

# ------------------------------------------------------------
# Design tokens
# ------------------------------------------------------------
PALETTE = {
    "danger": "#B42318",
    "warning": "#F79009",
    "success": "#12B76A",
    "info": "#1570EF",
    "text": "#101828",
    "muted": "#667085",
    "border": "#EAECF0",
    "surface": "#FFFFFF",
    "bg": "#F8FAFC",
}

RATING_COLORS = {
    "A": "#12B76A",
    "B": "#1570EF",
    "C": "#F79009",
    "D": "#B42318",
}


# ------------------------------------------------------------
# UI helpers
# ------------------------------------------------------------
def inject_css():
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: {PALETTE['bg']};
            }}
            .block-container {{
                padding-top: 1.2rem;
                padding-bottom: 2rem;
            }}
            .hero-card, .app-card, .kpi-card, .status-card {{
                background: white;
                border: 1px solid {PALETTE['border']};
                border-radius: 18px;
                padding: 18px 20px;
                box-shadow: 0 1px 2px rgba(16,24,40,0.04);
            }}
            .section-label {{
                color: {PALETTE['muted']};
                font-size: 0.84rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: .04em;
                margin-bottom: 6px;
            }}
            .big-title {{
                color: {PALETTE['text']};
                font-size: 2rem;
                font-weight: 800;
                line-height: 1.1;
                margin-bottom: .35rem;
            }}
            .subtle {{
                color: {PALETTE['muted']};
                font-size: .96rem;
            }}
            .pill {{
                display: inline-block;
                padding: .2rem .55rem;
                border-radius: 999px;
                font-size: .78rem;
                font-weight: 700;
                border: 1px solid {PALETTE['border']};
                background: #fff;
                color: {PALETTE['text']};
                margin-right: 6px;
                margin-bottom: 6px;
            }}
            .step-pill {{
                display: inline-block;
                padding: .24rem .55rem;
                border-radius: 999px;
                font-size: .78rem;
                font-weight: 700;
                margin-right: 6px;
                background: #EEF4FF;
                color: {PALETTE['info']};
            }}
            .summary-banner {{
                border-radius: 18px;
                padding: 18px 20px;
                color: white;
                margin-bottom: 14px;
            }}
            .metric-label {{
                color: {PALETTE['muted']};
                font-size: .82rem;
                font-weight: 600;
                margin-bottom: 4px;
            }}
            .metric-value {{
                color: {PALETTE['text']};
                font-size: 1.7rem;
                font-weight: 800;
                line-height: 1.1;
            }}
            .metric-sub {{
                color: {PALETTE['muted']};
                font-size: .82rem;
                margin-top: 6px;
            }}
            .small-list li {{
                margin-bottom: .35rem;
            }}
            div[data-testid="stMetric"] {{
                background: white;
                border: 1px solid {PALETTE['border']};
                border-radius: 16px;
                padding: 12px 14px;
            }}
            div[data-testid="stDataFrame"] {{
                border-radius: 14px;
                overflow: hidden;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero():
    st.markdown(
        """
        <div class="hero-card">
            <div class="section-label">Corporate disclosure monitoring</div>
            <div class="big-title">Corporate Disclosure Crisis Radar</div>
            <div class="subtle">
                Detect disclosure-driven crisis signals in SEC filings, compare them with media tone,
                and turn the output into a usable communication response plan.
            </div>
            <div style="margin-top:12px;">
                <span class="pill">SEC EDGAR</span>
                <span class="pill">Yahoo Finance News</span>
                <span class="pill">LM Dictionary</span>
                <span class="pill">SCCT Guidance</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def step_status(name: str, done: bool, active: bool) -> str:
    if done:
        icon = "✅"
    elif active:
        icon = "🔵"
    else:
        icon = "⬜"
    return f"{icon} {name}"


def sidebar_navigation():
    with st.sidebar:
        st.markdown("## 📡 Crisis Radar")
        st.caption("Cleaner decision workflow")
        st.divider()

        has_setup = st.session_state.word_sets is not None
        has_model = st.session_state.crisis_model is not None
        has_results = bool(st.session_state.results)

        st.caption("**Workflow**")
        st.markdown(step_status("1. Setup", has_setup, st.session_state.active_step == "Setup"))
        st.markdown(step_status("2. Analyze", has_results is False and st.session_state.active_step == "Analyze", st.session_state.active_step == "Analyze"))
        st.markdown(step_status("3. Review", has_results, st.session_state.active_step == "Review"))
        st.markdown(step_status("4. Action Plan", has_results, st.session_state.active_step == "Action Plan"))
        st.divider()

        page = st.radio(
            "Go to",
            ["Overview", "Setup", "Analyze", "Review", "Action Plan", "Methods"],
            index=["Overview", "Setup", "Analyze", "Review", "Action Plan", "Methods"].index(
                st.session_state.active_step if st.session_state.active_step in ["Setup", "Analyze"] else "Overview"
            ) if st.session_state.active_step in ["Setup", "Analyze"] else 0,
        )

        st.divider()
        st.caption("**System status**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Dictionary", "Ready" if has_setup else "Missing")
        c2.metric("Model", "Ready" if has_model else "Optional")
        c3.metric("Results", "Yes" if has_results else "No")

        return page


def render_summary_banner(title: str, text: str, color: str):
    st.markdown(
        f"""
        <div class="summary-banner" style="background:{color};">
            <div style="font-size:1.05rem;font-weight:800;margin-bottom:4px;">{title}</div>
            <div style="font-size:.95rem;opacity:.95;">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, subtext: str = ""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_get_rating_color(rating: str, default: str = PALETTE["info"]):
    return RATING_COLORS.get(rating, default)


# ------------------------------------------------------------
# Data / logic helpers
# ------------------------------------------------------------
def ensure_dictionary_loaded(path: str = "data/LM_MasterDictionary.csv"):
    if st.session_state.word_sets is None:
        try:
            st.session_state.word_sets = scorer.load_lm_dictionary(path)
        except Exception:
            st.session_state.word_sets = scorer.load_lm_dictionary()


def analyze_media_headlines(ticker: str, word_sets):
    pr_div = None
    mt_analysis = None
    try:
        ticker_obj = yf.Ticker(ticker)
        raw_news = ticker_obj.news
        if not raw_news:
            return None, None

        news_rows = []
        for item in raw_news:
            content = item.get("content", item)
            title = (content.get("title") or item.get("title") or "")
            publisher = content.get("provider", {}).get("displayName") or item.get("publisher") or "Unknown"
            ts = content.get("pubDate") or item.get("providerPublishTime") or 0
            if isinstance(ts, (int, float)) and ts > 1e9:
                pub_dt = pd.Timestamp(ts, unit="s")
            else:
                pub_dt = pd.to_datetime(ts, errors="coerce")
            if title:
                news_rows.append({"title": title, "publisher": publisher, "date": pub_dt})

        news_df = pd.DataFrame(news_rows)
        if news_df.empty:
            return None, None

        all_headlines = " ".join(news_df["title"].fillna("").tolist())
        media_scores = scorer.score_text(all_headlines, word_sets)

        def classify_headline(text: str) -> str:
            s = scorer.score_text(text, word_sets)
            net = s.get("net_sentiment", 0)
            if net > 0.01:
                return "Positive"
            if net < -0.01:
                return "Negative"
            return "Neutral"

        news_df["sentiment"] = news_df["title"].apply(classify_headline)
        counts = news_df["sentiment"].value_counts()
        total = len(news_df)
        pos = int(counts.get("Positive", 0))
        neg = int(counts.get("Negative", 0))
        neu = int(counts.get("Neutral", 0))
        mt_neg_pct = neg / total * 100 if total else 0

        mt_analysis = {
            "total_articles": total,
            "positive_count": pos,
            "neutral_count": neu,
            "negative_count": neg,
            "mt_neg_pct": round(mt_neg_pct, 1),
            "top_sources": news_df["publisher"].value_counts().head(5),
            "window_df": news_df,
            "media_scores": media_scores,
            "source_label": "Yahoo Finance News (recent headlines)",
        }
        return mt_analysis, None
    except Exception:
        return None, None


def run_analysis(ticker: str, form_code: str):
    ensure_dictionary_loaded()
    word_sets = st.session_state.word_sets

    progress = st.progress(0, text="Fetching company identifier...")

    cik = edgar.get_cik(ticker)
    if not cik:
        progress.empty()
        raise ValueError(f"Could not find CIK for {ticker}.")

    progress.progress(20, text="Loading filing list...")
    filings = edgar.get_filings(cik, form_code)
    if filings.empty:
        progress.empty()
        raise ValueError(f"No {form_code} filings found for {ticker}.")

    latest = filings.iloc[0]
    progress.progress(45, text="Downloading filing text...")
    filing_text = edgar.get_filing_text(cik, latest["accession"])

    progress.progress(65, text="Scoring disclosure language...")
    scores = scorer.score_text(filing_text, word_sets)
    rating, color, rating_label = scorer.assign_rating(scores["crisis_score"])

    progress.progress(78, text="Running optional model and media checks...")
    crisis_prob = None
    if st.session_state.crisis_model:
        crisis_prob = st.session_state.crisis_model.predict_from_scores(scores)

    mt_analysis = None
    pr_div = None
    try:
        mt_analysis, _ = analyze_media_headlines(ticker, word_sets)
        if mt_analysis and mt_analysis.get("media_scores"):
            filing_crisis = scores.get("crisis_score", 0)
            media_crisis = mt_analysis["media_scores"].get("crisis_score", 0)
            pr_div = round(abs(filing_crisis - media_crisis), 2)
            mt_analysis["filing_crisis"] = filing_crisis
            mt_analysis["media_crisis"] = media_crisis
    except Exception:
        mt_analysis = None
        pr_div = None

    progress.progress(86, text="Generating communication guidance...")
    guidance = scorer.get_scct_guidance(scores, pr_div)
    trigger_sentences = scorer.extract_top_trigger_sentences(filing_text, word_sets, n=5)
    short_seller_signal = {"detected": False, "firms_mentioned": [], "headlines": [], "severity": "🟢 None"}
    if mt_analysis and "window_df" in mt_analysis:
        short_seller_signal = scorer.detect_short_seller_signal(mt_analysis["window_df"])

    lerbinger_type = scorer.classify_lerbinger_type(scores)
    lifecycle_stage = scorer.get_lifecycle_stage(scores, rating)
    triage = scorer.triage_issue_severity(scores, rating)
    checklist = scorer.get_proactive_checklist(scores, rating)

    progress.progress(92, text="Comparing with recent 8-K filings...")
    eightk_filings = edgar.get_8k_filings(cik, count=5)
    eightk_scores = None
    eightk_div = None
    if not eightk_filings.empty:
        try:
            ek_texts = []
            for _, ek_row in eightk_filings.head(3).iterrows():
                ek_text = edgar.get_filing_text(cik, ek_row["accession"], max_chars=30000)
                if ek_text:
                    ek_texts.append(ek_text)
            if ek_texts:
                combined_ek = " ".join(ek_texts)
                eightk_scores = scorer.score_text(combined_ek, word_sets)
                eightk_div = abs(scores.get("negative_pct", 0) - eightk_scores.get("negative_pct", 0)) + abs(
                    scores.get("uncertainty_pct", 0) - eightk_scores.get("uncertainty_pct", 0)
                )
        except Exception:
            eightk_scores = None
            eightk_div = None

    progress.progress(100, text="Analysis complete.")
    progress.empty()

    st.session_state.results = {
        "ticker": ticker,
        "form_code": form_code,
        "filed_date": latest["filed_date"],
        "cik": cik,
        "scores": scores,
        "rating": rating,
        "color": color,
        "rating_label": rating_label,
        "crisis_prob": crisis_prob,
        "pr_div": pr_div,
        "guidance": guidance,
        "filings": filings,
        "lerbinger_type": lerbinger_type,
        "lifecycle_stage": lifecycle_stage,
        "triage": triage,
        "checklist": checklist,
        "eightk_filings": eightk_filings,
        "eightk_scores": eightk_scores,
        "eightk_div": eightk_div,
        "mt_analysis": mt_analysis,
        "trigger_sentences": trigger_sentences,
        "short_seller_signal": short_seller_signal,
        "filing_text_snippet": filing_text[:3000],
    }
    st.session_state.active_step = "Review"


def create_radar(scores: dict, color: str):
    dims_k = ["uncertainty", "litigious", "negative", "weak_modal", "constraining"]
    dims_l = ["Uncertainty", "Litigious", "Negative", "Hedging", "Constraint"]
    vals = [scores.get(f"{d}_pct", 0) for d in dims_k]

    def hex_to_rgba(hex_color: str, alpha: float = 0.18) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    fig = go.Figure(
        go.Scatterpolar(
            r=vals + [vals[0]],
            theta=dims_l + [dims_l[0]],
            fill="toself",
            fillcolor=hex_to_rgba(color),
            line=dict(color=color, width=2.5),
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(vals) * 1.35 or 1])),
        showlegend=False,
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def create_dimension_bar(scores: dict, color: str):
    dim_map = {
        "uncertainty": "Uncertainty",
        "litigious": "Litigious",
        "negative": "Negative",
        "weak_modal": "Hedging",
        "constraining": "Constraint",
        "positive": "Positive",
        "strong_modal": "Confidence",
    }
    vals = [scores.get(f"{k}_pct", 0) for k in dim_map]
    labels = list(dim_map.values())
    colors = [PALETTE["success"] if k in ("positive", "strong_modal") else color for k in dim_map]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.2f}" for v in vals],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=330,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        xaxis_title="Words per 1,000",
        margin=dict(l=20, r=50, t=20, b=30),
    )
    return fig


def create_media_comparison(scores: dict, media_scores: dict, color: str):
    dims = ["negative", "uncertainty", "litigious", "weak_modal", "constraining"]
    labels = ["Negative", "Uncertainty", "Litigious", "Hedging", "Constraint"]
    filing_v = [scores.get(f"{d}_pct", 0) for d in dims]
    media_v = [media_scores.get(f"{d}_pct", 0) for d in dims]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="SEC Filing", x=labels, y=filing_v, marker_color=color, opacity=0.88))
    fig.add_trace(go.Bar(name="Media Headlines", x=labels, y=media_v, marker_color=PALETTE["info"], opacity=0.88))
    fig.update_layout(
        height=320,
        barmode="group",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        yaxis_title="Words per 1,000",
        margin=dict(l=20, r=20, t=20, b=40),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def create_event_study(ticker: str, filed_date: str, color: str, rating: str, form_code: str):
    import numpy as np
    from sklearn.linear_model import LinearRegression

    filing_dt = datetime.strptime(filed_date, "%Y-%m-%d")
    dl_start = (filing_dt - timedelta(days=400)).strftime("%Y-%m-%d")
    dl_end = (filing_dt + timedelta(days=90)).strftime("%Y-%m-%d")

    def clean_download(sym: str):
        df = yf.download(sym, start=dl_start, end=dl_end, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"])
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        return df[["Date", "Close"]].dropna().set_index("Date")

    stock_df = clean_download(ticker)
    market_df = clean_download("SPY")
    merged = stock_df.join(market_df, how="inner", lsuffix="_s", rsuffix="_m")
    merged.columns = ["stock", "market"]
    merged["r_stock"] = merged["stock"].pct_change()
    merged["r_market"] = merged["market"].pct_change()
    merged = merged.dropna()

    filing_idx = merged.index.searchsorted(filing_dt)
    est_start = max(0, filing_idx - 250)
    est_end = max(0, filing_idx - 60)
    evt_start = max(0, filing_idx - 60)
    evt_end = min(len(merged), filing_idx + 61)

    est_df = merged.iloc[est_start:est_end]
    evt_df = merged.iloc[evt_start:evt_end].copy()
    if len(est_df) < 30 or len(evt_df) < 5:
        raise ValueError("Insufficient trading days for event study.")

    X_est = est_df[["r_market"]].values
    y_est = est_df["r_stock"].values
    reg = LinearRegression().fit(X_est, y_est)
    alpha_hat = reg.intercept_
    beta_hat = reg.coef_[0]

    evt_df["expected_r"] = alpha_hat + beta_hat * evt_df["r_market"]
    evt_df["AR"] = evt_df["r_stock"] - evt_df["expected_r"]
    evt_df["CAR"] = evt_df["AR"].cumsum() * 100
    evt_df_reset = evt_df.reset_index()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=evt_df_reset["Date"],
            y=evt_df["CAR"].values,
            mode="lines",
            line=dict(color=PALETTE["info"], width=2.5),
            name="CAR (%)",
            hovertemplate="%{x|%b %d, %Y}<br>CAR: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_shape(
        type="line",
        x0=evt_df_reset["Date"].iloc[0],
        x1=evt_df_reset["Date"].iloc[-1],
        y0=0,
        y1=0,
        yref="y",
        line=dict(color="#98A2B3", width=1, dash="dot"),
    )
    fig.add_shape(
        type="line",
        x0=filed_date,
        x1=filed_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color=color, width=2, dash="dash"),
    )
    fig.add_annotation(
        x=filed_date,
        y=1.03,
        yref="paper",
        text=f"{form_code} filed [{rating}]",
        showarrow=False,
        font=dict(color=color, size=10),
        xanchor="left",
    )
    fig.update_layout(
        height=350,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        xaxis_title="Date",
        yaxis_title="Cumulative abnormal return (%)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=20, b=40),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig, evt_df, beta_hat


def create_peer_comparison(peers: list):
    fig = go.Figure()
    for dim, label, clr in [
        ("crisis_score", "Crisis Score", PALETTE["danger"]),
        ("uncertainty", "Uncertainty", PALETTE["warning"]),
        ("litigious", "Litigious", "#F04438"),
        ("negative", "Negative", PALETTE["info"]),
    ]:
        fig.add_trace(
            go.Bar(
                name=label,
                x=[p["ticker"] for p in peers],
                y=[p[dim] for p in peers],
                marker_color=clr,
                opacity=0.86,
            )
        )
    fig.update_layout(
        barmode="group",
        height=360,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        yaxis_title="Score",
        legend=dict(orientation="h", y=-0.22),
        margin=dict(l=20, r=20, t=20, b=50),
    )
    return fig


def build_executive_takeaway(results: dict) -> str:
    scores = results["scores"]
    mt = results.get("mt_analysis")
    rating = results["rating"]
    label = results["rating_label"]

    driver_pairs = [
        ("uncertainty_pct", "uncertainty"),
        ("negative_pct", "negative tone"),
        ("litigious_pct", "legal-risk"),
        ("weak_modal_pct", "hedging"),
        ("constraining_pct", "constraint"),
    ]
    drivers = sorted(driver_pairs, key=lambda x: scores.get(x[0], 0), reverse=True)[:2]
    driver_text = " and ".join([d[1] for d in drivers])

    if mt and results.get("pr_div") is not None:
        if mt.get("media_crisis", 0) > mt.get("filing_crisis", 0) + 0.5:
            media_line = "Media tone is running more negative than the filing, which suggests a public narrative gap."
        elif mt.get("filing_crisis", 0) > mt.get("media_crisis", 0) + 0.5:
            media_line = "The filing itself appears more risk-laden than recent media coverage."
        else:
            media_line = "Media tone is broadly aligned with the disclosure language."
    else:
        media_line = "Media comparison was not available for this run."

    return f"This company is currently rated {rating} ({label}), with the score driven mainly by {driver_text}. {media_line}"


# ------------------------------------------------------------
# Pages
# ------------------------------------------------------------
def page_overview():
    st.session_state.active_step = "Overview"
    hero()
    st.write("")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="app-card">
                <div class="section-label">What the app does</div>
                <div style="font-weight:800;font-size:1.1rem;margin-bottom:8px;">Find disclosure risk early</div>
                <ul class="small-list">
                    <li>Scores SEC language across multiple crisis dimensions</li>
                    <li>Compares disclosure tone against recent media headlines</li>
                    <li>Translates outputs into communication guidance</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="app-card">
                <div class="section-label">Suggested workflow</div>
                <div style="font-weight:800;font-size:1.1rem;margin-bottom:8px;">Use it in four steps</div>
                <ul class="small-list">
                    <li>Load the LM dictionary and optional model</li>
                    <li>Select a company and filing type</li>
                    <li>Review score drivers and market context</li>
                    <li>Turn findings into an action plan</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="app-card">
                <div class="section-label">Data sources</div>
                <div style="font-weight:800;font-size:1.1rem;margin-bottom:8px;">Built around public inputs</div>
                <ul class="small-list">
                    <li>SEC EDGAR: 10-K, 10-Q, and 8-K filings</li>
                    <li>Yahoo Finance: price data and recent headlines</li>
                    <li>WRDS/Compustat: optional model training workflow</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    if st.button("Start setup →", type="primary"):
        st.session_state.active_step = "Setup"
        st.rerun()


def page_setup():
    st.session_state.active_step = "Setup"
    st.markdown("## Step 1 · Setup")
    st.caption("Load the dictionary first. The predictive model is optional.")

    col_a, col_b = st.columns([1.2, 0.8])
    with col_a:
        st.markdown("### Required")
        lm_path = st.text_input("Dictionary path", value="data/LM_MasterDictionary.csv")
        if st.button("Load LM dictionary", type="primary"):
            try:
                with st.spinner("Loading dictionary..."):
                    ws = scorer.load_lm_dictionary(lm_path)
                    st.session_state.word_sets = ws
                total = sum(len(v) for v in ws.values())
                st.success(f"Loaded {total:,} entries across {len(ws)} dimensions.")
            except Exception as e:
                st.error(f"Could not load dictionary: {e}")

        with st.expander("Advanced / optional model setup"):
            st.write("Train the logistic model only if you want the extra crisis-probability estimate.")
            wrds_user = st.text_input("WRDS username", placeholder="your_wrds_username")
            if st.button("Connect WRDS and train model"):
                try:
                    with st.spinner("Connecting and training model..."):
                        model = models.build_and_train_model(wrds_user if wrds_user else None)
                        st.session_state.crisis_model = model
                    auc_str = f"{model.cv_auc[0]:.3f} ± {model.cv_auc[1]:.3f}" if model.cv_auc else "N/A"
                    st.success(f"Model trained. 5-fold CV AUC: {auc_str}")
                except Exception as e:
                    st.error(f"Model training failed: {e}")

    with col_b:
        st.markdown("### Status")
        render_metric_card(
            "LM Dictionary",
            "Ready" if st.session_state.word_sets is not None else "Missing",
            "Required for filing and headline scoring",
        )
        st.write("")
        render_metric_card(
            "Predictive Model",
            "Ready" if st.session_state.crisis_model is not None else "Optional",
            "Used for the estimated crisis probability",
        )
        st.write("")
        render_metric_card(
            "Media News",
            "Auto",
            "Yahoo Finance headlines are fetched during analysis",
        )

    st.write("")
    render_summary_banner(
        "Next step",
        "Once the LM dictionary is ready, go to Analyze to run the company workflow.",
        PALETTE["info"],
    )
    if st.button("Continue to analyze →"):
        st.session_state.active_step = "Analyze"
        st.rerun()


def page_analyze():
    st.session_state.active_step = "Analyze"
    st.markdown("## Step 2 · Analyze a company")
    st.caption("Pick one company and filing type to generate the core risk profile.")

    ensure_dictionary_loaded()

    with st.spinner("Loading S&P 500 list..."):
        sp500 = edgar.get_sp500_tickers()

    options = sp500.apply(lambda r: f"{r['ticker']} — {r['company']}", axis=1).tolist()

    left, right = st.columns([1.3, 0.7])
    with left:
        selected = st.selectbox("Company", options)
        ticker = selected.split(" — ")[0]
        form_type = st.radio("Filing type", ["10-K (Annual)", "10-Q (Quarterly)"], horizontal=True)
        form_code = "10-K" if "10-K" in form_type else "10-Q"

        run = st.button("Run analysis", type="primary")
        if run:
            try:
                run_analysis(ticker, form_code)
                st.success(f"Analysis complete for {ticker} {form_code}.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    with right:
        st.markdown(
            """
            <div class="app-card">
                <div class="section-label">What happens</div>
                <div style="font-weight:800;font-size:1.05rem;margin-bottom:8px;">This run includes</div>
                <ul class="small-list">
                    <li>Latest filing retrieval from EDGAR</li>
                    <li>LM dictionary scoring across risk dimensions</li>
                    <li>Optional crisis probability estimate</li>
                    <li>News headline comparison and communication guidance</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    with st.expander("Optional: quick peer comparison"):
        st.caption("Benchmark the main company against up to three peers.")
        peer_tickers = st.text_input(
            "Peer tickers",
            placeholder="MSFT, GOOGL, META",
            help="Add up to three peer companies",
        )
        if st.button("Run peer comparison"):
            if not peer_tickers.strip():
                st.warning("Enter at least one peer ticker.")
            else:
                peers = [t.strip().upper() for t in peer_tickers.split(",") if t.strip()][:3]
                peer_results = []
                if st.session_state.results:
                    main_r = st.session_state.results
                    peer_results.append(
                        {
                            "ticker": main_r["ticker"],
                            "filing": main_r["form_code"],
                            "filed_date": main_r["filed_date"],
                            "crisis_score": main_r["scores"].get("crisis_score", 0),
                            "rating": main_r["rating"],
                            "uncertainty": main_r["scores"].get("uncertainty_pct", 0),
                            "litigious": main_r["scores"].get("litigious_pct", 0),
                            "negative": main_r["scores"].get("negative_pct", 0),
                        }
                    )
                ws = st.session_state.word_sets
                for peer in peers:
                    try:
                        p_cik = edgar.get_cik(peer)
                        if not p_cik:
                            continue
                        p_filings = edgar.get_filings(p_cik, "10-K", count=1)
                        if p_filings.empty:
                            continue
                        p_latest = p_filings.iloc[0]
                        p_text = edgar.get_filing_text(p_cik, p_latest["accession"], max_chars=60000)
                        p_scores = scorer.score_text(p_text, ws)
                        p_rating, _, _ = scorer.assign_rating(p_scores["crisis_score"])
                        peer_results.append(
                            {
                                "ticker": peer,
                                "filing": "10-K",
                                "filed_date": p_latest["filed_date"],
                                "crisis_score": p_scores.get("crisis_score", 0),
                                "rating": p_rating,
                                "uncertainty": p_scores.get("uncertainty_pct", 0),
                                "litigious": p_scores.get("litigious_pct", 0),
                                "negative": p_scores.get("negative_pct", 0),
                            }
                        )
                    except Exception:
                        pass
                st.session_state.peer_results = peer_results
                if peer_results:
                    peer_df = pd.DataFrame(peer_results).sort_values("crisis_score", ascending=False)
                    st.dataframe(peer_df, use_container_width=True)
                    st.plotly_chart(create_peer_comparison(peer_results), use_container_width=True)


def page_review():
    st.session_state.active_step = "Review"
    results = st.session_state.results
    if not results:
        st.warning("No analysis results yet. Run an analysis first.")
        return

    ticker = results["ticker"]
    form_code = results["form_code"]
    filed_date = results["filed_date"]
    scores = results["scores"]
    rating = results["rating"]
    rating_color = safe_get_rating_color(rating, results.get("color", PALETTE["info"]))
    guidance = results["guidance"]
    mt_analysis = results.get("mt_analysis")
    crisis_prob = results.get("crisis_prob")
    pr_div = results.get("pr_div")

    st.markdown("## Step 3 · Review results")
    render_summary_banner(
        f"{ticker} · {form_code} filed on {filed_date}",
        build_executive_takeaway(results),
        rating_color,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Risk Rating", f"{rating}", results["rating_label"])
    with c2:
        render_metric_card("Net Sentiment", f"{scores.get('net_sentiment', 0):+.4f}", "Positive is better; negative suggests caution")
    with c3:
        render_metric_card(
            "Estimated Crisis Probability",
            f"{crisis_prob:.1%}" if crisis_prob is not None else "N/A",
            "Optional model-based estimate",
        )
    with c4:
        render_metric_card(
            "Negative Media Headlines",
            f"{mt_analysis['mt_neg_pct']:.1f}%" if mt_analysis else "N/A",
            f"Based on {mt_analysis['total_articles']} headlines" if mt_analysis else "Media comparison unavailable",
        )

    st.write("")
    st.markdown("### Executive summary")
    s1, s2, s3 = st.columns(3)
    filing_crisis = scores.get("crisis_score", 0)
    eightk_crisis = results.get("eightk_scores", {}).get("crisis_score", 0) if results.get("eightk_scores") else None
    media_crisis = mt_analysis["media_scores"].get("crisis_score", 0) if mt_analysis and mt_analysis.get("media_scores") else None

    with s1:
        st.metric("SEC filing score", f"{filing_crisis:.2f}", help="Overall crisis exposure score from the filing")
    with s2:
        st.metric("8-K score", f"{eightk_crisis:.2f}" if eightk_crisis is not None else "N/A", help="Tone from recent 8-K disclosures")
    with s3:
        st.metric("Media score", f"{media_crisis:.2f}" if media_crisis is not None else "N/A", help="Tone from recent headlines")

    if mt_analysis and pr_div is not None:
        if pr_div > 10:
            st.warning(f"Public narrative gap detected: media and filing tone diverge meaningfully (score gap {pr_div:.1f}).")
        else:
            st.success(f"Media tone is broadly consistent with filing language (gap {pr_div:.1f}).")

    ss = results.get("short_seller_signal", {})
    if ss.get("detected"):
        st.error(
            f"Short seller signal detected: {ss.get('severity', '')}. Mentions: {', '.join(ss.get('firms_mentioned', []))}"
        )

    # Tabs for cleaner information architecture
    tab1, tab2, tab3, tab4 = st.tabs(["Drivers", "Charts", "Market & Media", "Details"])

    with tab1:
        left, right = st.columns([1.1, 0.9])
        with left:
            st.markdown("#### What is driving this score?")
            top_dims = sorted(
                [
                    ("Uncertainty", scores.get("uncertainty_pct", 0)),
                    ("Litigious", scores.get("litigious_pct", 0)),
                    ("Negative", scores.get("negative_pct", 0)),
                    ("Hedging", scores.get("weak_modal_pct", 0)),
                    ("Constraint", scores.get("constraining_pct", 0)),
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            dim_df = pd.DataFrame(top_dims, columns=["Dimension", "Score per 1,000 words"])
            st.dataframe(dim_df, use_container_width=True, hide_index=True)
            st.plotly_chart(create_dimension_bar(scores, rating_color), use_container_width=True)

        with right:
            st.markdown("#### Trigger sentences")
            triggers = results.get("trigger_sentences", [])
            if triggers:
                for i, item in enumerate(triggers, 1):
                    dims_str = ", ".join(item["dimensions"])
                    flagged = ", ".join(item["flagged_words"])
                    st.markdown(
                        f"**{i}.** {item['sentence']}\n\n"
                        f"Score: `{item['crisis_score']:.3f}` · Dimensions: `{dims_str}` · Flag words: `{flagged}`"
                    )
                    st.divider()
            else:
                st.info("Trigger sentence extraction was not available.")

    with tab2:
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("#### Dimension radar")
            st.caption("A quick shape view of which language dimensions stand out most.")
            st.plotly_chart(create_radar(scores, rating_color), use_container_width=True)

        with c_right:
            st.markdown("#### Event study")
            st.caption("Cumulative abnormal return relative to SPY around the filing date.")
            try:
                fig_evt, evt_df, beta_hat = create_event_study(ticker, filed_date, rating_color, rating, form_code)
                st.plotly_chart(fig_evt, use_container_width=True)
                c_a, c_b, c_c = st.columns(3)
                pre = evt_df.loc[evt_df.index < pd.to_datetime(filed_date), "CAR"]
                post = evt_df.loc[evt_df.index >= pd.to_datetime(filed_date), "CAR"]
                c_a.metric("Pre-filing CAR", f"{pre.iloc[-1]:+.2f}%" if len(pre) else "N/A")
                c_b.metric("Post-filing CAR", f"{post.iloc[-1]:+.2f}%" if len(post) else "N/A")
                c_c.metric("Market beta", f"{beta_hat:.3f}")
            except Exception as e:
                st.info(f"Event study unavailable: {e}")

    with tab3:
        if mt_analysis and mt_analysis.get("media_scores"):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Articles", mt_analysis["total_articles"])
            m2.metric("Positive", mt_analysis["positive_count"])
            m3.metric("Neutral", mt_analysis["neutral_count"])
            m4.metric("Negative", mt_analysis["negative_count"])

            left, right = st.columns(2)
            with left:
                pie = go.Figure(
                    go.Pie(
                        labels=["Positive", "Neutral", "Negative"],
                        values=[
                            mt_analysis["positive_count"],
                            mt_analysis["neutral_count"],
                            mt_analysis["negative_count"],
                        ],
                        marker_colors=[PALETTE["success"], "#98A2B3", PALETTE["danger"]],
                        hole=0.55,
                        textinfo="label+percent",
                    )
                )
                pie.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
                st.plotly_chart(pie, use_container_width=True)
            with right:
                st.plotly_chart(
                    create_media_comparison(scores, mt_analysis["media_scores"], rating_color),
                    use_container_width=True,
                )

            with st.expander("Recent headlines"):
                display_cols = [c for c in ["date", "title", "publisher", "sentiment"] if c in mt_analysis["window_df"].columns]
                st.dataframe(mt_analysis["window_df"][display_cols], use_container_width=True, hide_index=True)

            with st.expander("Top media sources"):
                src_df = mt_analysis["top_sources"].reset_index()
                src_df.columns = ["Source", "Article count"]
                st.dataframe(src_df, use_container_width=True, hide_index=True)
        else:
            st.info("Media comparison was not available for this analysis.")

        if results.get("eightk_scores") is not None:
            st.markdown("#### Filing vs. recent 8-K tone")
            ek1, ek2, ek3 = st.columns(3)
            ek1.metric("10-K / 10-Q negative", f"{scores.get('negative_pct', 0):.2f}")
            ek2.metric("8-K negative", f"{results['eightk_scores'].get('negative_pct', 0):.2f}")
            ek3.metric("Narrative gap", f"{results.get('eightk_div', 0):.2f}")

    with tab4:
        with st.expander("Historical filings"):
            st.dataframe(results["filings"], use_container_width=True)
        with st.expander("Raw scores"):
            score_df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in scores.items()])
            st.dataframe(score_df, use_container_width=True, hide_index=True)
        if st.session_state.peer_results:
            with st.expander("Saved peer comparison"):
                st.dataframe(pd.DataFrame(st.session_state.peer_results), use_container_width=True, hide_index=True)

    st.write("")
    if st.button("Continue to action plan →"):
        st.session_state.active_step = "Action Plan"
        st.rerun()


def page_action_plan():
    st.session_state.active_step = "Action Plan"
    results = st.session_state.results
    if not results:
        st.warning("No analysis results yet. Run an analysis first.")
        return

    rating = results["rating"]
    rating_color = safe_get_rating_color(rating, results.get("color", PALETTE["info"]))
    guidance = results["guidance"]
    lifecycle = results.get("lifecycle_stage", {})
    triage = results.get("triage", {})
    lerbinger = results.get("lerbinger_type", {})
    checklist = results.get("checklist", [])

    st.markdown("## Step 4 · Action plan")
    render_summary_banner(
        "Recommended response posture",
        f"{guidance.get('strategy', 'N/A')} · {guidance.get('action', '')}",
        rating_color,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("SCCT cluster", guidance.get("crisis_type", "N/A"), "Recommended communication stance")
    with c2:
        render_metric_card("Lifecycle stage", lifecycle.get("stage", "Unknown"), lifecycle.get("urgency", ""))
    with c3:
        render_metric_card("Issues triage", triage.get("level", "N/A"), "Escalation recommendation")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("### Communication guidance")
        st.info(guidance.get("action", "No action guidance available."))
        st.markdown("**Advocacy ↔ Accommodation**")
        st.write(guidance.get("advocacy_accommodation", "N/A"))
        st.markdown("**LP / Investor trust signal**")
        st.write(guidance.get("lp_signal", "N/A"))

        if lerbinger:
            st.markdown("### Crisis type")
            st.success(f"{lerbinger.get('icon', '⚠️')} {lerbinger.get('type_name', 'Unknown')}")
            st.write(lerbinger.get("description", ""))
            if lerbinger.get("stealing_thunder_tip"):
                st.warning(f"Stealing Thunder: {lerbinger['stealing_thunder_tip']}")

    with right:
        st.markdown("### Immediate checklist")
        if checklist:
            for item in checklist:
                st.markdown(f"- **{item['priority']}** — {item['action']}  ")
                st.caption(f"{item['rationale']}")
        else:
            st.info("No checklist was generated.")

        if triage.get("response_guidance"):
            st.markdown("### Triage guidance")
            st.write(triage["response_guidance"])

    with st.expander("Framework details"):
        st.markdown("**Lifecycle stage description**")
        st.write(lifecycle.get("description", "N/A"))
        if lerbinger.get("exacerbating_factors"):
            st.markdown("**Exacerbating factors**")
            for factor in lerbinger["exacerbating_factors"]:
                st.markdown(f"- {factor}")


def page_methods():
    st.markdown("## Methods")
    st.caption("Short version for reviewers and users who want the logic without the clutter.")

    st.markdown(
        """
        ### What the app measures
        - SEC filing language is scored with the Loughran-McDonald finance dictionary.
        - Scores are combined into a crisis exposure score and mapped to a rating.
        - Recent headlines are scored with the same language framework for comparison.
        - Optional model output adds a crisis-probability estimate.

        ### Why this version is cleaner
        - Guided step-by-step workflow
        - Less text on the main pages
        - Results grouped into tabs
        - Details hidden behind expanders
        - Plain-English labels for non-technical users

        ### Core sources
        - SEC EDGAR filings
        - Yahoo Finance price data and headlines
        - WRDS / Compustat for optional model training
        """
    )


# ------------------------------------------------------------
# Main app
# ------------------------------------------------------------
inject_css()
page = sidebar_navigation()

if page == "Overview":
    page_overview()
elif page == "Setup":
    page_setup()
elif page == "Analyze":
    page_analyze()
elif page == "Review":
    page_review()
elif page == "Action Plan":
    page_action_plan()
else:
    page_methods()
