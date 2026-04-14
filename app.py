# app.py
# Corporate Disclosure Crisis Radar — Main Streamlit Application
# AC820/BA870 Financial Analytics, Spring 2026 | Boston University Questrom
# 中文: Streamlit 主程序，运行方式: streamlit run app.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

import edgar
import scorer
import models
import newswire

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Corporate Disclosure Crisis Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state initialisation ─────────────────────────────────────────────
# 中文: 用 session_state 保存跨页面状态，避免重复计算
if "word_sets"    not in st.session_state: st.session_state.word_sets    = None
if "crisis_model" not in st.session_state: st.session_state.crisis_model = None
if "results"      not in st.session_state: st.session_state.results      = {}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 Crisis Radar")
    st.caption("Corporate Disclosure Risk Monitor")
    st.divider()

    # Navigation
    # 中文: 页面导航
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔐 Settings", "🔍 Analyze Company", "📊 Results & Charts"],
        label_visibility="collapsed",
    )
    st.divider()

    # Quick status
    # 中文: 快速显示各模块连接状态
    st.caption("**Module Status**")
    lm_ready  = st.session_state.word_sets    is not None
    ml_ready  = st.session_state.crisis_model is not None
    nw_ready  = st.session_state.get("nw_connected", False)

    st.markdown(
        f"{'✅' if lm_ready  else '⬜'} LM Dictionary\n\n"
        f"{'✅' if ml_ready  else '⬜'} Logit Model\n\n"
        f"{'✅' if nw_ready  else '⬜'} Access Newswire"
    )

# ════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("📡 Corporate Disclosure Crisis Radar")
    st.markdown(
        "**Detect PR crisis signals in SEC filings before the market reacts.**\n\n"
        "This tool analyzes the language of S&P 500 companies' 10-K and 10-Q filings "
        "using the Loughran-McDonald Finance Dictionary, scores them across 7 dimensions, "
        "and maps the results to a PR risk rating (A–D) with SCCT-grounded communication guidance."
    )
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📥 Data Sources")
        st.markdown(
            "- **SEC EDGAR** — live 10-K / 10-Q text\n"
            "- **Yahoo Finance** — stock price overlay\n"
            "- **WRDS Compustat** — Logit model training\n"
            "- **Access Newswire** — PR tone comparison"
        )
    with col2:
        st.markdown("### 🤖 Models")
        st.markdown(
            "- **LM Dictionary** — 7-dimension NLP scoring\n"
            "- **Crisis Exposure Score** — weighted risk index\n"
            "- **Logistic Regression** — crisis probability\n"
            "- **SCCT Framework** — response strategy"
        )
    with col3:
        st.markdown("### 🗺️ How to Use")
        st.markdown(
            "1. Go to **Settings** — enter credentials\n"
            "2. Go to **Analyze Company** — pick ticker\n"
            "3. Go to **Results & Charts** — view output\n"
        )

    st.divider()
    st.info(
        "💡 **Start here →** Click **Settings** in the left panel to load the "
        "LM Dictionary and connect WRDS / Access Newswire."
    )

    # LP/GP context note
    # 中文: LP/GP 信任背景说明
    with st.expander("📌 Why this matters — the LP/GP trust problem"):
        st.markdown(
            "Recent events in U.S. credit markets have shown that poorly managed "
            "disclosure language can trigger institutional investor trust breakdowns. "
            "When GPs file disclosures with ambiguous, litigious, or sharply negative language, "
            "LPs may reduce commitments or exit — a suboptimal outcome for both parties.\n\n"
            "This tool helps both GPs and LPs **get ahead of the crisis** by identifying "
            "language risk patterns *before* the market or media amplifies them."
        )

# ════════════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔐 Settings":
    st.title("🔐 Settings & Credentials")
    st.caption("Credentials are stored in session memory only and never saved to disk.")
    # 中文: 账号信息只存在当前会话内存中，不写入磁盘

    st.divider()

    # ── LM Dictionary ──
    st.subheader("📖 Loughran-McDonald Dictionary")
    st.markdown(
        "Download the CSV from "
        "[Notre Dame SRAF](https://sraf.nd.edu/loughranmcdonald-master-dictionary/) "
        "and place it at `data/LM_MasterDictionary.csv`."
    )
    col_lm1, col_lm2 = st.columns([3, 1])
    with col_lm1:
        lm_path = st.text_input(
            "CSV path", value="data/LM_MasterDictionary.csv", label_visibility="collapsed"
        )
    with col_lm2:
        if st.button("Load Dictionary", use_container_width=True):
            with st.spinner("Loading..."):
                ws = scorer.load_lm_dictionary(lm_path)
                st.session_state.word_sets = ws
            total = sum(len(v) for v in ws.values())
            st.success(f"✅ Loaded {total:,} entries across {len(ws)} dimensions.")

    st.divider()

    # ── WRDS ──
    st.subheader("🏦 WRDS / Compustat (Model Training)")
    st.markdown("Used **only** for training the Logit model. Real-time predictions use public data.")
    # 中文: WRDS 仅用于训练，预测时用公开数据（符合课程要求）

    wrds_user = st.text_input("WRDS username", placeholder="your_wrds_username")
    if st.button("Connect WRDS & Train Model"):
        with st.spinner("Connecting to WRDS and training model... (~30s)"):
            m = models.build_and_train_model(wrds_user if wrds_user else None)
            st.session_state.crisis_model = m
        auc_str = (f"{m.cv_auc[0]:.3f} ± {m.cv_auc[1]:.3f}"
                   if m.cv_auc else "N/A")
        st.success(f"✅ Model trained. 5-Fold CV AUC: {auc_str}")

    st.divider()

    # ── Access Newswire ──
    st.subheader("📰 Access Newswire (PR Comparison)")
    nw_key = st.text_input("API key", type="password", placeholder="your_api_key_here")
    if st.button("Connect Newswire"):
        with st.spinner("Testing connection..."):
            ok = newswire.test_connection(nw_key)
        if ok:
            st.session_state["nw_key"]       = nw_key
            st.session_state["nw_connected"] = True
            st.success("✅ Access Newswire connected.")
        else:
            st.error("❌ Connection failed. Check your API key.")

    st.divider()
    st.info("Once all modules are loaded, go to **Analyze Company** in the sidebar.")

# ════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYZE COMPANY
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Analyze Company":
    st.title("🔍 Analyze Company")

    # Load word sets if not yet loaded
    # 中文: 如果词典还没加载就用 fallback 版本
    if st.session_state.word_sets is None:
        st.session_state.word_sets = scorer.load_lm_dictionary()

    # ── Company selection ──
    with st.spinner("Loading S&P 500 list..."):
        sp500 = edgar.get_sp500_tickers()

    options = sp500.apply(lambda r: f"{r['ticker']} — {r['company']}", axis=1).tolist()
    selected  = st.selectbox("Select a company", options)
    ticker    = selected.split(" — ")[0]
    form_type = st.radio("Filing type", ["10-K  (Annual)", "10-Q  (Quarterly)"],
                         horizontal=True)
    form_code = "10-K" if "10-K" in form_type else "10-Q"

    st.divider()

    if st.button("▶ Run Analysis", type="primary", use_container_width=False):
        progress = st.progress(0, text="Fetching CIK...")

        # Step 1 — CIK
        cik = edgar.get_cik(ticker)
        if not cik:
            st.error(f"Could not find CIK for {ticker}.")
            st.stop()
        progress.progress(20, text="Fetching filing list...")

        # Step 2 — Filings
        filings = edgar.get_filings(cik, form_code)
        if filings.empty:
            st.error(f"No {form_code} filings found for {ticker}.")
            st.stop()
        progress.progress(40, text="Downloading filing text (15–30s)...")

        # Step 3 — Filing text
        latest      = filings.iloc[0]
        filing_text = edgar.get_filing_text(cik, latest["accession"])
        progress.progress(65, text="Scoring text...")

        # Step 4 — Score
        scores = scorer.score_text(filing_text, st.session_state.word_sets)
        progress.progress(80, text="Computing ratings and guidance...")

        # Step 5 — Rating + SCCT
        rating, color, rating_label = scorer.assign_rating(scores["crisis_score"])

        # Step 6 — Logit probability
        crisis_prob = None
        if st.session_state.crisis_model:
            crisis_prob = st.session_state.crisis_model.predict_from_scores(scores)

        # Step 7 — Newswire comparison
        pr_div = None
        if st.session_state.get("nw_connected"):
            pr_df = newswire.fetch_releases(
                ticker, latest["filed_date"], st.session_state["nw_key"]
            )
            if not pr_df.empty:
                pr_text   = " ".join(pr_df["content"].fillna("").tolist())
                pr_scores = scorer.score_text(pr_text, st.session_state.word_sets)
                divergence = newswire.compute_divergence(scores, pr_scores)
                pr_div     = divergence.get("total")

        guidance = scorer.get_scct_guidance(scores, pr_div)
        progress.progress(100, text="Done.")
        progress.empty()

        # Save to session state
        # 中文: 保存结果到 session_state，供 Results 页面使用
        st.session_state.results = {
            "ticker":      ticker,
            "form_code":   form_code,
            "filed_date":  latest["filed_date"],
            "cik":         cik,
            "scores":      scores,
            "rating":      rating,
            "color":       color,
            "rating_label":rating_label,
            "crisis_prob": crisis_prob,
            "pr_div":      pr_div,
            "guidance":    guidance,
            "filings":     filings,
        }

        st.success(
            f"✅ Analysis complete — **{ticker}** {form_code} "
            f"({latest['filed_date']})  |  Rating: **[{rating}]** {rating_label}"
        )
        st.info("➡️ Go to **Results & Charts** in the sidebar to see full output.")

# ════════════════════════════════════════════════════════════════════════════
# PAGE: RESULTS & CHARTS
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Results & Charts":
    st.title("📊 Results & Charts")

    r = st.session_state.results
    if not r:
        st.warning("No results yet. Go to **Analyze Company** first.")
        st.stop()

    ticker      = r["ticker"]
    form_code   = r["form_code"]
    filed_date  = r["filed_date"]
    scores      = r["scores"]
    rating      = r["rating"]
    color       = r["color"]
    rating_label= r["rating_label"]
    crisis_prob = r["crisis_prob"]
    pr_div      = r["pr_div"]
    guidance    = r["guidance"]

    # ── Top KPI row ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])

    with c1:
        st.markdown(
            f"<div style='background:{color}18;border-left:6px solid {color};"
            f"padding:18px;border-radius:8px;text-align:center'>"
            f"<div style='font-size:52px;font-weight:700;color:{color}'>{rating}</div>"
            f"<div style='font-size:13px;font-weight:600;color:{color}'>PR Crisis Rating</div>"
            f"<div style='font-size:11px;color:#666;margin-top:4px'>{rating_label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with c2:
        s = scores["net_sentiment"]
        st.metric(
            "Net Sentiment",
            f"{s:+.4f}",
            "Optimistic ↑" if s > 0 else "Cautious ↓",
        )

    with c3:
        prob_str = f"{crisis_prob:.1%}" if crisis_prob is not None else "N/A"
        st.metric(
            "Logit Crisis Prob.",
            prob_str,
            help="Probability of a negative market/reputational event within 90 days",
            # 中文: 未来 90 天内出现负面市场/声誉事件的概率（Logit 模型预测）
        )

    with c4:
        div_str = f"{pr_div:.2f}" if pr_div is not None else "N/A"
        st.metric(
            "PR Divergence",
            div_str,
            help="Tone gap between SEC filing and press releases (>4.0 = significant)",
            # 中文: SEC 文件与新闻稿的语气差距（>4.0 视为显著不一致）
        )

    st.divider()

    # ── Charts row ───────────────────────────────────────────────────────────
    col_radar, col_price = st.columns(2)

    with col_radar:
        st.subheader("Crisis Dimension Radar")
        # 中文: 危机维度雷达图
        dims_k = ["uncertainty","litigious","negative","weak_modal","constraining"]
        dims_l = ["Uncertainty","Litigious","Negative","Weak Modal","Constraining"]
        vals   = [scores.get(f"{d}_pct", 0) for d in dims_k]

        # Convert hex color to rgba for fill opacity (Plotly rejects 8-digit hex)
        # 中文: Plotly 不支持 8 位 hex，改用 rgba 格式设置透明度
        def _hex_to_rgba(hex_color: str, alpha: float = 0.19) -> str:
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

        fig1 = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]], theta=dims_l + [dims_l[0]],
            fill="toself", fillcolor=_hex_to_rgba(color),
            line=dict(color=color, width=2.5),
        ))
        fig1.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(vals)*1.4 or 1])),
            showlegend=False, height=380,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_price:
        st.subheader(f"Stock Price ±60 Days Around {form_code} Filing")
        # 中文: 文件提交前后 60 天股价走势
        try:
            filing_dt = datetime.strptime(filed_date, "%Y-%m-%d")
            start_dt  = (filing_dt - timedelta(days=60)).strftime("%Y-%m-%d")
            end_dt    = (filing_dt + timedelta(days=60)).strftime("%Y-%m-%d")
            price_df  = yf.download(ticker, start=start_dt, end=end_dt,
                                    progress=False, auto_adjust=True)

            # Newer yfinance returns MultiIndex columns — flatten to single level
            # 中文: 新版 yfinance 返回多级列名，需先压平成单级
            if isinstance(price_df.columns, pd.MultiIndex):
                price_df.columns = [col[0] for col in price_df.columns]

            price_df.reset_index(inplace=True)

            # Ensure Date column is datetime and Close is numeric
            # 中文: 确保日期列是 datetime 类型，收盘价是数值类型
            price_df["Date"]  = pd.to_datetime(price_df["Date"])
            price_df["Close"] = pd.to_numeric(price_df["Close"], errors="coerce")
            price_df.dropna(subset=["Close"], inplace=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=price_df["Date"], y=price_df["Close"],
                mode="lines", line=dict(color="#1565C0", width=2), name="Close"
            ))
            # add_vline() has unstable datetime handling across Plotly versions.
            # Use add_shape + add_annotation instead — fully compatible.
            # 中文: add_vline 在不同 Plotly 版本对 datetime 支持不稳定
            #       改用 add_shape 画竖线 + add_annotation 加标注，兼容性更好
            fig2.add_shape(
                type="line",
                x0=filed_date, x1=filed_date,
                y0=0, y1=1, yref="paper",
                line=dict(color=color, width=2.5, dash="dash"),
            )
            fig2.add_annotation(
                x=filed_date, y=1.02, yref="paper",
                text=f"{form_code} filed [{rating}]",
                showarrow=False,
                font=dict(color=color, size=11),
                xanchor="left",
            )
            fig2.update_layout(
                height=380, plot_bgcolor="#FAFAFA",
                xaxis_title="Date", yaxis_title="Price (USD)",
                hovermode="x unified",
                margin=dict(l=20, r=20, t=20, b=40),
            )
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load stock data: {e}")

    st.divider()

    # ── Dimension bar chart ──────────────────────────────────────────────────
    st.subheader("LM Dimension Scores (per 1,000 words)")
    # 中文: 各语言维度得分柱状图
    dim_map = {
        "uncertainty":  "Uncertainty",
        "litigious":    "Litigious",
        "negative":     "Negative",
        "weak_modal":   "Weak Modal",
        "constraining": "Constraining",
        "positive":     "Positive",
        "strong_modal": "Strong Modal",
    }
    bar_vals   = [scores.get(f"{k}_pct", 0) for k in dim_map]
    bar_labels = list(dim_map.values())
    bar_colors = ["#2E7D32" if k in ("positive","strong_modal") else color
                  for k in dim_map]

    fig3 = go.Figure(go.Bar(
        x=bar_vals, y=bar_labels, orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.2f}" for v in bar_vals], textposition="outside",
    ))
    fig3.update_layout(
        height=340, plot_bgcolor="#FAFAFA",
        xaxis_title="Words per 1,000",
        margin=dict(l=20, r=60, t=20, b=40),
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # ── SCCT Guidance ────────────────────────────────────────────────────────
    st.subheader("🎯 SCCT Crisis Communication Guidance")
    # 中文: SCCT 危机沟通建议

    g_col1, g_col2 = st.columns([1, 2])
    with g_col1:
        st.markdown("**Crisis Type**")
        st.info(guidance["crisis_type"])
        st.markdown("**SCCT Cluster**")
        st.info(guidance["cluster"])
        st.markdown("**Recommended Strategy**")
        st.info(guidance["strategy"])

    with g_col2:
        st.markdown("**Recommended Action**")
        st.success(guidance["action"])
        st.markdown("**LP / Investor Trust Signal**")
        lp_color = (
            "🔴" if "HIGH"     in guidance["lp_signal"] else
            "🟡" if "MODERATE" in guidance["lp_signal"] else "🟢"
        )
        st.markdown(f"{lp_color} {guidance['lp_signal']}")

    st.divider()

    # ── Historical filings table ─────────────────────────────────────────────
    with st.expander("📁 Historical Filing List"):
        # 中文: 历史文件列表（Phase 2 会对每份文件打分并画趋势折线图）
        st.caption("Phase 2: score each filing and plot sentiment trend over time.")
        st.dataframe(r["filings"], use_container_width=True)

    # ── Raw scores table ────────────────────────────────────────────────────
    with st.expander("🔢 Raw Scores"):
        score_df = pd.DataFrame([
            {"Metric": k, "Value": v}
            for k, v in scores.items()
        ])
        st.dataframe(score_df, use_container_width=True)
