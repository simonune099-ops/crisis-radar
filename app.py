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
# newswire import removed — Access Newswire API not available
# Replaced by: (1) SEC 8-K filings as free PR proxy, (2) Meltwater CSV upload

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
        ["🏠 Home", "🔐 Settings", "🔍 Analyze Company",
         "📊 Results & Charts", "📋 Crisis Playbook"],
        label_visibility="collapsed",
    )
    st.divider()

    # Quick status
    # 中文: 快速显示各模块连接状态
    st.caption("**Module Status**")
    lm_ready  = st.session_state.word_sets    is not None
    ml_ready  = st.session_state.crisis_model is not None
    mt_ready  = st.session_state.get("meltwater_df") is not None

    st.markdown(
        f"{'✅' if lm_ready  else '⬜'} LM Dictionary\n\n"
        f"{'✅' if ml_ready  else '⬜'} Logit Model\n\n"
        f"{'✅' if mt_ready  else '⬜'} Meltwater Coverage"
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
            "- **SEC EDGAR** — live 10-K / 10-Q + 8-K filings\n"
            "- **Yahoo Finance** — Event Study (CAR vs SPY)\n"
            "- **WRDS Compustat** — Logit model training\n"
            "- **Meltwater CSV** — Media coverage analysis"
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
        "LM Dictionary, connect WRDS, and optionally upload Meltwater media coverage."
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

    # ── Meltwater CSV Upload ─────────────────────────────────────────────────
    # Meltwater doesn't offer API access at all tiers, but allows manual CSV export.
    # Workflow: Meltwater → Search/Monitor → Export → Upload here.
    # 中文: Meltwater 不开放 API，但支持手动导出 CSV。在 Meltwater 网页端搜索后导出，上传到这里。
    st.subheader("📊 Meltwater — Media Coverage Analysis")

    with st.expander("ℹ️ How to export from Meltwater", expanded=False):
        st.markdown(
            "**Step-by-step export from your Meltwater account:**\n\n"
            "1. Log into [meltwater.com](https://www.meltwater.com) with your account\n"
            "2. Go to **Media Monitoring** → search for the company ticker or name\n"
            "3. Set the date range to match ±90 days around the filing date\n"
            "4. Click **Export** (top-right) → select **CSV** or **Excel**\n"
            "5. Upload the downloaded file below\n\n"
            "**Expected columns** (Meltwater standard export):\n"
            "`Date` / `Published`, `Title` / `Headline`, `Source`, "
            "`Sentiment`, `Reach` / `Audience`, `Country`, `Snippet` / `Summary`\n\n"
            "The parser handles different column name variations automatically."
        )

    mt_file = st.file_uploader(
        "Upload Meltwater export (.csv or .xlsx)",
        type=["csv", "xlsx", "xls"],
        help="Export from Meltwater: Media Monitoring → Export → CSV",
    )

    if mt_file is not None:
        try:
            # Parse uploaded file — handle both CSV and Excel
            # 中文: 自动处理 CSV 和 Excel 两种格式
            if mt_file.name.endswith((".xlsx", ".xls")):
                mt_df = pd.read_excel(mt_file)
            else:
                mt_df = pd.read_csv(mt_file, encoding="utf-8", errors="replace")

            # Normalize column names — Meltwater uses different names by region/version
            # 中文: 统一列名（Meltwater 不同版本列名不同）
            col_map = {}
            for col in mt_df.columns:
                c_lower = col.lower().strip()
                if c_lower in ("date", "published", "publication date", "publish date", "hit date"):
                    col_map[col] = "date"
                elif c_lower in ("title", "headline", "article title", "hit title"):
                    col_map[col] = "title"
                elif c_lower in ("source", "media", "outlet", "media outlet", "source name"):
                    col_map[col] = "source"
                elif c_lower in ("sentiment", "sentiment score", "tone"):
                    col_map[col] = "sentiment"
                elif c_lower in ("reach", "audience", "estimated reach", "readership"):
                    col_map[col] = "reach"
                elif c_lower in ("country", "region", "geography"):
                    col_map[col] = "country"
                elif c_lower in ("snippet", "summary", "content", "excerpt", "body"):
                    col_map[col] = "snippet"
                elif c_lower in ("url", "link", "article url"):
                    col_map[col] = "url"

            mt_df = mt_df.rename(columns=col_map)

            # Parse date
            if "date" in mt_df.columns:
                mt_df["date"] = pd.to_datetime(mt_df["date"], errors="coerce")

            # Parse reach as numeric
            if "reach" in mt_df.columns:
                mt_df["reach"] = pd.to_numeric(
                    mt_df["reach"].astype(str).str.replace(",", "").str.replace(" ", ""),
                    errors="coerce"
                )

            st.session_state["meltwater_df"] = mt_df
            found_cols = [c for c in ["date","title","source","sentiment","reach","snippet"]
                          if c in mt_df.columns]

            st.success(
                f"✅ Meltwater data loaded: **{len(mt_df):,} articles** | "
                f"Columns detected: {', '.join(found_cols)}"
            )

            # Quick preview
            preview_cols = [c for c in ["date","title","source","sentiment","reach"]
                            if c in mt_df.columns]
            st.dataframe(mt_df[preview_cols].head(5), use_container_width=True)

        except Exception as e:
            st.error(f"Failed to parse file: {e}")
            st.caption("Try re-exporting from Meltwater as UTF-8 CSV.")

    elif st.session_state.get("meltwater_df") is not None:
        existing = st.session_state["meltwater_df"]
        st.info(f"✅ Meltwater data already loaded: {len(existing):,} articles")
        if st.button("Clear Meltwater data"):
            st.session_state["meltwater_df"] = None
            st.rerun()

    st.divider()
    st.info("Once modules are loaded, go to **Analyze Company** in the sidebar.")

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

        # Step 7 — Meltwater media coverage analysis (if CSV uploaded)
        # 中文: 如果已上传 Meltwater CSV，计算媒体情绪与 SEC 文件的对比
        pr_div      = None
        mt_analysis = None
        mt_df       = st.session_state.get("meltwater_df")

        if mt_df is not None and "date" in mt_df.columns:
            try:
                filing_dt_mt = pd.to_datetime(latest["filed_date"])
                # Filter to ±90 days around filing date
                window = mt_df[
                    (mt_df["date"] >= filing_dt_mt - pd.Timedelta(days=90)) &
                    (mt_df["date"] <= filing_dt_mt + pd.Timedelta(days=90))
                ].copy()

                if len(window) > 0:
                    # Sentiment distribution from Meltwater's own column
                    if "sentiment" in window.columns:
                        sent_counts = window["sentiment"].str.lower().value_counts()
                        pos_count = sent_counts.get("positive", 0)
                        neg_count = sent_counts.get("negative", 0)
                        neu_count = sent_counts.get("neutral", 0)
                        total_mt  = len(window)
                        mt_neg_pct = neg_count / total_mt * 100 if total_mt else 0

                        # Divergence: compare SEC filing negative% with media negativity%
                        sec_neg_pct = scores.get("negative_pct", 0) * 0.1  # scale to 0-100
                        pr_div = abs(sec_neg_pct - mt_neg_pct)
                    else:
                        neg_count = neu_count = pos_count = 0
                        mt_neg_pct = 0

                    # Top sources by reach
                    if "reach" in window.columns:
                        top_sources = (window.groupby("source")["reach"].sum()
                                       .sort_values(ascending=False).head(5)
                                       if "source" in window.columns else pd.Series())
                    else:
                        top_sources = (window["source"].value_counts().head(5)
                                       if "source" in window.columns else pd.Series())

                    mt_analysis = {
                        "total_articles":  len(window),
                        "positive_count":  int(pos_count),
                        "neutral_count":   int(neu_count),
                        "negative_count":  int(neg_count),
                        "mt_neg_pct":      round(mt_neg_pct, 1),
                        "top_sources":     top_sources,
                        "window_df":       window,
                        "filing_date":     filing_dt_mt,
                    }
            except Exception:
                pass

        guidance = scorer.get_scct_guidance(scores, pr_div)

        # Step 8 — Enhanced PR/Crisis frameworks (new)
        # 中文: 新增：Lerbinger 分类、生命周期阶段、Issues Management 分级、行动清单
        lerbinger_type  = scorer.classify_lerbinger_type(scores)
        lifecycle_stage = scorer.get_lifecycle_stage(scores, rating)
        triage          = scorer.triage_issue_severity(scores, rating)
        checklist       = scorer.get_proactive_checklist(scores, rating)

        # Step 9 — 8-K filing comparison (replaces Access Newswire when unavailable)
        # 中文: 抓取近期 8-K（SEC 原生新闻稿），与 10-K 语气对比
        eightk_filings = edgar.get_8k_filings(cik, count=5)
        eightk_scores  = None
        eightk_div     = None
        if not eightk_filings.empty and st.session_state.word_sets:
            try:
                ek_texts = []
                for _, ek_row in eightk_filings.head(3).iterrows():
                    ek_text = edgar.get_filing_text(cik, ek_row["accession"], max_chars=30_000)
                    if ek_text:
                        ek_texts.append(ek_text)
                if ek_texts:
                    combined_ek = " ".join(ek_texts)
                    eightk_scores = scorer.score_text(combined_ek, st.session_state.word_sets)
                    eightk_div = abs(
                        scores.get("negative_pct", 0) - eightk_scores.get("negative_pct", 0)
                    ) + abs(
                        scores.get("uncertainty_pct", 0) - eightk_scores.get("uncertainty_pct", 0)
                    )
            except Exception:
                pass

        progress.progress(100, text="Done.")
        progress.empty()

        # Save to session state
        # 中文: 保存结果到 session_state，供 Results 页面使用
        st.session_state.results = {
            "ticker":         ticker,
            "form_code":      form_code,
            "filed_date":     latest["filed_date"],
            "cik":            cik,
            "scores":         scores,
            "rating":         rating,
            "color":          color,
            "rating_label":   rating_label,
            "crisis_prob":    crisis_prob,
            "pr_div":         pr_div,
            "guidance":       guidance,
            "filings":        filings,
            "lerbinger_type": lerbinger_type,
            "lifecycle_stage": lifecycle_stage,
            "triage":         triage,
            "checklist":      checklist,
            "eightk_filings": eightk_filings,
            "eightk_scores":  eightk_scores,
            "eightk_div":     eightk_div,
            "mt_analysis":    mt_analysis,
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
    mt_analysis = r.get("mt_analysis")

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
        if mt_analysis:
            neg_pct = mt_analysis["mt_neg_pct"]
            st.metric(
                "Media Neg. Coverage",
                f"{neg_pct:.1f}%",
                delta=f"{mt_analysis['total_articles']} articles",
                help="% negative articles in Meltwater coverage ±90 days around filing",
                # 中文: Meltwater 媒体报道中负面文章占比（文件日期前后 90 天）
            )
        else:
            div_str = f"{pr_div:.2f}" if pr_div is not None else "Upload Meltwater CSV"
            st.metric(
                "Media Divergence",
                div_str,
                help="Upload Meltwater CSV in Settings to enable media analysis",
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
        st.subheader(f"📈 Event Study — CAR ±60 Days")
        st.caption("Cumulative Abnormal Return vs. S&P 500 market model around filing date")
        # 中文: 事件研究——超额累计回报（相对 S&P 500 基准模型）
        # Method: Market Model (Brown & Warner 1985)
        #   Estimation window: [-250, -61] trading days before filing
        #   Event window:      [-60, +60]  trading days around filing
        #   AR_t = R_stock_t - (alpha + beta * R_market_t)
        #   CAR  = cumulative sum of AR_t over event window
        try:
            import numpy as np
            from sklearn.linear_model import LinearRegression

            filing_dt   = datetime.strptime(filed_date, "%Y-%m-%d")
            # Wider download to ensure enough trading days in estimation window
            dl_start = (filing_dt - timedelta(days=400)).strftime("%Y-%m-%d")
            dl_end   = (filing_dt + timedelta(days=90)).strftime("%Y-%m-%d")

            def _clean_download(sym):
                df = yf.download(sym, start=dl_start, end=dl_end,
                                 progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                df = df.reset_index()
                df["Date"]  = pd.to_datetime(df["Date"])
                df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
                return df[["Date", "Close"]].dropna().set_index("Date")

            stock_df  = _clean_download(ticker)
            market_df = _clean_download("SPY")   # S&P 500 ETF as benchmark

            # Align on common trading dates
            merged = stock_df.join(market_df, how="inner", lsuffix="_s", rsuffix="_m")
            merged.columns = ["stock", "market"]
            merged["r_stock"]  = merged["stock"].pct_change()
            merged["r_market"] = merged["market"].pct_change()
            merged = merged.dropna()

            # Split estimation vs event window
            filing_idx = merged.index.searchsorted(filing_dt)
            est_start  = max(0, filing_idx - 250)
            est_end    = max(0, filing_idx - 60)
            evt_start  = max(0, filing_idx - 60)
            evt_end    = min(len(merged), filing_idx + 61)

            est_df = merged.iloc[est_start:est_end]
            evt_df = merged.iloc[evt_start:evt_end].copy()

            if len(est_df) < 30 or len(evt_df) < 5:
                raise ValueError("Insufficient trading days for event study.")

            # OLS market model
            X_est = est_df[["r_market"]].values
            y_est = est_df["r_stock"].values
            reg   = LinearRegression().fit(X_est, y_est)
            alpha_hat = reg.intercept_
            beta_hat  = reg.coef_[0]

            # Abnormal returns in event window
            evt_df["expected_r"] = alpha_hat + beta_hat * evt_df["r_market"]
            evt_df["AR"]         = evt_df["r_stock"] - evt_df["expected_r"]
            evt_df["CAR"]        = evt_df["AR"].cumsum() * 100   # in %

            # Compute event-day index offset for x-axis
            evt_df["event_day"] = range(-len(evt_df[:filing_idx - evt_start]),
                                        len(evt_df) - len(evt_df[:filing_idx - evt_start]))

            filing_day_idx = filing_idx - evt_start
            evt_df_reset   = evt_df.reset_index()

            fig2 = go.Figure()
            # CAR line
            fig2.add_trace(go.Scatter(
                x=evt_df_reset["Date"], y=evt_df["CAR"].values,
                mode="lines", line=dict(color="#1565C0", width=2.5),
                name="CAR (%)", hovertemplate="%{x|%b %d}<br>CAR: %{y:.2f}%<extra></extra>",
            ))
            # Zero baseline
            fig2.add_shape(type="line",
                x0=evt_df_reset["Date"].iloc[0], x1=evt_df_reset["Date"].iloc[-1],
                y0=0, y1=0, yref="y",
                line=dict(color="#999", width=1, dash="dot"),
            )
            # Filing date vertical line
            fig2.add_shape(type="line",
                x0=filed_date, x1=filed_date, y0=0, y1=1, yref="paper",
                line=dict(color=color, width=2, dash="dash"),
            )
            fig2.add_annotation(
                x=filed_date, y=1.04, yref="paper",
                text=f"{form_code} filed [{rating}]",
                showarrow=False, font=dict(color=color, size=10), xanchor="left",
            )
            # Final CAR annotation
            final_car = evt_df["CAR"].iloc[-1]
            car_color = "#B71C1C" if final_car < 0 else "#2E7D32"
            fig2.add_annotation(
                x=evt_df_reset["Date"].iloc[-1], y=final_car,
                text=f"  CAR: {final_car:+.2f}%",
                showarrow=False, font=dict(color=car_color, size=11), xanchor="left",
            )
            fig2.update_layout(
                height=380, plot_bgcolor="#FAFAFA",
                xaxis_title="Date",
                yaxis_title="Cumulative Abnormal Return (%)",
                hovermode="x unified",
                margin=dict(l=20, r=20, t=20, b=40),
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig2, use_container_width=True)

            # CAR summary stats
            car_pre  = evt_df.loc[evt_df.index < filing_dt, "CAR"]
            car_post = evt_df.loc[evt_df.index >= filing_dt, "CAR"]
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Pre-filing CAR",  f"{car_pre.iloc[-1]:+.2f}%" if len(car_pre)  else "N/A")
            cc2.metric("Post-filing CAR", f"{car_post.iloc[-1]:+.2f}%" if len(car_post) else "N/A")
            cc3.metric("Market β", f"{beta_hat:.3f}",
                       help="Beta estimated from 190-day pre-event window vs SPY")

        except Exception as e:
            st.warning(f"Could not compute event study: {e}")
            # Fallback: simple price chart
            # 中文: CAR 失败时降级为简单股价图
            try:
                filing_dt = datetime.strptime(filed_date, "%Y-%m-%d")
                start_dt  = (filing_dt - timedelta(days=60)).strftime("%Y-%m-%d")
                end_dt    = (filing_dt + timedelta(days=60)).strftime("%Y-%m-%d")
                price_df  = yf.download(ticker, start=start_dt, end=end_dt,
                                        progress=False, auto_adjust=True)
                if isinstance(price_df.columns, pd.MultiIndex):
                    price_df.columns = [c[0] for c in price_df.columns]
                price_df.reset_index(inplace=True)
                price_df["Date"]  = pd.to_datetime(price_df["Date"])
                price_df["Close"] = pd.to_numeric(price_df["Close"], errors="coerce")
                price_df.dropna(subset=["Close"], inplace=True)
                fig2b = go.Figure(go.Scatter(
                    x=price_df["Date"], y=price_df["Close"],
                    mode="lines", line=dict(color="#1565C0", width=2)
                ))
                fig2b.add_shape(type="line", x0=filed_date, x1=filed_date,
                    y0=0, y1=1, yref="paper",
                    line=dict(color=color, width=2, dash="dash"))
                fig2b.update_layout(height=380, plot_bgcolor="#FAFAFA",
                    margin=dict(l=20, r=20, t=20, b=40))
                st.plotly_chart(fig2b, use_container_width=True)
            except Exception:
                st.info("Stock data unavailable.")

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

    # ── Risk Map ─────────────────────────────────────────────────────────────
    st.subheader("🗺️ Crisis Risk Map")
    st.caption("Severity × Likelihood — inspired by AC820 Risk Map framework")
    # 中文: 危机风险地图（严重程度 × 发生可能性），来自 Week 1-2 课件 Risk Map

    rmap_col, rmap_info = st.columns([2, 1])
    with rmap_col:
        likelihood = scores.get("uncertainty_pct", 0) + scores.get("weak_modal_pct", 0)
        severity   = scores.get("negative_pct", 0) + scores.get("litigious_pct", 0) * 1.5

        # Background quadrant zones
        fig_map = go.Figure()
        # Green zone (low-low)
        fig_map.add_shape(type="rect", x0=0, y0=0, x1=7, y1=5,
            fillcolor="rgba(46,125,50,0.08)", line_width=0)
        # Yellow zone (high likelihood, low severity)
        fig_map.add_shape(type="rect", x0=7, y0=0, x1=14, y1=5,
            fillcolor="rgba(249,168,37,0.10)", line_width=0)
        # Orange zone (low likelihood, high severity)
        fig_map.add_shape(type="rect", x0=0, y0=5, x1=7, y1=12,
            fillcolor="rgba(230,81,0,0.10)", line_width=0)
        # Red zone (high-high)
        fig_map.add_shape(type="rect", x0=7, y0=5, x1=14, y1=12,
            fillcolor="rgba(183,28,28,0.12)", line_width=0)

        # Zone labels
        for (lx, ly, label) in [(3.5, 2.5, "Low Risk"), (10.5, 2.5, "Monitor"),
                                  (3.5, 8.5, "Manage"), (10.5, 8.5, "Critical")]:
            fig_map.add_annotation(x=lx, y=ly, text=label, showarrow=False,
                font=dict(size=12, color="#ccc"), opacity=0.7)

        # Company dot
        fig_map.add_trace(go.Scatter(
            x=[likelihood], y=[severity], mode="markers+text",
            marker=dict(size=18, color=color, symbol="diamond",
                        line=dict(color="white", width=2)),
            text=[f"  {ticker}"], textposition="middle right",
            textfont=dict(color=color, size=12),
            name=ticker,
        ))
        fig_map.update_layout(
            height=320, plot_bgcolor="#FAFAFA",
            xaxis=dict(title="Likelihood Proxy (Uncertainty + Weak Modal per 1000w)",
                       range=[0, 14], gridcolor="#eee"),
            yaxis=dict(title="Severity Proxy (Negative + Litigious×1.5 per 1000w)",
                       range=[0, 12], gridcolor="#eee"),
            showlegend=False,
            margin=dict(l=20, r=20, t=10, b=40),
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with rmap_info:
        st.markdown("**Risk Map Position**")
        quad = (
            "🔴 Critical Zone — High likelihood AND high severity"
            if likelihood > 7 and severity > 5 else
            "🟠 Manage — Low likelihood, high severity impact"
            if likelihood <= 7 and severity > 5 else
            "🟡 Monitor — High likelihood, manageable severity"
            if likelihood > 7 and severity <= 5 else
            "🟢 Low Risk — Both dimensions within normal range"
        )
        st.info(quad)
        st.metric("Likelihood Score", f"{likelihood:.2f}")
        st.metric("Severity Score",   f"{severity:.2f}")
        st.caption(
            "Likelihood: Uncertainty + Weak Modal language signals\n\n"
            "Severity: Negative + Litigious language signals (weighted)\n\n"
            "Source: AC820 Week 1-2 Risk Map framework"
        )

    st.divider()

    # ── SCCT Guidance ────────────────────────────────────────────────────────
    st.subheader("🎯 SCCT Crisis Communication Guidance")
    st.caption("Coombs (2007) SCCT + Lerbinger Crisis Types + Victim Recovery Cycle")
    # 中文: SCCT 危机沟通建议（整合 Lerbinger 类型 + 受害者恢复循环）

    lerbinger_type  = r.get("lerbinger_type",  {})
    lifecycle_stage = r.get("lifecycle_stage", {})

    # Lerbinger type banner
    if lerbinger_type:
        l_icon = lerbinger_type.get("icon", "⚠️")
        l_name = lerbinger_type.get("type_name", "")
        l_color_map = {
            "Mismanagement / Misconduct":  "#B71C1C",
            "Stakeholder Confrontation":   "#E65100",
            "Technological Failure":       "#F9A825",
            "Environment & Sustainability":"#1565C0",
        }
        l_color = l_color_map.get(l_name, "#666")
        st.markdown(
            f"<div style='background:{l_color}15;border-left:5px solid {l_color};"
            f"padding:12px 16px;border-radius:6px;margin-bottom:12px'>"
            f"<b>{l_icon} Lerbinger Crisis Type: {l_name}</b><br>"
            f"<span style='font-size:13px;color:#444'>{lerbinger_type.get('description','')}</span><br>"
            f"<span style='font-size:11px;color:#888'>Root vulnerability: "
            f"{lerbinger_type.get('root_vulnerability','')} | "
            f"LM basis: {lerbinger_type.get('lm_signal_basis','')}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if lerbinger_type.get("secondary_type"):
            st.caption(f"⚠️ Secondary type also detected: {lerbinger_type['secondary_type']}")

    g_col1, g_col2 = st.columns([1, 2])
    with g_col1:
        st.markdown("**SCCT Crisis Cluster**")
        st.info(guidance["crisis_type"])
        st.markdown("**Recommended Strategy**")
        st.info(guidance["strategy"])
        st.markdown("**Advocacy ↔ Accommodation**")
        st.caption(guidance.get("advocacy_accommodation", ""))
        st.markdown("**LP / Investor Trust Signal**")
        st.markdown(guidance["lp_signal"])

    with g_col2:
        st.markdown("**Recommended Action**")
        st.success(guidance["action"])

        # Victim Recovery Cycle
        vr = guidance.get("victim_recovery", {})
        if vr:
            with st.expander("🫂 Victim Recovery Cycle Guidance (Coombs / AC820 Week 1)"):
                st.markdown("**Stage 1 — Feelings (Emotions & Trauma)**")
                st.markdown(f"> {vr.get('stage_1_feelings', '')}")
                st.markdown("**Stage 2 — Seeking Retribution (Accountability)**")
                st.markdown(f"> {vr.get('stage_2_retribution', '')}")
                st.markdown("**Stage 3 — Search for Healing (Remediation)**")
                st.markdown(f"> {vr.get('stage_3_healing', '')}")
                st.markdown("**Stage 4 — Victim's Needs (Safety, Dignity, Truth)**")
                st.markdown(f"> {vr.get('stage_4_needs', '')}")

        # Stealing Thunder tip
        if lerbinger_type.get("stealing_thunder_tip"):
            with st.expander("⚡ Stealing Thunder — Proactive Disclosure Strategy"):
                st.info(lerbinger_type["stealing_thunder_tip"])

    # 8-K vs 10-K tone comparison
    eightk_filings = r.get("eightk_filings")
    eightk_scores  = r.get("eightk_scores")
    eightk_div     = r.get("eightk_div")

    if eightk_scores is not None:
        st.divider()
        st.subheader("📰 SEC 8-K vs 10-K Tone Comparison")
        st.caption(
            "8-K filings = company's immediate public disclosures (earnings, M&A, leadership changes). "
            "Tone gap between 8-K and 10-K may signal PR narrative management risk."
        )
        # 中文: 8-K 是即时披露，与 10-K 语气差距大意味着公司对外沟通与内部文件不一致
        ek_col1, ek_col2, ek_col3 = st.columns(3)
        ek_col1.metric("10-K Negative",   f"{scores.get('negative_pct',0):.2f}/1000w")
        ek_col2.metric("8-K Negative",    f"{eightk_scores.get('negative_pct',0):.2f}/1000w")
        ek_col3.metric("Tone Divergence", f"{eightk_div:.2f}",
                       delta="⚠️ Narrative gap" if eightk_div > 3 else "✅ Consistent",
                       delta_color="inverse")

        if eightk_div and eightk_div > 3:
            st.warning(
                f"**Tone divergence of {eightk_div:.2f}** between 10-K and recent 8-K filings. "
                "The company's public narrative may not align with its formal disclosure — "
                "this is a PR risk signal. Apply the 'Stealing Thunder' principle: "
                "ensure the 10-K filing is not more pessimistic than public communications."
            )

        if not eightk_filings.empty:
            with st.expander(f"📋 Recent 8-K Filings ({len(eightk_filings)} records)"):
                st.dataframe(eightk_filings[["form","filed_date","accession"]],
                             use_container_width=True)

    st.divider()

    # ── Meltwater Media Coverage Analysis ────────────────────────────────────
    if mt_analysis:
        st.divider()
        st.subheader("📡 Meltwater — Media Coverage Analysis")
        st.caption(
            f"±90 days around {form_code} filing ({filed_date}) — "
            f"{mt_analysis['total_articles']} articles analysed"
        )
        # 中文: Meltwater 媒体报道分析（文件日期前后 90 天）

        # Sentiment breakdown
        mt_c1, mt_c2, mt_c3, mt_c4 = st.columns(4)
        mt_c1.metric("Total Articles",  mt_analysis["total_articles"])
        mt_c2.metric("Positive",  mt_analysis["positive_count"],
                     delta=f"{mt_analysis['positive_count']/mt_analysis['total_articles']*100:.0f}%")
        mt_c3.metric("Neutral",   mt_analysis["neutral_count"])
        mt_c4.metric("Negative",  mt_analysis["negative_count"],
                     delta=f"-{mt_analysis['mt_neg_pct']:.1f}%", delta_color="inverse")

        mt_left, mt_right = st.columns(2)

        # Sentiment pie chart
        with mt_left:
            if mt_analysis["positive_count"] + mt_analysis["negative_count"] + mt_analysis["neutral_count"] > 0:
                fig_pie = go.Figure(go.Pie(
                    labels=["Positive", "Neutral", "Negative"],
                    values=[mt_analysis["positive_count"],
                            mt_analysis["neutral_count"],
                            mt_analysis["negative_count"]],
                    marker_colors=["#2E7D32", "#9E9E9E", "#B71C1C"],
                    hole=0.45,
                    textinfo="label+percent",
                ))
                fig_pie.update_layout(
                    height=280, showlegend=False,
                    margin=dict(l=10, r=10, t=10, b=10),
                    title_text="Media Sentiment Mix",
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        # Coverage timeline (articles per week)
        with mt_right:
            wdf = mt_analysis["window_df"].copy()
            if "date" in wdf.columns and len(wdf) > 0:
                wdf["week"] = wdf["date"].dt.to_period("W").apply(lambda p: p.start_time)
                weekly = wdf.groupby("week").size().reset_index(name="count")
                if "sentiment" in wdf.columns:
                    neg_weekly = (wdf[wdf["sentiment"].str.lower() == "negative"]
                                  .groupby("week").size().reset_index(name="neg_count"))
                    weekly = weekly.merge(neg_weekly, on="week", how="left").fillna(0)

                fig_timeline = go.Figure()
                fig_timeline.add_trace(go.Bar(
                    x=weekly["week"], y=weekly["count"],
                    name="All articles", marker_color="#90CAF9",
                ))
                if "neg_count" in weekly.columns:
                    fig_timeline.add_trace(go.Bar(
                        x=weekly["week"], y=weekly["neg_count"],
                        name="Negative", marker_color="#EF9A9A",
                    ))
                # Filing date line
                fig_timeline.add_shape(
                    type="line", x0=filed_date, x1=filed_date,
                    y0=0, y1=1, yref="paper",
                    line=dict(color=color, width=2, dash="dash"),
                )
                fig_timeline.add_annotation(
                    x=filed_date, y=1.05, yref="paper",
                    text=f"{form_code} filed", showarrow=False,
                    font=dict(color=color, size=10),
                )
                fig_timeline.update_layout(
                    height=280, barmode="overlay",
                    xaxis_title="Week", yaxis_title="Articles",
                    plot_bgcolor="#FAFAFA",
                    margin=dict(l=10, r=10, t=10, b=40),
                    legend=dict(orientation="h", y=-0.25),
                    title_text="Coverage Volume Timeline",
                )
                st.plotly_chart(fig_timeline, use_container_width=True)

        # Top sources table
        top_src = mt_analysis.get("top_sources")
        if top_src is not None and len(top_src) > 0:
            with st.expander("📰 Top Media Sources by Reach"):
                src_df = top_src.reset_index()
                src_df.columns = ["Source", "Reach / Article Count"]
                st.dataframe(src_df, use_container_width=True)

        # SEC vs Media divergence warning
        if pr_div is not None and pr_div > 10:
            st.warning(
                f"⚠️ **Media-SEC divergence detected** (score: {pr_div:.1f}). "
                f"Media coverage is significantly more negative than the SEC filing language. "
                f"This gap is a PR risk signal — investors who read media before filings "
                f"may form a more negative view than the filing alone warrants. "
                f"Consider proactive IR communications to bridge the narrative gap."
            )
        elif pr_div is not None:
            st.success(
                f"✅ Media tone is broadly consistent with SEC filing language "
                f"(divergence score: {pr_div:.1f}). No significant narrative gap detected."
            )

        # Raw data table
        with st.expander("📋 Raw Meltwater Data (filtered window)"):
            display_cols = [c for c in ["date", "title", "source", "sentiment", "reach", "url"]
                            if c in mt_analysis["window_df"].columns]
            st.dataframe(mt_analysis["window_df"][display_cols], use_container_width=True)

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

# ════════════════════════════════════════════════════════════════════════════
# PAGE: CRISIS PLAYBOOK
# Full action plan integrating AC820 crisis management frameworks
# 中文: 危机行动手册页面——整合课程所有框架
# ════════════════════════════════════════════════════════════════════════════
elif page == "📋 Crisis Playbook":
    st.title("📋 Crisis Playbook")
    st.markdown(
        "Integrated crisis management action plan based on your filing analysis. "
        "Grounded in AC820 frameworks: Crisis Lifecycle, Lerbinger Types, "
        "SCCT, Issues Management Process, and Proactive Approach."
    )
    # 中文: 基于文件分析生成的完整危机应对手册，整合 AC820 所有核心框架

    r = st.session_state.results
    if not r:
        st.warning("No analysis results yet. Please go to **Analyze Company** first.")
        st.stop()

    ticker          = r["ticker"]
    rating          = r["rating"]
    color           = r["color"]
    scores          = r["scores"]
    lerbinger_type  = r.get("lerbinger_type",  {})
    lifecycle_stage = r.get("lifecycle_stage", {})
    triage          = r.get("triage",          {})
    checklist       = r.get("checklist",       [])
    filed_date      = r["filed_date"]
    form_code       = r["form_code"]

    # ── Header summary banner ────────────────────────────────────────────────
    stage_icon = lifecycle_stage.get("icon", "⚪")
    stage_name = lifecycle_stage.get("stage", "Unknown")
    urgency    = lifecycle_stage.get("urgency", "")
    triage_color = triage.get("color", "#666")
    triage_level = triage.get("level", "")

    st.markdown(
        f"<div style='background:{color}12;border:2px solid {color};"
        f"padding:20px;border-radius:10px;margin-bottom:20px'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
        f"<div><span style='font-size:32px;font-weight:700;color:{color}'>[{rating}]</span>"
        f" <span style='font-size:18px;font-weight:600'>{ticker} — {form_code} "
        f"({filed_date})</span></div>"
        f"<div style='text-align:right'><div style='font-size:14px;font-weight:600'>"
        f"{stage_icon} {stage_name}</div>"
        f"<div style='font-size:12px;color:{triage_color};font-weight:600'>"
        f"Response Mode: {urgency}</div></div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # ── Section 1: Crisis Lifecycle Stage ───────────────────────────────────
    st.subheader("1️⃣ Crisis Lifecycle Position")
    st.caption("Source: Crandall (2013) + AC820 Week 1 & Week 6 — Before / During / After")
    # 中文: 危机生命周期定位（Week 1+6）

    lc_col1, lc_col2 = st.columns([1, 2])
    with lc_col1:
        phase = lifecycle_stage.get("action_phase", "")
        phase_colors = {"BEFORE": "#2E7D32", "BEFORE / EARLY": "#F9A825",
                        "DURING (Early)": "#E65100", "DURING": "#B71C1C",
                        "AFTER": "#1565C0"}
        ph_color = phase_colors.get(phase, "#666")
        st.markdown(
            f"<div style='background:{ph_color}15;border-left:5px solid {ph_color};"
            f"padding:16px;border-radius:8px;text-align:center'>"
            f"<div style='font-size:36px'>{stage_icon}</div>"
            f"<div style='font-weight:700;color:{ph_color};font-size:15px'>{phase}</div>"
            f"<div style='font-size:12px;color:#666;margin-top:6px'>"
            f"{lifecycle_stage.get('elevated_dimensions',0)} risk dimensions elevated</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with lc_col2:
        st.info(lifecycle_stage.get("description", ""))
        st.caption(
            "**Lifecycle stages:** Preconditions → Trigger Event → Active Crisis → Post-Crisis. "
            "The key insight: crises rarely arrive suddenly — "
            "warning signs are missed due to optimism bias, denial, or overconfidence. "
            "(AC820 Week 3: Why we miss warning signs)"
        )

    st.divider()

    # ── Section 2: Issues Management Triage ─────────────────────────────────
    st.subheader("2️⃣ Issues Management Triage")
    st.caption(
        "Source: Crisis Ready® Flowchart + AC820 Week 1 Issues Management Process\n"
        "Spectrum: Business as Usual → Issues Management → Crisis Management"
    )
    # 中文: Issues Management 分级（Crisis Ready 决策树 + Week 1）

    tr_col1, tr_col2 = st.columns([1, 2])
    with tr_col1:
        escalate_flag = triage.get("escalate_to_cmt", False)
        st.markdown(
            f"<div style='background:{triage_color}15;border-left:5px solid {triage_color};"
            f"padding:14px;border-radius:8px'>"
            f"<div style='font-weight:700;color:{triage_color}'>{triage_level}</div>"
            f"<div style='font-size:12px;margin-top:8px'>"
            f"{'🚨 Escalate to CMT' if escalate_flag else '👁️ Monitor & Respond'}"
            f"</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown("**Crisis Ready® Triage Checklist:**")
        st.markdown(
            f"{'✅' if triage.get('triage_emotionally_charged') else '⬜'} "
            f"Emotionally charged / negative attention\n\n"
            f"{'✅' if not triage.get('triage_controllable') else '⬜'} "
            f"Difficult to control / regain narrative\n\n"
            f"{'✅' if triage.get('triage_long_term_risk') else '⬜'} "
            f"Long-term relationship/reputation risk"
        )
    with tr_col2:
        st.markdown("**Response Guidance:**")
        if triage_color == "#B71C1C":
            st.error(triage.get("response_guidance", ""))
        elif triage_color == "#F9A825":
            st.warning(triage.get("response_guidance", ""))
        else:
            st.success(triage.get("response_guidance", ""))

        st.markdown("**Issues Management Process (AC820 Week 1):**")
        st.markdown(
            "> **Identify → Listen → Investigate → React → Respond → Communicate → Debrief**\n\n"
            "Each step must be documented. Write down everything: "
            "information received (when, how), actions taken, people responsible. "
            "This documentation is critical for the post-crisis debrief and organizational learning."
        )

    st.divider()

    # ── Section 3: Lerbinger Root Cause ─────────────────────────────────────
    st.subheader("3️⃣ Crisis Root Cause — Lerbinger Classification")
    st.caption("Source: Lerbinger (1997) + AC820 Week 2 Vulnerability Framework")
    # 中文: Lerbinger 危机类型与根源漏洞（Week 2）

    if lerbinger_type:
        l_icon  = lerbinger_type.get("icon", "")
        l_name  = lerbinger_type.get("type_name", "")
        l_color_map = {
            "Mismanagement / Misconduct":  "#B71C1C",
            "Stakeholder Confrontation":   "#E65100",
            "Technological Failure":       "#F9A825",
            "Environment & Sustainability":"#1565C0",
        }
        l_color = l_color_map.get(l_name, "#666")

        lb_col1, lb_col2 = st.columns(2)
        with lb_col1:
            st.markdown(
                f"<div style='background:{l_color}15;border-left:5px solid {l_color};"
                f"padding:14px;border-radius:8px'>"
                f"<div style='font-size:28px'>{l_icon}</div>"
                f"<div style='font-weight:700;color:{l_color};font-size:16px'>{l_name}</div>"
                f"<div style='font-size:12px;color:#666;margin-top:4px'>"
                f"Root: {lerbinger_type.get('root_vulnerability','')}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with lb_col2:
            st.markdown("**Exacerbating Factors (AC820 Week 2):**")
            for factor in lerbinger_type.get("exacerbating_factors", []):
                st.markdown(f"⚡ {factor}")
            if lerbinger_type.get("secondary_type"):
                st.caption(f"Secondary type also detected: **{lerbinger_type['secondary_type']}**")

        st.markdown("**Stealing Thunder Recommendation:**")
        st.info(f"⚡ {lerbinger_type.get('stealing_thunder_tip', '')}")

    st.divider()

    # ── Section 4: CMT Activation Checklist ─────────────────────────────────
    st.subheader("4️⃣ Crisis Management Team (CMT) — Activation Checklist")
    st.caption("Source: AC820 Week 3 — CMT as the 'nerve center' of crisis response")
    # 中文: CMT 启动清单（Week 3）

    cmt_col1, cmt_col2 = st.columns(2)
    with cmt_col1:
        st.markdown("**Core CMT Composition (per AC820 Week 3):**")
        cmt_roles = [
            ("👔", "CEO / COO", "Decision authority, public face"),
            ("📣", "PR / Communications Lead", "Message strategy, media relations"),
            ("⚖️", "Legal Counsel", "Regulatory exposure, disclosure compliance"),
            ("👥", "HR",  "Employee communications, internal stakeholders"),
            ("💰", "Finance / Accounting", "Financial impact assessment"),
            ("🏢", "Affected Business Unit Lead", "Operational facts and remediation"),
        ]
        for icon, role, note in cmt_roles:
            st.markdown(f"{icon} **{role}** — *{note}*")

    with cmt_col2:
        st.markdown("**CMT Operational Protocol:**")
        st.markdown(
            "**Command Center:** Designate a 'War Room' with backup location. "
            "Single version of the truth — all updates flow through one source.\n\n"
            "**Cadence:** Hourly status updates during active crisis. "
            "Written records of every decision, action, and communication.\n\n"
            "**Stakeholder Priority:** (1) Life & property first, "
            "(2) Employees, (3) Customers/suppliers, (4) Regulators, (5) Media/public.\n\n"
            "**Dark Site:** Prepare a pre-built communications website "
            "ready to activate within 1 hour of crisis declaration.\n\n"
            "**CMT Pitfalls to Avoid:** Groupthink, decision paralysis, "
            "not understanding symbolic/sacred aspects of the crisis."
        )

    st.divider()

    # ── Section 5: Proactive Action Checklist ───────────────────────────────
    st.subheader("5️⃣ Prioritized Action Checklist")
    st.caption(
        "Source: AC820 Week 3 Proactive Approach — "
        "Vulnerability Audit, Process Improvement, Stealing Thunder, "
        "Leaders Ready, Monitor Radar Screen"
    )
    # 中文: 主动预防行动清单，按优先级排序（Week 3）

    if checklist:
        priority_colors = {
            "🔴 Immediate":     "#B71C1C",
            "🟠 This Quarter":  "#E65100",
            "🟡 This Year":     "#F9A825",
            "🟢 Ongoing":       "#2E7D32",
        }
        for i, item in enumerate(checklist, 1):
            p = item["priority"]
            p_color = priority_colors.get(p, "#666")
            with st.expander(
                f"{p}  |  **{item['action']}**  "
                f"*(by: {item['timing']})*"
            ):
                st.markdown(f"**Rationale:** {item['rationale']}")
                st.caption(f"Framework: {item['framework']}")
    else:
        st.info("Run analysis first to generate the action checklist.")

    st.divider()

    # ── Section 6: Good Ongoing PR Principles ───────────────────────────────
    st.subheader("6️⃣ Foundational PR Principles — Your Crisis Insurance")
    st.caption("Source: PR Page Principle + AC820 Week 1 'Best crisis management = prevention'")
    # 中文: PR 基本原则（Week 1 — 好的 PR 就是最好的危机保险）

    pr_col1, pr_col2 = st.columns(2)
    with pr_col1:
        st.markdown("**PR Page Principle (Crisis Prevention Foundation):**")
        principles = [
            "1. Do the right thing — because it's the right thing to do",
            "2. Operate ethically, transparently, lawfully, per your org's values",
            "3. Put people and the environment first",
            "4. Listen — to weak signals before they become loud crises",
            "5. Fix small issues before they grow",
        ]
        for p in principles:
            st.markdown(f"✅ {p}")

    with pr_col2:
        st.markdown("**Crisis Culture Checklist (AC820 Week 4):**")
        culture_items = [
            "Leadership willing to recognize risks (no optimism bias)",
            "Board actively involved in crisis preparedness",
            "Regular tabletop simulations conducted",
            "Crisis plan tested and updated annually",
            "Culture of psychological safety for raising concerns",
            "No 'this won't happen to us' thinking",
        ]
        for ci in culture_items:
            st.markdown(f"📌 {ci}")

    st.caption(
        "**Remember:** Crisis failure is usually a culture and governance problem, "
        "not a capability problem. (AC820 Week 1)"
    )
