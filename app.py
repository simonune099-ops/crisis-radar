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
         "📊 Results & Charts", "📋 Crisis Playbook",
         "📚 Methods & Glossary"],
        label_visibility="collapsed",
    )
    st.divider()

    # Quick status
    # 中文: 快速显示各模块连接状态
    st.caption("**Module Status**")
    lm_ready  = st.session_state.word_sets    is not None
    ml_ready  = st.session_state.crisis_model is not None
    st.markdown(
        f"{'✅' if lm_ready  else '⬜'} LM Dictionary\n\n"
        f"{'✅' if ml_ready  else '⬜'} Logit Model\n\n"
        f"{'✅' if True       else '⬜'} Yahoo News (auto)"
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

    # ── Media Intelligence — Yahoo Finance News (auto) ──────────────────────
    # No API key required. Uses yfinance (already installed) to fetch recent
    # news headlines for each analyzed company, scored with LM dictionary.
    # 中文: 无需 API Key，使用 yfinance 内置新闻接口自动抓取媒体标题
    #       分析时自动运行，无需任何配置
    st.subheader("📰 Media Intelligence")
    st.success(
        "✅ **Yahoo Finance News** is enabled automatically — no setup required.\n\n"
        "When you run an analysis, the app fetches recent news headlines for the "
        "selected company, scores them with the same LM Dictionary, and compares "
        "media tone against the SEC filing language to detect PR narrative gaps."
    )
    st.caption(
        "Data source: Yahoo Finance News (via yfinance) — "
        "covers major financial and business news outlets. "
        "Headlines are scored using the Loughran-McDonald Finance Dictionary, "
        "same methodology as the SEC filing analysis."
    )

    st.divider()
    st.info("Once LM Dictionary and WRDS are loaded, go to **Analyze Company** in the sidebar.")

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

        # Step 7 — Yahoo Finance News media sentiment analysis (auto, no API key)
        # Method: fetch recent headlines via yfinance, score with LM dictionary,
        # compare against SEC filing tone to detect PR narrative gap.
        # 中文: 自动抓取 Yahoo Finance 新闻标题，用 LM 词典打分，与 SEC 文件语气对比
        #       无需任何 API Key，使用已安装的 yfinance 库
        pr_div      = None
        mt_analysis = None
        progress.progress(72, text="Fetching media headlines (Yahoo Finance)...")

        try:
            ticker_obj = yf.Ticker(ticker)
            raw_news   = ticker_obj.news  # list of dicts: title, publisher, providerPublishTime

            if raw_news:
                import time as _time
                news_rows = []
                for item in raw_news:
                    # Extract content from nested structure (yfinance ≥0.2.x)
                    content = item.get("content", item)   # fallback for older versions
                    title   = (content.get("title") or item.get("title") or "")
                    pub     = (content.get("provider", {}).get("displayName")
                               or item.get("publisher") or "Unknown")
                    ts      = (content.get("pubDate") or
                               item.get("providerPublishTime") or 0)
                    # Convert timestamp to datetime
                    if isinstance(ts, (int, float)) and ts > 1e9:
                        pub_dt = pd.Timestamp(ts, unit="s")
                    else:
                        try:
                            pub_dt = pd.to_datetime(ts)
                        except Exception:
                            pub_dt = pd.NaT
                    if title:
                        news_rows.append({"title": title, "publisher": pub, "date": pub_dt})

                news_df = pd.DataFrame(news_rows)

                if not news_df.empty:
                    # Score all headlines combined with LM dictionary
                    # 中文: 把所有标题合并后用 LM 词典打分
                    all_headlines = " ".join(news_df["title"].fillna("").tolist())
                    media_scores  = scorer.score_text(all_headlines, st.session_state.word_sets)

                    # LM-based sentiment classification per headline
                    # 中文: 对每条标题独立打分，判断正面/负面/中性
                    word_sets_local = st.session_state.word_sets
                    def _classify_headline(text):
                        s = scorer.score_text(text, word_sets_local)
                        if not s:
                            return "Neutral"
                        net = s.get("net_sentiment", 0)
                        if net > 0.01:   return "Positive"
                        elif net < -0.01: return "Negative"
                        else:            return "Neutral"

                    news_df["sentiment"] = news_df["title"].apply(_classify_headline)

                    counts     = news_df["sentiment"].value_counts()
                    pos_count  = int(counts.get("Positive", 0))
                    neg_count  = int(counts.get("Negative", 0))
                    neu_count  = int(counts.get("Neutral",  0))
                    total      = len(news_df)
                    mt_neg_pct = neg_count / total * 100 if total else 0

                    # PR divergence: difference in crisis exposure scores
                    # 中文: 危机暴露分差异 = 媒体语气 vs SEC 文件语气
                    filing_crisis  = scores.get("crisis_score", 0)
                    media_crisis   = media_scores.get("crisis_score", 0)
                    pr_div         = round(abs(filing_crisis - media_crisis), 2)

                    top_publishers = news_df["publisher"].value_counts().head(5)

                    mt_analysis = {
                        "total_articles":  total,
                        "positive_count":  pos_count,
                        "neutral_count":   neu_count,
                        "negative_count":  neg_count,
                        "mt_neg_pct":      round(mt_neg_pct, 1),
                        "top_sources":     top_publishers,
                        "window_df":       news_df,
                        "media_scores":    media_scores,
                        "filing_crisis":   filing_crisis,
                        "media_crisis":    media_crisis,
                        "source_label":    "Yahoo Finance News (recent headlines)",
                    }
        except Exception:
            pass   # media analysis is non-blocking — app works fine without it

        guidance = scorer.get_scct_guidance(scores, pr_div)

        # Step 7b — Extract top trigger sentences from filing text
        # 中文: 提取文件中触发高分的具体句子，回应教授"是什么驱动了评级"的问题
        trigger_sentences = scorer.extract_top_trigger_sentences(
            filing_text, st.session_state.word_sets, n=6
        )

        # Step 7c — Short seller signal detection from news headlines
        # 中文: 从新闻标题检测做空机构信号（Hindenburg、Muddy Waters 等）
        short_seller_signal = {"detected": False, "firms_mentioned": [], "headlines": [], "severity": "🟢 None"}
        if mt_analysis and "window_df" in mt_analysis:
            short_seller_signal = scorer.detect_short_seller_signal(mt_analysis["window_df"])

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
            "mt_analysis":         mt_analysis,
            "trigger_sentences":   trigger_sentences,
            "short_seller_signal": short_seller_signal,
            "filing_text_snippet": filing_text[:3000],
        }

        st.success(
            f"✅ Analysis complete — **{ticker}** {form_code} "
            f"({latest['filed_date']})  |  Rating: **[{rating}]** {rating_label}"
        )
        st.info("➡️ Go to **Results & Charts** in the sidebar to see full output.")

    # ── Peer Comparison ───────────────────────────────────────────────────────
    # Professor feedback: "I'd also add at least 2-3 peer firms to show their ratings"
    # 中文: 同行对比功能——教授建议加入 2-3 家竞争对手评级
    st.divider()
    st.subheader("🏢 Quick Peer Comparison")
    st.caption("Compare crisis ratings across multiple companies side by side.")

    peer_tickers = st.text_input(
        "Enter peer tickers (comma-separated)",
        placeholder="e.g. MSFT, GOOGL, META",
        help="Add 2–3 competitor tickers to benchmark against your analyzed company",
    )

    if st.button("▶ Run Peer Comparison", type="secondary"):
        if not peer_tickers.strip():
            st.warning("Please enter at least one peer ticker.")
        else:
            peers = [t.strip().upper() for t in peer_tickers.split(",") if t.strip()]
            peer_results = []

            # Include main company if already analyzed
            if st.session_state.results:
                main_r = st.session_state.results
                peer_results.append({
                    "ticker":       main_r["ticker"],
                    "filing":       main_r["form_code"],
                    "filed_date":   main_r["filed_date"],
                    "crisis_score": main_r["scores"].get("crisis_score", 0),
                    "rating":       main_r["rating"],
                    "color":        main_r["color"],
                    "uncertainty":  main_r["scores"].get("uncertainty_pct", 0),
                    "litigious":    main_r["scores"].get("litigious_pct", 0),
                    "negative":     main_r["scores"].get("negative_pct", 0),
                })

            ws = st.session_state.word_sets or scorer.load_lm_dictionary()

            for peer in peers[:3]:  # limit to 3 peers to keep it fast
                with st.spinner(f"Fetching {peer}..."):
                    try:
                        p_cik = edgar.get_cik(peer)
                        if not p_cik:
                            st.warning(f"CIK not found for {peer}, skipping.")
                            continue
                        p_filings = edgar.get_filings(p_cik, "10-K", count=1)
                        if p_filings.empty:
                            continue
                        p_latest = p_filings.iloc[0]
                        p_text   = edgar.get_filing_text(p_cik, p_latest["accession"],
                                                          max_chars=60_000)
                        p_scores = scorer.score_text(p_text, ws)
                        p_rating, p_color, _ = scorer.assign_rating(p_scores["crisis_score"])
                        peer_results.append({
                            "ticker":       peer,
                            "filing":       "10-K",
                            "filed_date":   p_latest["filed_date"],
                            "crisis_score": p_scores.get("crisis_score", 0),
                            "rating":       p_rating,
                            "color":        p_color,
                            "uncertainty":  p_scores.get("uncertainty_pct", 0),
                            "litigious":    p_scores.get("litigious_pct", 0),
                            "negative":     p_scores.get("negative_pct", 0),
                        })
                    except Exception as e:
                        st.warning(f"Could not fetch {peer}: {e}")

            if peer_results:
                st.session_state["peer_results"] = peer_results
                # Comparison table
                peer_df = pd.DataFrame(peer_results)
                st.dataframe(
                    peer_df[["ticker","filing","filed_date","crisis_score",
                              "rating","uncertainty","litigious","negative"]]
                    .sort_values("crisis_score", ascending=False)
                    .style.background_gradient(subset=["crisis_score"], cmap="RdYlGn_r"),
                    use_container_width=True,
                )
                # Visual comparison bar chart
                fig_peer = go.Figure()
                for dim, label, clr in [
                    ("crisis_score", "Crisis Score", "#B71C1C"),
                    ("uncertainty",  "Uncertainty",  "#E65100"),
                    ("litigious",    "Litigious",    "#F9A825"),
                    ("negative",     "Negative",     "#1565C0"),
                ]:
                    fig_peer.add_trace(go.Bar(
                        name=label,
                        x=[p["ticker"] for p in peer_results],
                        y=[p[dim] for p in peer_results],
                        marker_color=clr, opacity=0.8,
                    ))
                fig_peer.update_layout(
                    barmode="group", height=350, plot_bgcolor="#FAFAFA",
                    yaxis_title="Score", legend=dict(orientation="h", y=-0.25),
                    margin=dict(l=20, r=20, t=20, b=60),
                )
                st.plotly_chart(fig_peer, use_container_width=True)

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
    mt_analysis         = r.get("mt_analysis")
    trigger_sentences   = r.get("trigger_sentences", [])
    short_seller_signal = r.get("short_seller_signal", {})

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
            st.metric(
                "Media Neg. Headlines",
                f"{mt_analysis['mt_neg_pct']:.1f}%",
                delta=f"{mt_analysis['total_articles']} headlines",
                help="% negative headlines from Yahoo Finance News (LM-scored)",
                # 中文: Yahoo Finance 新闻标题中负面比例（LM 词典打分）
            )
        else:
            st.metric("Media Sentiment", "—",
                      help="Fetched automatically during analysis")

    st.divider()

    # ── What's Driving This Rating? — Source Attribution ─────────────────────
    # Professor feedback: "which sources (10-K or news) are driving the results?"
    # 中文: 信号来源归因面板——回应教授"哪个数据源驱动了评级"的问题
    st.subheader("🔍 What's Driving This Rating?")
    src_c1, src_c2, src_c3 = st.columns(3)

    filing_crisis = scores.get("crisis_score", 0)
    media_crisis  = mt_analysis["media_scores"].get("crisis_score", 0) if mt_analysis and mt_analysis.get("media_scores") else None
    eightk_crisis = r.get("eightk_scores", {}).get("crisis_score", 0) if r.get("eightk_scores") else None

    with src_c1:
        st.markdown(
            f"<div style='border:1px solid {color};border-radius:8px;padding:14px;text-align:center'>"
            f"<div style='font-size:11px;color:#888;font-weight:600'>📄 SEC {form_code} FILING</div>"
            f"<div style='font-size:28px;font-weight:700;color:{color}'>{filing_crisis:.2f}</div>"
            f"<div style='font-size:11px;color:#666'>Crisis Exposure Score</div>"
            f"<div style='font-size:11px;color:#666;margin-top:4px'>Source: EDGAR full text<br>"
            f"{scores.get('total_words',0):,} words analysed</div>"
            f"</div>", unsafe_allow_html=True)

    with src_c2:
        ek_score_str = f"{eightk_crisis:.2f}" if eightk_crisis is not None else "N/A"
        ek_color = "#1565C0" if eightk_crisis is not None else "#999"
        st.markdown(
            f"<div style='border:1px solid {ek_color};border-radius:8px;padding:14px;text-align:center'>"
            f"<div style='font-size:11px;color:#888;font-weight:600'>📢 8-K PRESS RELEASES</div>"
            f"<div style='font-size:28px;font-weight:700;color:{ek_color}'>{ek_score_str}</div>"
            f"<div style='font-size:11px;color:#666'>Crisis Exposure Score</div>"
            f"<div style='font-size:11px;color:#666;margin-top:4px'>Source: SEC EDGAR 8-K filings<br>"
            f"(Company's own press releases)</div>"
            f"</div>", unsafe_allow_html=True)

    with src_c3:
        med_score_str = f"{media_crisis:.2f}" if media_crisis is not None else "N/A"
        med_color = "#2E7D32" if media_crisis is not None else "#999"
        ss = short_seller_signal
        ss_flag = f"⚠️ Short Seller: {ss.get('severity','')}" if ss.get("detected") else "✅ No short seller signal"
        st.markdown(
            f"<div style='border:1px solid {med_color};border-radius:8px;padding:14px;text-align:center'>"
            f"<div style='font-size:11px;color:#888;font-weight:600'>📰 MEDIA / NEWS</div>"
            f"<div style='font-size:28px;font-weight:700;color:{med_color}'>{med_score_str}</div>"
            f"<div style='font-size:11px;color:#666'>Crisis Exposure Score</div>"
            f"<div style='font-size:11px;color:#666;margin-top:4px'>Source: Yahoo Finance News<br>"
            f"{ss_flag}</div>"
            f"</div>", unsafe_allow_html=True)

    # Short Seller Alert (prominent if detected)
    if ss.get("detected"):
        st.error(
            f"🚨 **Short Seller Report Detected** — {ss.get('severity','')}\n\n"
            f"Mentions found: **{', '.join(ss.get('firms_mentioned',[]))}**\n\n"
            + "\n".join(f"• {h}" for h in ss.get("headlines", []))
            + "\n\n*Short seller reports are adversarial external signals. "
            "Companies targeted by activist short sellers typically experience "
            "-8% to -15% stock price impact (Da Fonseca Salvador, 2021). "
            "An immediate 8-K response is recommended.*"
        )

    # Top trigger sentences from filing
    if trigger_sentences:
        with st.expander(f"📝 Top {len(trigger_sentences)} Trigger Sentences from {form_code} Filing"):
            st.caption(
                "These sentences contain the highest concentration of Loughran-McDonald "
                "risk-dimension words and are the primary drivers of the filing's crisis score."
            )
            for i, item in enumerate(trigger_sentences, 1):
                dims_str = " · ".join(item["dimensions"])
                words_str = ", ".join(f"`{w}`" for w in item["flagged_words"])
                st.markdown(
                    f"**{i}.** {item['sentence']}  \n"
                    f"<span style='font-size:11px;color:#888'>Score: {item['crisis_score']:.3f} | "
                    f"Dimensions: {dims_str} | Flag words: {words_str}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("---")

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

    # ── Media Sentiment vs SEC Filing — Yahoo Finance News ───────────────────
    if mt_analysis:
        st.divider()
        st.subheader("📰 Media Sentiment vs SEC Filing Tone")
        st.caption(
            f"{mt_analysis.get('source_label','Yahoo Finance News')} — "
            f"{mt_analysis['total_articles']} headlines scored with LM Dictionary"
        )
        # 中文: 媒体情绪 vs SEC 文件语气对比，数据来源：Yahoo Finance 新闻标题（LM 词典打分）

        # Sentiment breakdown
        mt_c1, mt_c2, mt_c3, mt_c4 = st.columns(4)
        mt_c1.metric("Total Articles",  mt_analysis["total_articles"])
        mt_c2.metric("Positive",  mt_analysis["positive_count"],
                     delta=f"{mt_analysis['positive_count']/mt_analysis['total_articles']*100:.0f}%")
        mt_c3.metric("Neutral",   mt_analysis["neutral_count"])
        mt_c4.metric("Negative",  mt_analysis["negative_count"],
                     delta=f"-{mt_analysis['mt_neg_pct']:.1f}%", delta_color="inverse")

        mt_left, mt_right = st.columns(2)

        # Sentiment donut
        with mt_left:
            if mt_analysis["total_articles"] > 0:
                fig_pie = go.Figure(go.Pie(
                    labels=["Positive", "Neutral", "Negative"],
                    values=[mt_analysis["positive_count"],
                            mt_analysis["neutral_count"],
                            mt_analysis["negative_count"]],
                    marker_colors=["#2E7D32", "#9E9E9E", "#B71C1C"],
                    hole=0.5,
                    textinfo="label+percent",
                ))
                fig_pie.update_layout(
                    height=280, showlegend=False,
                    margin=dict(l=10, r=10, t=30, b=10),
                    title_text="Media Headline Sentiment (LM-scored)",
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        # Filing vs Media LM score comparison bar chart
        with mt_right:
            media_s  = mt_analysis.get("media_scores", {})
            dims     = ["negative", "uncertainty", "litigious", "weak_modal", "constraining"]
            dim_lbls = ["Negative", "Uncertainty", "Litigious", "Weak Modal", "Constraining"]
            filing_v = [scores.get(f"{d}_pct", 0)     for d in dims]
            media_v  = [media_s.get(f"{d}_pct", 0)    for d in dims]

            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(
                name="SEC Filing", x=dim_lbls, y=filing_v,
                marker_color=color, opacity=0.85,
            ))
            fig_cmp.add_trace(go.Bar(
                name="Media Headlines", x=dim_lbls, y=media_v,
                marker_color="#1565C0", opacity=0.85,
            ))
            fig_cmp.update_layout(
                height=280, barmode="group", plot_bgcolor="#FAFAFA",
                yaxis_title="per 1,000 words",
                legend=dict(orientation="h", y=-0.3),
                margin=dict(l=10, r=10, t=30, b=10),
                title_text="LM Scores: Filing vs Media",
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

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

        # Divergence detail
        fc = mt_analysis.get("filing_crisis", 0)
        mc = mt_analysis.get("media_crisis",  0)
        if fc > mc:
            st.info(
                f"📊 **Filing more risk-laden than media** (Filing crisis score: {fc:.2f} vs "
                f"Media: {mc:.2f}). The SEC disclosure contains more cautious language than "
                f"current media coverage — investors reading filings may perceive more risk "
                f"than the market narrative reflects."
            )
        elif mc > fc + 0.5:
            st.warning(
                f"⚠️ **Media more negative than SEC filing** (Media crisis score: {mc:.2f} vs "
                f"Filing: {fc:.2f}). Media narrative is running ahead of formal disclosures — "
                f"apply Stealing Thunder principle: proactive communications needed to "
                f"align public narrative with company's own framing."
            )

        # Recent headlines table
        with st.expander("📋 Recent Headlines (Yahoo Finance)"):
            display_cols = [c for c in ["date", "title", "publisher", "sentiment"]
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

# ════════════════════════════════════════════════════════════════════════════
# PAGE: METHODS & GLOSSARY
# Professor feedback: "add a section on methods" + "explain abbreviations"
# 中文: 方法论与术语解释页面
# ════════════════════════════════════════════════════════════════════════════
elif page == "📚 Methods & Glossary":
    st.title("📚 Methods & Glossary")
    st.markdown("A guide to the data sources, analytical methods, and terminology used in this tool.")

    tab_methods, tab_glossary = st.tabs(["🔬 Methods", "📖 Glossary"])

    # ── METHODS TAB ──────────────────────────────────────────────────────────
    with tab_methods:
        st.subheader("Overview")
        st.markdown(
            "This tool detects PR and financial crisis signals in corporate disclosures "
            "by combining **NLP-based text analysis** with **financial event study methodology**. "
            "Signals are sourced from three independent channels: company filings, "
            "company press releases, and external media — allowing users to identify "
            "divergences between what a company says officially versus what the market perceives."
        )

        st.subheader("1. Data Sources")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**📄 SEC EDGAR (Firm)**")
            st.markdown(
                "- 10-K (Annual Report): full disclosure of operations, risk factors, financials\n"
                "- 10-Q (Quarterly Report): interim update\n"
                "- 8-K (Current Report): material events, press releases, earnings announcements\n"
                "- Source: EDGAR REST API (free, public)\n"
                "- Coverage: All SEC-registered public companies"
            )
        with col2:
            st.markdown("**📰 Yahoo Finance News (External)**")
            st.markdown(
                "- Recent financial news headlines for the analyzed company\n"
                "- Covers major outlets: WSJ, Reuters, Bloomberg, CNBC, etc.\n"
                "- Scored with same LM Dictionary for apple-to-apple comparison\n"
                "- Short seller report detection: scans for Hindenburg, Muddy Waters, "
                "Citron, Grizzly, Iceberg, Viceroy, Blue Orca, etc.\n"
                "- Source: yfinance (free, no API key required)"
            )
        with col3:
            st.markdown("**📈 Yahoo Finance Prices (Market)**")
            st.markdown(
                "- Daily adjusted closing prices for the analyzed company\n"
                "- S&P 500 ETF (SPY) as market benchmark\n"
                "- Used exclusively for event study (CAR) calculation\n"
                "- Source: yfinance (free, no API key required)"
            )

        st.subheader("2. NLP Scoring — Loughran-McDonald Finance Dictionary")
        st.markdown(
            "The **Loughran-McDonald (LM) Master Dictionary** (Loughran & McDonald, 2011, "
            "*Journal of Finance*) is the industry-standard financial text lexicon. "
            "Unlike general-purpose sentiment tools (e.g., VADER, LIWC), it was built "
            "specifically from SEC filings, making it appropriate for 10-K/10-Q analysis."
        )
        lm_df = pd.DataFrame([
            ["Negative",    "Words indicating adverse outcomes", "LOSS, DECLINE, IMPAIR, FAILED", "20%"],
            ["Uncertainty", "Hedge language, unknown outcomes",   "MAY, MIGHT, COULD, UNCERTAIN", "30%"],
            ["Litigious",   "Legal exposure signals",             "LITIGATION, LAWSUIT, ALLEGED", "25%"],
            ["Weak Modal",  "Low-conviction management language", "POSSIBLY, APPEAR, SUGGEST",    "15%"],
            ["Constraining","Regulatory/contractual obligations", "REQUIRED, PROHIBITED, SHALL",  "10%"],
            ["Positive",    "Optimistic language (informational)", "GROWTH, EXCEEDED, CONFIDENT", "—"],
            ["Strong Modal","High-conviction commitments",        "WILL, GUARANTEE, COMMIT",      "—"],
        ], columns=["Dimension", "What it captures", "Example words", "Crisis weight"])
        st.dataframe(lm_df, use_container_width=True, hide_index=True)

        st.markdown(
            "**Crisis Exposure Score** = weighted sum of 5 risk dimensions (per 1,000 words):\n\n"
            "> *Score = 0.30 × Uncertainty + 0.25 × Litigious + 0.20 × Negative "
            "+ 0.15 × Weak Modal + 0.10 × Constraining*\n\n"
            "Weights are derived from Loughran & McDonald (2011) empirical findings on "
            "which dimensions most strongly predict litigation and negative market outcomes."
        )

        st.subheader("3. PR Risk Rating (A–D)")
        rating_df = pd.DataFrame([
            ["A", "< 2.0",    "Low Risk",      "Disclosure language is transparent and consistent. "
                                                "Normal window for proactive IR communications."],
            ["B", "2.0–4.0",  "Moderate Risk", "Selected signals elevated. Issues Management territory. "
                                                "Monitor and consider proactive response."],
            ["C", "4.0–6.5",  "Elevated Risk", "Multiple indicators simultaneously triggered. "
                                                "Trigger Event phase. CMT assessment recommended."],
            ["D", "> 6.5",    "High Risk",     "Strong crisis signals across multiple dimensions. "
                                                "Active crisis protocols should be initiated."],
        ], columns=["Rating", "Score Range", "Level", "Interpretation"])
        st.dataframe(rating_df, use_container_width=True, hide_index=True)
        st.caption(
            "Note: Thresholds are calibrated on S&P 500 10-K filings using the fallback "
            "LM dictionary. Phase 2 will recalibrate using Z-score normalization against "
            "sector-specific benchmarks from WRDS Compustat historical data."
        )

        st.subheader("4. Event Study — Cumulative Abnormal Return (CAR)")
        st.markdown(
            "We implement the **Brown & Warner (1985) market model** event study methodology:\n\n"
            "**Estimation window:** Trading days [−250, −61] before the filing date — "
            "used to estimate the company's normal return relationship with the market.\n\n"
            "**Market model (OLS):** *R_stock = α + β × R_market + ε*  "
            "(estimated using SPY as the market benchmark)\n\n"
            "**Event window:** Trading days [−60, +60] around the filing date.\n\n"
            "**Abnormal Return:** *AR_t = R_stock_t − (α̂ + β̂ × R_market_t)*\n\n"
            "**CAR:** *CAR = Σ AR_t* (cumulative sum over event window)\n\n"
            "A negative CAR after filing indicates the market reacted worse than the "
            "benchmark predicts, consistent with the filing containing adverse information."
        )

        st.subheader("5. Crisis Communication Frameworks")
        st.markdown(
            "**SCCT (Situational Crisis Communication Theory)** — Coombs (2007): "
            "Maps crisis type (Victim / Accidental / Intentional) to optimal communication "
            "strategy (Deny / Diminish / Rebuild). Crisis type is inferred from LM score profile.\n\n"
            "**Lerbinger Crisis Typology** (1997): Classifies root cause into "
            "Mismanagement/Misconduct, Stakeholder Confrontation, Technological Failure, "
            "or Environment & Sustainability based on dominant LM signal dimensions.\n\n"
            "**Crisis Ready® Issue Management** (Bernstein, 2020): Decision-tree triage "
            "for escalation decisions — Business as Usual → Issues Management → Crisis Management.\n\n"
            "**Victim Recovery Cycle**: Guides communication sequencing — "
            "Feelings → Retribution → Healing → Needs."
        )

        st.subheader("6. Short Seller Signal Detection")
        st.markdown(
            "Activist short sellers (Hindenburg Research, Muddy Waters, Citron Research, etc.) "
            "publish adversarial research reports targeting companies they believe are overvalued "
            "or fraudulent. These reports cause average stock price drops of **−8% to −15%** "
            "(Da Fonseca Salvador, 2021), making them a critical external risk signal.\n\n"
            "This tool scans Yahoo Finance headlines for mentions of known short seller firms "
            "and adversarial language patterns. Detection of a short seller signal triggers "
            "an immediate **CMT escalation alert** and **Stealing Thunder** recommendation "
            "(proactive company response before media amplification)."
        )

    # ── GLOSSARY TAB ─────────────────────────────────────────────────────────
    with tab_glossary:
        st.subheader("Abbreviations & Terminology")
        st.caption("Professor feedback: 'explain the abbreviations so it is straightforward for the end user'")

        glossary = pd.DataFrame([
            # Financial
            ["10-K",  "Annual Report",              "Comprehensive annual financial and operational disclosure filed with the SEC by public companies"],
            ["10-Q",  "Quarterly Report",            "Interim financial report filed three times a year (Q1, Q2, Q3); less comprehensive than 10-K"],
            ["8-K",   "Current Report",              "Immediate disclosure of material events: earnings, mergers, leadership changes, legal proceedings"],
            ["SEC",   "Securities and Exchange Commission", "U.S. federal agency that regulates public company disclosures and enforces securities laws"],
            ["EDGAR", "Electronic Data Gathering, Analysis, and Retrieval", "SEC's public database of all company filings; the primary data source for this tool"],
            ["S&P 500", "Standard & Poor's 500",    "Index of 500 large U.S. public companies; used as the universe for company selection in this tool"],
            ["CAR",   "Cumulative Abnormal Return",  "Measure of how much a stock's return deviates from the market benchmark during an event window"],
            ["SPY",   "SPDR S&P 500 ETF",            "Exchange-traded fund tracking the S&P 500 index; used as the market benchmark in event study"],
            ["IPO",   "Initial Public Offering",     "First public sale of a company's stock; point at which SEC disclosure obligations begin"],
            # NLP
            ["NLP",   "Natural Language Processing",  "Branch of AI that analyzes and extracts meaning from text; used here to score SEC filings"],
            ["LM",    "Loughran-McDonald",            "Authors of the Finance Dictionary (2011, Journal of Finance) used for scoring financial text"],
            ["LM Dictionary", "Loughran-McDonald Finance Dictionary", "Finance-specific lexicon of ~86,000 words categorized into 7 sentiment dimensions; standard in academic finance NLP research"],
            # Crisis/PR
            ["PR",    "Public Relations",             "Management of communication between an organization and its stakeholders (investors, media, public)"],
            ["IR",    "Investor Relations",           "Subset of PR focused specifically on communication with shareholders and the investment community"],
            ["SCCT",  "Situational Crisis Communication Theory", "Academic framework by Coombs (2007) that maps crisis type to recommended communication strategy"],
            ["CMT",   "Crisis Management Team",       "Cross-functional team (CEO, PR, Legal, Finance, HR) responsible for coordinating crisis response"],
            ["LP",    "Limited Partner",              "Passive investor in a fund (e.g., pension fund, endowment); provides capital but has limited liability and control"],
            ["GP",    "General Partner",              "Active fund manager who makes investment decisions and has unlimited liability; communicates with LPs"],
            ["ESG",   "Environmental, Social, Governance", "Framework for evaluating non-financial corporate risk and performance; increasingly important in investor communications"],
            ["CAR",   "Crisis-Adjusted Return",       "In this context: Cumulative Abnormal Return around filing date; measures market reaction to disclosure"],
            # Short Sellers
            ["Short Seller", "Activist Short Seller", "Investor who profits when a stock price declines; activist short sellers publish public reports alleging fraud or overvaluation"],
            ["Short and Distort", "Market Manipulation Tactic", "Illegal practice of short-selling then spreading false negative information to drive down a stock price"],
            ["Stealing Thunder", "Proactive Disclosure Strategy", "Voluntarily disclosing negative information before it is discovered externally; reduces reputational damage by ~35% (Arpan & Roskos-Ewoldsen, 2005)"],
        ], columns=["Term", "Full Name", "Definition"])

        # Search filter
        search = st.text_input("🔍 Search glossary", placeholder="Type a term...")
        if search:
            glossary = glossary[
                glossary["Term"].str.lower().str.contains(search.lower()) |
                glossary["Full Name"].str.lower().str.contains(search.lower()) |
                glossary["Definition"].str.lower().str.contains(search.lower())
            ]

        st.dataframe(glossary, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("📚 Key References")
        st.markdown(
            "- Loughran, T. & McDonald, B. (2011). *When is a liability not a liability? "
            "Textual analysis, dictionaries, and 10-Ks.* Journal of Finance, 66(1), 35–65.\n"
            "- Coombs, W.T. (2007). *Protecting organization reputations during a crisis: "
            "The development and application of situational crisis communication theory.* "
            "Corporate Reputation Review, 10(3), 163–176.\n"
            "- Brown, S.J. & Warner, J.B. (1985). *Using daily stock returns: The case of "
            "event studies.* Journal of Financial Economics, 14(1), 3–31.\n"
            "- Lerbinger, O. (1997). *The Crisis Manager: Facing Risk and Responsibility.* "
            "Lawrence Erlbaum Associates.\n"
            "- Da Fonseca Salvador, R. (2021). *Activist short sellers: What are their "
            "performances?* Louvain School of Management.\n"
            "- Arpan, L.M. & Roskos-Ewoldsen, D.R. (2005). *Stealing thunder: Analysis of "
            "the effects of proactive disclosure of crisis information.* "
            "Public Relations Review, 31(3), 425–433.\n"
            "- Stellmach, W., Hussein, S.M. & Sandoloski, S. (2024). *Short and Distort: "
            "Understanding and Responding to Short-Seller Reports.* Willkie Farr & Gallagher LLP."
        )
