# edgar.py
# SEC EDGAR data retrieval module
# Corporate Disclosure Crisis Radar — AC820/BA870
# 中文: SEC EDGAR 数据抓取模块

import requests
import pandas as pd
import re
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "BU-FinAnalytics student@bu.edu"}


def get_cik(ticker: str) -> str | None:
    """
    Convert a stock ticker to its SEC Central Index Key (CIK).
    Returns a zero-padded 10-digit string, or None if not found.
    中文: 把股票代码转成 SEC 公司识别码 CIK（10位数字）
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    for entry in r.json().values():
        if entry["ticker"] == ticker.upper():
            return str(entry["cik_str"]).zfill(10)
    return None


def get_filings(cik: str, form_type: str = "10-K", count: int = 8) -> pd.DataFrame:
    """
    Return the most recent filings of a given form type for a company.
    中文: 返回该公司最近 count 份指定类型（10-K 或 10-Q）的文件列表
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    f = r.json()["filings"]["recent"]
    df = pd.DataFrame({
        "form":       f["form"],
        "filed_date": f["filingDate"],
        "accession":  f["accessionNumber"],
    })
    return df[df["form"] == form_type].head(count).reset_index(drop=True)


def get_filing_text(cik: str, accession: str, max_chars: int = 100_000) -> str:
    """
    Fetch the full plain text of an SEC filing.
    Strategy: (1) locate the primary HTML document from the filing index,
    (2) parse with lxml which handles malformed HTML entities correctly,
    (3) fall back to regex stripping if all else fails.
    Returns up to max_chars characters after stripping HTML.

    中文: 抓取 SEC 文件全文
    策略：(1)先从索引找主文档 (2)用 lxml 解析避免 html.parser 的 hex 实体报错
          (3)最后兜底用正则去标签
    Phase 2: 改进为精准提取 MD&A 和 Risk Factors 章节
    """
    cik_int   = int(cik)
    acc_clean = accession.replace("-", "")

    # Disable gzip compression so we always get plain text/HTML bytes
    # 中文: 禁用 gzip 压缩，确保收到可读文本
    headers = {**HEADERS, "Accept-Encoding": "identity"}

    # ── Step 1: find the primary document via the filing index ──────────────
    # 中文: 从索引 JSON 定位主文档（10-K 或 10-Q 的 .htm 文件）
    primary_url = None
    try:
        idx_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_int}/{acc_clean}/{accession}-index.json"
        )
        idx_r = requests.get(idx_url, headers=headers, timeout=15)
        if idx_r.status_code == 200:
            for doc in idx_r.json().get("documents", []):
                doc_type = doc.get("type", "")
                doc_name = doc.get("name", "")
                if doc_type in ("10-K", "10-Q", "10-K/A", "10-Q/A") and doc_name:
                    primary_url = (
                        f"https://www.sec.gov/Archives/edgar/data/"
                        f"{cik_int}/{acc_clean}/{doc_name}"
                    )
                    break
    except Exception:
        pass  # fall through to the wrapper .txt below

    # ── Step 2: fetch & parse with lxml ─────────────────────────────────────
    # lxml tolerates malformed HTML entities (e.g. &#9CF5; hex refs) that
    # crash Python's built-in html.parser with "invalid literal for int()"
    # 中文: lxml 能正确处理 EDGAR 文件里的十六进制 HTML 实体（html.parser 会崩溃）
    fetch_url = primary_url or (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int}/{acc_clean}/{accession}.txt"
    )

    try:
        r = requests.get(fetch_url, headers=headers, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "lxml")   # use bytes so lxml picks encoding
        text = soup.get_text(separator=" ")
    except Exception:
        # ── Step 3: last-resort regex fallback ──────────────────────────────
        # 中文: 兜底方案：用正则直接删除所有 HTML 标签
        try:
            raw = r.text
        except Exception:
            raw = ""
        text = re.sub(r"<[^>]+>", " ", raw)

    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def get_sp500_tickers() -> pd.DataFrame:
    """
    Fetch the current S&P 500 constituent list from Wikipedia.
    Returns a DataFrame with columns: ticker, company, sector.
    中文: 从 Wikipedia 获取 S&P 500 成分股列表
    """
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )
        df = tables[0][["Symbol", "Security", "GICS Sector"]].copy()
        df.columns = ["ticker", "company", "sector"]
        return df.sort_values("ticker").reset_index(drop=True)
    except Exception:
        # Fallback list if Wikipedia is unavailable
        # 中文: Wikipedia 不可用时的备用列表
        return pd.DataFrame({
            "ticker":  ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
                        "JPM", "GS", "MS", "BAC", "BLK"],
            "company": ["Apple", "Microsoft", "Alphabet", "Amazon", "Meta",
                        "JPMorgan", "Goldman Sachs", "Morgan Stanley",
                        "Bank of America", "BlackRock"],
            "sector":  ["Technology"] * 5 + ["Financials"] * 5,
        })
