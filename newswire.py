# newswire.py
# Access Newswire press release retrieval and tone comparison
# Corporate Disclosure Crisis Radar — AC820/BA870
# 中文: Access Newswire 新闻稿抓取与语气对比模块

import requests
import pandas as pd
from datetime import datetime, timedelta

NEWSWIRE_BASE = "https://api.accesswire.com/v1"


def test_connection(api_key: str) -> bool:
    """
    Verify that the Access Newswire API key is valid.
    中文: 验证 API Key 是否有效
    """
    try:
        r = requests.get(
            f"{NEWSWIRE_BASE}/releases",
            headers={"Authorization": f"Bearer {api_key}"},
            params={"limit": 1},
            timeout=10,
        )
        return r.status_code == 200
    except Exception:
        return False


def fetch_releases(
    ticker: str,
    filed_date: str,
    api_key: str,
    window_days: int = 30,
) -> pd.DataFrame:
    """
    Fetch press releases for a company within ±window_days of the filing date.
    Returns a DataFrame with columns: date, title, content.
    中文: 获取文件提交日期前后 window_days 天内的新闻稿
    """
    filed_dt   = datetime.strptime(filed_date, "%Y-%m-%d")
    start_date = (filed_dt - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end_date   = (filed_dt + timedelta(days=window_days)).strftime("%Y-%m-%d")

    try:
        r = requests.get(
            f"{NEWSWIRE_BASE}/releases",
            headers={"Authorization": f"Bearer {api_key}"},
            params={
                "search":     ticker,
                "start_date": start_date,
                "end_date":   end_date,
                "limit":      20,
            },
            timeout=15,
        )
        if r.status_code != 200:
            print(f"[newswire] API returned {r.status_code}")
            return pd.DataFrame()

        releases = r.json().get("data", [])
        return pd.DataFrame([
            {
                "date":    rel.get("publish_date", ""),
                "title":   rel.get("headline", ""),
                "content": rel.get("body", ""),
            }
            for rel in releases
        ])

    except Exception as e:
        print(f"[newswire] Fetch error: {e}")
        return pd.DataFrame()


def compute_divergence(filing_scores: dict, pr_scores: dict) -> dict:
    """
    Compute tone divergence between an SEC filing and press releases.
    A high divergence score signals messaging inconsistency — an independent PR risk.

    Dimensions compared: positive, negative, uncertainty (per 1,000 words)
    Divergence > 4.0 is considered significant.

    中文: 计算 SEC 文件和新闻稿的语气差距；差距 > 4.0 视为显著不一致
    """
    dims = ["positive_pct", "negative_pct", "uncertainty_pct"]
    result = {}
    for d in dims:
        result[d.replace("_pct", "")] = round(
            abs(filing_scores.get(d, 0) - pr_scores.get(d, 0)), 3
        )
    result["total"] = round(sum(result.values()), 3)
    return result
