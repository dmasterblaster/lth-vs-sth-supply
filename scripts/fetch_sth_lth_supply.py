import os
import io
import re
import json
from pathlib import Path

import pandas as pd
import requests

# Try v1 first, fallback to non-v1 if needed.
SUPPLY_URLS = [
    "https://api.bitcoinmagazinepro.com/v1/metrics/sth-vs-lth-supply",
    "https://api.bitcoinmagazinepro.com/metrics/sth-vs-lth-supply",
]

# If the supply endpoint does not include BTC price, we fetch it separately and merge by date.
PRICE_URLS = [
    "https://api.bitcoinmagazinepro.com/v1/metrics/price",
    "https://api.bitcoinmagazinepro.com/metrics/price",
]

OUT_PATH = Path("data/sth-vs-lth-supply.json")


def _strip_wrapping_quotes(text: str) -> str:
    s = text.strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1]
    return s


def _unescape_newlines(text: str) -> str:
    # BMP sometimes returns a quoted string where "\n" is literal
    return text.replace("\\n", "\n")


def _download_csv(urls, headers, timeout=30) -> str:
    last_err = None
    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            print("BMP API URL:", url)
            print("BMP API status code:", resp.status_code)
            resp.raise_for_status()
            raw = resp.text
            if not raw.strip():
                raise RuntimeError("Empty response body")
            raw = _strip_wrapping_quotes(raw)
            raw = _unescape_newlines(raw)
            return raw
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"All endpoints failed. Last error: {last_err}")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize column names to simpler lowercase tokens
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Some CSVs have a blank first column that is just an index
    if df.columns.size > 0 and (df.columns[0] == "" or df.columns[0].lower() in ["unnamed: 0", "index"]):
        df = df.drop(columns=[df.columns[0]])

    return df


def _pick_col(cols, candidates):
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _to_num(x):
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.strip()
            if x == "" or x.lower() in ["nan", "null", "none"]:
                return None
            # Remove commas
            x = x.replace(",", "")
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def _parse_supply_csv(csv_text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(csv_text))
    df = _normalize_columns(df)

    if df.empty:
        raise RuntimeError("Parsed empty DataFrame from supply CSV")

    cols = list(df.columns)

    date_col = _pick_col(cols, ["Date", "date", "time", "timestamp"])
    price_col = _pick_col(cols, ["Price", "price", "btc_price", "btcPrice", "price_usd", "usd_price"])

    # Supply columns can vary. We try a bunch of likely names.
    sth_col = _pick_col(cols, [
        "sth", "sth_supply", "sth supply", "short_term_supply", "short term supply",
        "sth_coins", "sth_coins_m", "sth_supply_m", "Short-Term Holders", "STH"
    ])
    lth_col = _pick_col(cols, [
        "lth", "lth_supply", "lth supply", "long_term_supply", "long term supply",
        "lth_coins", "lth_coins_m", "lth_supply_m", "Long-Term Holders", "LTH"
    ])

    if date_col is None:
        raise RuntimeError(f"Could not find a date column. Columns: {cols}")

    # If we did not find sth/lth by name, try heuristic: look for two numeric columns besides date/price.
    if sth_col is None or lth_col is None:
        numeric_candidates = []
        for c in cols:
            if c == date_col:
                continue
            if price_col is not None and c == price_col:
                continue
            # Heuristic: if column has lots of numeric values, consider it
            sample = df[c].head(50).tolist()
            nums = [_to_num(v) for v in sample]
            numeric_ratio = sum(v is not None for v in nums) / max(1, len(nums))
            if numeric_ratio > 0.7:
                numeric_candidates.append(c)

        # Choose first two numeric columns as STH/LTH if we can
        if len(numeric_candidates) >= 2:
            if sth_col is None:
                sth_col = numeric_candidates[0]
            if lth_col is None:
                lth_col = numeric_candidates[1]

    if sth_col is None or lth_col is None:
        raise RuntimeError(
            "Could not identify STH and LTH columns.\n"
            f"Columns: {cols}\n"
            "Tip: open the raw CSV response in Actions logs and tell me the exact column headers."
        )

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    out["sth"] = df[sth_col].map(_to_num)
    out["lth"] = df[lth_col].map(_to_num)

    if price_col is not None:
        out["price"] = df[price_col].map(_to_num)
    else:
        out["price"] = None

    out = out.dropna(subset=["date"])
    out = out.sort_values("date").reset_index(drop=True)

    # Drop rows where both supplies are missing
    out = out[~(out["sth"].isna() & out["lth"].isna())].reset_index(drop=True)

    return out


def _parse_price_csv(csv_text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(csv_text))
    df = _normalize_columns(df)
    if df.empty:
        raise RuntimeError("Parsed empty DataFrame from price CSV")

    cols = list(df.columns)
    date_col = _pick_col(cols, ["Date", "date", "time", "timestamp"])
    price_col = _pick_col(cols, ["Price", "price", "btc_price", "btcPrice", "price_usd", "usd_price", "value"])

    if date_col is None or price_col is None:
        raise RuntimeError(f"Could not parse price CSV. Columns: {cols}")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    out["price"] = df[price_col].map(_to_num)
    out = out.dropna(subset=["date"])
    out = out.sort_values("date").reset_index(drop=True)
    return out


def main():
    api_key = os.environ.get("BMP_API_KEY")
    if not api_key:
        raise RuntimeError("Missing BMP_API_KEY env var. Add it to GitHub Secrets as BMP_API_KEY.")

    headers = {"Authorization": f"Bearer {api_key}"}

    supply_csv = _download_csv(SUPPLY_URLS, headers=headers)
    supply_df = _parse_supply_csv(supply_csv)

    # If price is missing entirely, fetch price series and merge
    if supply_df["price"].isna().all():
        print("Supply endpoint has no price column, fetching BTC price separately...")
        price_csv = _download_csv(PRICE_URLS, headers=headers)
        price_df = _parse_price_csv(price_csv)

        merged = supply_df.merge(price_df, on="date", how="left", suffixes=("", "_p"))
        merged["price"] = merged["price_p"]
        merged = merged.drop(columns=["price_p"])
        supply_df = merged

    records = []
    for _, r in supply_df.iterrows():
        records.append({
            "date": r["date"],
            "lth": None if pd.isna(r["lth"]) else float(r["lth"]),
            "sth": None if pd.isna(r["sth"]) else float(r["sth"]),
            "price": None if pd.isna(r["price"]) else float(r["price"]),
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")

    print(f"Wrote {len(records)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
