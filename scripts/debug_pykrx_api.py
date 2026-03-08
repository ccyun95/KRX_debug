#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd
import pykrx
from pykrx import stock

REQ_COLS = [
    "일자", "시가", "고가", "저가", "종가", "거래량", "등락률",
    "기관 합계", "기타법인", "개인", "외국인 합계", "전체",
    "공매도", "공매도비중", "공매도잔고", "공매도잔고비중",
]


def empty_with_cols(cols: List[str]) -> pd.DataFrame:
    data = {}
    for c in cols:
        data[c] = pd.Series(dtype="object") if c == "일자" else pd.Series(dtype="float64")
    return pd.DataFrame(data)


def normalize_date_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return empty_with_cols(["일자"])
    df = df.copy()
    if df.index.name is None:
        df.index.name = "일자"
    idx = pd.to_datetime(df.index, errors="coerce")
    df.index = idx
    df.reset_index(inplace=True)
    df.rename(columns={df.columns[0]: "일자"}, inplace=True)
    df["일자"] = pd.to_datetime(df["일자"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def _to_float_clean(s: Any) -> float:
    try:
        if pd.isna(s):
            return 0.0
        x = str(s).strip()
        if x.endswith("%"):
            x = x[:-1]
        x = x.replace(",", "").replace(" ", "")
        return float(x)
    except Exception:
        return 0.0


def _pick_first_col(cols: List[str], candidates: List[str]):
    for key in candidates:
        for c in cols:
            if key in str(c):
                return c
    return None


def rename_investor_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "일자" not in df.columns:
        return empty_with_cols(["일자", "기관 합계", "기타법인", "개인", "외국인 합계", "전체"])
    mapping = {
        "기관합계": "기관 합계", "외국인합계": "외국인 합계",
        "기관 합계": "기관 합계", "외국인 합계": "외국인 합계",
        "개인": "개인", "기타법인": "기타법인", "전체": "전체",
    }
    df = df.rename(columns={c: mapping.get(c, c) for c in df.columns})
    for need in ["기관 합계", "기타법인", "개인", "외국인 합계", "전체"]:
        if need not in df.columns:
            df[need] = 0
    return df[["일자", "기관 합계", "기타법인", "개인", "외국인 합계", "전체"]]


def rename_short_cols(df: pd.DataFrame, is_balance: bool = False) -> pd.DataFrame:
    if df is None or df.empty or "일자" not in df.columns:
        base = ["공매도잔고", "공매도잔고비중"] if is_balance else ["공매도", "공매도비중"]
        return empty_with_cols(["일자"] + base)

    dfc = df.copy()

    if is_balance:
        amt_col = _pick_first_col(
            list(dfc.columns),
            ["공매도잔고", "잔고수량", "잔고금액", "잔고", "BAL_QTY", "BAL_AMT"],
        )
        rto_col = _pick_first_col(
            list(dfc.columns),
            ["공매도잔고비중", "잔고비중", "BAL_RTO", "비중"],
        )
        dfc["공매도잔고"] = dfc[amt_col].apply(_to_float_clean) if amt_col else 0.0
        dfc["공매도잔고비중"] = dfc[rto_col].apply(_to_float_clean) if rto_col else 0.0
        out = dfc[["일자", "공매도잔고", "공매도잔고비중"]].copy()
    else:
        amt_col = _pick_first_col(
            list(dfc.columns),
            ["공매도거래량", "공매도", "거래량", "SV_QTY", "SV_AMT"],
        )
        rto_col = _pick_first_col(
            list(dfc.columns),
            ["공매도비중", "비중", "SV_RTO"],
        )
        dfc["공매도"] = dfc[amt_col].apply(_to_float_clean) if amt_col else 0.0
        dfc["공매도비중"] = dfc[rto_col].apply(_to_float_clean) if rto_col else 0.0
        out = dfc[["일자", "공매도", "공매도비중"]].copy()

    out["일자"] = pd.to_datetime(out["일자"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


def ensure_all_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in REQ_COLS:
        if col not in df.columns:
            df[col] = 0
    return df[REQ_COLS]


def preview_records(df: pd.DataFrame, rows: int = 5) -> List[Dict[str, Any]]:
    if df is None:
        return []
    try:
        return json.loads(df.head(rows).reset_index(drop=True).to_json(force_ascii=False, orient="records", date_format="iso"))
    except Exception:
        try:
            return df.head(rows).astype(str).reset_index(drop=True).to_dict(orient="records")
        except Exception:
            return []


def dataframe_info(df: pd.DataFrame, rows: int = 5) -> Dict[str, Any]:
    if df is None:
        return {
            "is_none": True,
            "is_empty": True,
            "shape": None,
            "columns": [],
            "index_name": None,
            "preview": [],
        }
    return {
        "is_none": False,
        "is_empty": bool(df.empty),
        "shape": list(df.shape),
        "columns": [str(c) for c in df.columns],
        "index_name": str(df.index.name) if df.index.name is not None else None,
        "preview": preview_records(df, rows=rows),
    }


def print_block(title: str, payload: Dict[str, Any]):
    print("\n" + "=" * 110)
    print(title)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def call_and_capture(name: str, fn, *args, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    try:
        df = fn(*args, **kwargs)
        return df, {"status": "OK", "call": name, "args": list(args), "kwargs": kwargs, "info": dataframe_info(df)}
    except Exception as e:
        return None, {"status": "ERROR", "call": name, "args": list(args), "kwargs": kwargs, "error": repr(e)}


def summarize_key_columns(df: pd.DataFrame) -> Dict[str, Any]:
    cols = [
        "기관 합계", "기타법인", "개인", "외국인 합계", "전체",
        "공매도", "공매도비중", "공매도잔고", "공매도잔고비중",
    ]
    out: Dict[str, Any] = {}
    if df is None or df.empty:
        return {c: {"nonzero_rows": 0, "all_zero": True} for c in cols}

    for c in cols:
        series = pd.to_numeric(df[c], errors="coerce").fillna(0)
        out[c] = {
            "nonzero_rows": int((series != 0).sum()),
            "all_zero": bool((series == 0).all()),
            "sample_top5": series.head(5).tolist(),
        }
    return out


def build_fetch_block_like(ticker: str, start: str, end: str, investor_on: str) -> Dict[str, Any]:
    raw_results: Dict[str, Any] = {}

    ohlcv_raw, raw_results["ohlcv"] = call_and_capture(
        "get_market_ohlcv", stock.get_market_ohlcv, start, end, ticker
    )
    inv_raw_default, raw_results["investor_default"] = call_and_capture(
        "get_market_trading_volume_by_date(default)", stock.get_market_trading_volume_by_date, start, end, ticker
    )
    inv_raw_selected, raw_results["investor_selected_on"] = call_and_capture(
        f"get_market_trading_volume_by_date(on={investor_on})", stock.get_market_trading_volume_by_date, start, end, ticker, on=investor_on
    )
    sv_raw, raw_results["short_volume"] = call_and_capture(
        "get_shorting_volume_by_date", stock.get_shorting_volume_by_date, start, end, ticker
    )
    sb_raw, raw_results["short_balance"] = call_and_capture(
        "get_shorting_balance_by_date", stock.get_shorting_balance_by_date, start, end, ticker
    )

    df1 = normalize_date_index(ohlcv_raw)
    df2_default = rename_investor_cols(normalize_date_index(inv_raw_default))
    df2_selected = rename_investor_cols(normalize_date_index(inv_raw_selected))
    df3 = rename_short_cols(normalize_date_index(sv_raw), is_balance=False)
    df4 = rename_short_cols(normalize_date_index(sb_raw), is_balance=True)

    normalized = {
        "ohlcv_norm": dataframe_info(df1),
        "investor_default_norm": dataframe_info(df2_default),
        "investor_selected_norm": dataframe_info(df2_selected),
        "short_volume_norm": dataframe_info(df3),
        "short_balance_norm": dataframe_info(df4),
    }

    merged_default = df1.merge(df2_default, on="일자", how="left").merge(df3, on="일자", how="left").merge(df4, on="일자", how="left")
    merged_default = ensure_all_cols(merged_default)
    for c in [c for c in merged_default.columns if c != "일자"]:
        merged_default[c] = pd.to_numeric(merged_default[c], errors="coerce").fillna(0)
    merged_default = merged_default.sort_values("일자", ascending=False).reset_index(drop=True)

    merged_selected = df1.merge(df2_selected, on="일자", how="left").merge(df3, on="일자", how="left").merge(df4, on="일자", how="left")
    merged_selected = ensure_all_cols(merged_selected)
    for c in [c for c in merged_selected.columns if c != "일자"]:
        merged_selected[c] = pd.to_numeric(merged_selected[c], errors="coerce").fillna(0)
    merged_selected = merged_selected.sort_values("일자", ascending=False).reset_index(drop=True)

    merged_payload = {
        "final_default_like_fetch_block": {
            "info": dataframe_info(merged_default, rows=10),
            "key_column_summary": summarize_key_columns(merged_default),
        },
        "final_selected_like_fetch_block": {
            "investor_on": investor_on,
            "info": dataframe_info(merged_selected, rows=10),
            "key_column_summary": summarize_key_columns(merged_selected),
        },
    }

    return {
        "raw_results": raw_results,
        "normalized_results": normalized,
        "merged_results": merged_payload,
    }


def main():
    parser = argparse.ArgumentParser(description="fetch_block() 유사 흐름까지 포함한 standalone pykrx 진단")
    parser.add_argument("--ticker", default="005930", help="종목코드 6자리")
    parser.add_argument("--start", default="20260219", help="시작일 YYYYMMDD")
    parser.add_argument("--end", default="20260306", help="종료일 YYYYMMDD")
    parser.add_argument("--investor-on", default="순매수", choices=["순매수", "매도", "매수"], help="추가 비교용 투자자 거래량 기준")
    args = parser.parse_args()

    env = {
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "pykrx_version": getattr(pykrx, "__version__", "unknown"),
        "ticker": args.ticker,
        "start": args.start,
        "end": args.end,
        "investor_on": args.investor_on,
    }
    print_block("[ENV]", env)

    result = build_fetch_block_like(args.ticker, args.start, args.end, args.investor_on)
    print_block("[RAW API RESULTS]", result["raw_results"])
    print_block("[NORMALIZED RESULTS]", result["normalized_results"])
    print_block("[FETCH_BLOCK-LIKE MERGED RESULTS]", result["merged_results"])

    print("\n" + "#" * 110)
    print("[SUMMARY_JSON]")
    print(json.dumps({"env": env, **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
