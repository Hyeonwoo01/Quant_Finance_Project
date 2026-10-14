"""yfinance로 실제 시장 데이터(주가·옵션 체인·금리·배당)를 가져온다."""
from datetime import date, datetime

import numpy as np
import pandas as pd
import yfinance as yf

import kr_data
from quant_engine import RiskNeutralPricingEngine


def load_history(ticker, years):
    h = yf.Ticker(ticker).history(period=f"{years}y", auto_adjust=True)
    if h.empty:
        raise ValueError(f"'{ticker}'의 주가 데이터를 찾을 수 없습니다. 티커를 확인하세요 (예: AAPL, SPY, 005930.KS).")
    close = h["Close"].dropna()
    close.index = close.index.tz_localize(None)
    return close


def is_us_ticker(ticker):
    return "." not in ticker or ticker.endswith(".US")


def load_macro(ticker, years=10):
    """방향 예측용 시장 전체 지표. 실패하면 None.

    한국 종목: 기준 지수를 코스피/코스닥으로 바꾸고, 원/달러 환율과 코스피·코스닥 둘 다 추가한다.
    """
    us = is_us_ticker(ticker)
    market = "SPY" if us else ("^KQ11" if ticker.endswith(".KQ") else "^KS11")
    pairs = [("^VIX", "vix"), ("^VIX3M", "vix3m"), ("^TNX", "tnx"), ("^IRX", "irx"), ("HYG", "hyg"),
             ("IEF", "ief"), ("^SKEW", "skew"), ("^VVIX", "vvix"), (market, "market")]
    if not us:
        pairs += [("KRW=X", "usdkrw"), ("^KS11", "kospi"), ("^KQ11", "kosdaq")]
    try:
        d = yf.download(list(dict.fromkeys(s for s, _ in pairs)), period=f"{years}y", auto_adjust=True,
                        progress=False)["Close"]
        d = pd.DataFrame({name: d[sym] for sym, name in pairs})      # 같은 지수가 market/kospi 두 이름을 가질 수 있음
        d.index = d.index.tz_localize(None) if d.index.tz is not None else d.index
        d = d.ffill(limit=5).dropna()
    except Exception:
        return None
    if not us:      # 한국은행 ECOS 금리 (키가 있을 때만; 실패해도 나머지는 그대로 사용)
        try:
            rates = kr_data.ecos_rates(d.index[0])
            if rates is not None:
                d = d.join(rates.reindex(d.index.union(rates.index)).ffill(limit=5).reindex(d.index))
        except Exception:
            pass
    return d if len(d) > 500 else None


def risk_free_rate(ticker="", default=0.04):
    """종목 통화에 맞는 단기 무위험이자율과 출처. 반환: (금리(소수), 출처 이름).

    한국 종목(원화) → 한국은행 ECOS CD(91일), 그 외 → 미국 13주 국채(^IRX). 실패하면 기본값."""
    if ticker and not is_us_ticker(ticker):
        kr = kr_data.kr_short_rate()
        if kr is not None and np.isfinite(kr):
            return kr, "CD 91일 (ECOS)"
        return default, "기본값 (ECOS 조회 실패)"
    try:
        v = yf.Ticker("^IRX").history(period="10d")["Close"].dropna()
        if len(v):
            return float(v.iloc[-1]) / 100.0, "미국 13주 국채 (^IRX)"
    except Exception:
        pass
    return default, "기본값 (^IRX 조회 실패)"


def dividend_yield(ticker, spot):
    """최근 1년 배당금 합 / 현재가 (연속 배당수익률 근사)."""
    try:
        d = yf.Ticker(ticker).dividends
        if d.empty:
            return 0.0
        d.index = d.index.tz_localize(None)
        last_year = d[d.index > pd.Timestamp.now() - pd.Timedelta(days=365)]
        return float(last_year.sum()) / spot
    except Exception:
        return 0.0


def _quote_price(row):
    bid, ask, last = row.get("bid", 0) or 0, row.get("ask", 0) or 0, row.get("lastPrice", 0) or 0
    if bid > 0 and ask > 0:
        return (bid + ask) / 2, "mid"
    return (last, "last") if last > 0 else (np.nan, None)


def atm_implied_vol(ticker, spot, target_cal_days, r, q):
    """목표 기간에 가장 가까운 만기의 ATM 콜/풋 호가에서 IV를 역산한다.

    반환: dict(iv, expiry, cal_days, strike, call_px, put_px, source) 또는 옵션이 없으면 None.
    """
    t = yf.Ticker(ticker)
    try:
        expiries = t.options
    except Exception:
        expiries = ()
    if not expiries:
        return None

    today = date.today()
    days = [(datetime.strptime(e, "%Y-%m-%d").date() - today).days for e in expiries]
    cands = [(abs(d - target_cal_days), e, d) for e, d in zip(expiries, days) if d >= 7]
    if not cands:
        return None
    _, expiry, cal_days = min(cands)

    chain = t.option_chain(expiry)
    calls, puts = chain.calls, chain.puts
    strike = float(calls.loc[(calls["strike"] - spot).abs().idxmin(), "strike"])
    T = cal_days / 365.0

    ivs, prices, sources = [], {}, set()
    for typ, df in (("call", calls), ("put", puts)):
        row = df[df["strike"] == strike]
        if row.empty:
            continue
        row = row.iloc[0].to_dict()
        px, src = _quote_price(row)
        prices[typ] = px
        if src:
            iv = RiskNeutralPricingEngine(spot, strike, T, r, typ, q).implied_volatility(px)
            if np.isfinite(iv):
                ivs.append(iv)
                sources.add(src)
        if not ivs and (row.get("impliedVolatility") or 0) > 0.01:   # 최후 수단: yfinance 제공 IV
            ivs.append(float(row["impliedVolatility"]))
            sources.add("yahoo")
    if not ivs:
        return None
    return {"iv": float(np.mean(ivs)), "expiry": expiry, "cal_days": cal_days, "strike": strike,
            "call_px": prices.get("call", np.nan), "put_px": prices.get("put", np.nan),
            "source": "/".join(sorted(sources))}
