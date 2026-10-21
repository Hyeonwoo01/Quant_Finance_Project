"""변동성 예측 & 트레이딩 신호 프로그램 (Streamlit).

실행:  run.bat  또는  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import alt_data as ad
import direction as dr
import kr_data
import market_data as md
import regime as rg
import ui
import universe as uv
from quant_engine import (GarchVolForecaster, GammaScalpingEngine, VolatilityTradingLogic, add_combined_forecast,
                          forecast_errors, max_drawdown, short_only_split, simulate_gbm_paths, summarize_vol_backtest,
                          var_es, vol_backtest_robustness, vol_signal_backtest)

st.set_page_config(page_title="변동성 예측기", page_icon="📈", layout="wide")
ui.apply()


# ---------------- 데이터 (캐시) ----------------
@st.cache_data(ttl=3600, show_spinner=False)
def find_ticker(query):
    """종목명 또는 티커 → [(티커, 표시 이름)]. 한국 종목 목록은 하루 캐시, 영문 이름은 Yahoo 검색."""
    return md.search_ticker(query)


@st.cache_data(ttl=900, show_spinner=False)
def get_market(ticker, years, horizon):
    close = md.load_history(ticker, years)
    spot = float(close.iloc[-1])
    r, r_src = md.risk_free_rate(ticker)     # 한국 종목은 원화 금리(CD 91일)
    q = md.dividend_yield(ticker, spot)
    opt = md.atm_implied_vol(ticker, spot, round(horizon * 365 / 252), r, q)
    return close, spot, r, q, opt, r_src


@st.cache_data(ttl=900, show_spinner=False)
def get_direction(ticker, h, cost_bps):
    """방향 모델은 표본이 많이 필요해서 학습 기간 설정과 별개로 최대 10년치를 쓴다."""
    close = md.load_history(ticker, 10)
    macro = md.load_macro(ticker)
    alt, alt_status = ad.build_alt_features(ticker, close.index)
    # 거시 지표 시차는 티커로 자동 결정 (한국 종목이면 미국 지표를 하루 늦춤)
    wf, _ = dr.walk_forward(close, h, macro, alt=alt, ticker=ticker)
    if wf.empty:
        raise ValueError("과거 데이터가 너무 짧습니다 (최소 약 3년 필요)")
    ev = dr.evaluate(wf, h, np.log(close).diff(), cost_bps)
    p_up, pick, now_all, contrib, asof = dr.predict_now(close, h, wf, macro, alt=alt, ticker=ticker)
    alt_status["시장 옵션 심리(SKEW·VVIX)"] = "사용" if macro is not None and "skew" in macro else "데이터 없음"
    alt_status["거시(VIX·금리·신용)"] = "사용" if macro is not None else "데이터 없음"
    if not md.is_us_ticker(ticker):
        alt_status["환율·코스닥 위험선호"] = "사용" if macro is not None and "usdkrw" in macro else "데이터 없음"
        alt_status["한국은행 금리(ECOS)"] = ("키 없음" if not kr_data.get_key("ECOS_API_KEY") else
                                       "사용" if macro is not None and "kr3y" in macro else "오류")
    return p_up, pick, now_all, contrib, asof, wf, ev, close, alt_status


@st.cache_data(ttl=900, show_spinner=False)
def get_snapshot(ticker, spot, r, q):
    """과거 기록이 무료로 없어 검증할 수 없는 '현재 값' 지표들."""
    out = {}
    for key, fn, args in (("options", ad.options_snapshot, (ticker, spot, r, q)),
                          ("revisions", ad.estimate_revisions, (ticker,)), ("news", ad.news_sentiment, (ticker,))):
        try:
            out[key] = fn(*args)
        except Exception:
            out[key] = None
    return out


@st.cache_data(show_spinner=False, max_entries=4)
def get_universe(h, cost_bps, data_stamp):
    """data_stamp: 데이터 갱신 시각 — 새로 받으면 다시 계산되도록 캐시 키에 넣는다."""
    panel = uv.build_panel()
    wf = uv.walk_forward(panel, h)
    table, curves = uv.evaluate(wf, cost_bps)
    d, pick, top, bottom = uv.current_picks(panel, wf)
    n_univ = int(panel["universe"].iloc[-1].sum())
    return {"table": table, "curves": curves, "ic": wf["ic"]["auto"], "chosen": wf["chosen"], "date": d,
            "pick": pick, "top": top, "bottom": bottom, "coverage": panel["coverage"], "cols": wf["cols"],
            "n_univ": n_univ}


@st.cache_data(ttl=3600, show_spinner=False)
def get_vol_backtest(ticker, vrp, threshold, cost_bps, r, q, iv_gap, option_spread):
    """변동성 지수(VIX 등) − iv_gap 을 과거 ATM IV로 써서 변동성 신호를 워크포워드 검증 (최대 10년) + 현실성 점검.

    지수가 없으면 None.
    """
    iv = md.load_vol_index(ticker, 10)
    if iv is None:
        return None
    close = md.load_history(ticker, 10)
    kw = dict(vrp=vrp, threshold=threshold, cost_bps=cost_bps, r=r, q=q, iv_gap=iv_gap, option_spread=option_spread)
    bt = vol_signal_backtest(close, iv, **kw)
    if bt.empty:
        raise ValueError("과거 데이터가 너무 짧습니다 (최소 약 2년 필요)")
    table, curves = summarize_vol_backtest(bt)
    by_gap, by_offset = vol_backtest_robustness(close, iv, bt, **kw)
    return {"bt": bt, "table": table, "curves": curves, "by_gap": by_gap, "by_offset": by_offset}


@st.cache_data(ttl=3600, show_spinner=False)
def get_regime(ticker, cost_bps):
    """국면 모델은 표본이 많이 필요해서 학습 기간 설정과 별개로 최대 10년치를 쓴다."""
    return rg.analyze(md.load_history(ticker, 10), cost_bps)


@st.cache_data(show_spinner=False)
def run_backtest(returns, h, step=10):
    """과거 여러 시점에서 '그때까지 데이터만으로' 예측 → 이후 실제와 비교 (look-ahead 없음).

    범위·VaR는 앱 화면과 같은 방식(t분포 GARCH 시뮬레이션의 분위수)으로 판정한다.
    """
    r = np.asarray(returns)
    start = max(250, len(r) - 500)
    rows = []
    for o in range(start, len(r) - h + 1, step):
        g = GarchVolForecaster().fit(r[max(0, o - 750):o])
        f = g.forecast_vol(h)
        real = np.sqrt(np.mean(r[o:o + h] ** 2) * 252)
        naive = np.sqrt(np.mean(r[o - max(h, 21):o] ** 2) * 252)
        q05, q025, q16, q84, q975 = np.quantile(g.simulate(h, 2000, seed=o)[:, -1], [0.05, 0.025, 0.16, 0.84, 0.975])
        move = r[o:o + h].sum()
        rows.append((o, f, naive, real, q16 <= move <= q84, q025 <= move <= q975, move < q05))
    return pd.DataFrame(rows, columns=["origin", "garch", "naive", "realized", "in1", "in2", "var_breach"])


# ---------------- 사이드바 ----------------
with st.sidebar:
    st.header("설정")
    query = st.text_input("종목명 또는 티커", "SPY",
                          help="한국: 삼성전자, 에코프로, 005930 / 미국: apple, tesla, AAPL, SPY (한국 개별 종목은 옵션 데이터 없음)")
    with st.spinner("종목 찾는 중…"):
        matches = find_ticker(query)
    if not matches:
        st.error(f"'{query}'에 해당하는 종목을 찾지 못했어요. 이름을 다르게 쓰거나 종목코드를 입력해 보세요.")
        st.stop()
    if len(matches) == 1:
        ticker = matches[0][0]
        if matches[0][1] != ticker:
            st.caption(f"→ {matches[0][1]}")
    else:
        ticker = st.selectbox(f"검색 결과 {len(matches)}개", matches, format_func=lambda m: m[1])[0]
    horizon = st.slider("예측 기간 (거래일)", 5, 252, 63, help="주가 범위·변동성을 예측할 기간. 21일 ≈ 1개월, 63일 ≈ 3개월")
    years = st.slider("학습 데이터 (년)", 2, 10, 3, help="변동성 모델이 참고할 과거 기간")
    dir_h = st.select_slider("방향 예측 기간 (거래일)", [5, 10, 21, 42, 63], 21, help="이 기간 뒤 주가가 지금보다 높을지 예측")
    st.subheader("변동성 신호 설정")
    vrp = st.number_input("VRP (변동성 위험 프리미엄, %p)", 0.0, 20.0, 2.0, 0.5,
                          help="옵션은 보통 실제 변동성보다 조금 비싸게 거래됩니다. 그만큼을 빼고 비교합니다.") / 100
    threshold = st.number_input("관망 구간 (±%p)", 0.0, 20.0, 2.0, 0.5,
                                help="차이가 이 범위 안이면 매수·매도 신호 없이 '관망'으로 둡니다.") / 100
    manual_iv = st.checkbox("시장 기대 변동성(IV) 직접 입력", help="옵션 데이터가 없는 종목(한국 개별 종목 등)일 때 켜세요.")
    iv_input = st.number_input("시장 기대 변동성 IV (%)", 1.0, 300.0, 20.0, 0.5, disabled=not manual_iv) / 100
    st.subheader("손익 시뮬레이션 설정")
    cost_bps = st.number_input("거래비용 (bp, 1bp = 0.01%)", 0.0, 50.0, 1.0, 0.5)
    n_paths = st.select_slider("시뮬레이션 횟수", [200, 500, 1000, 2000], 500)
    st.button("새로고침", on_click=st.cache_data.clear, width="stretch")

hero_slot = st.empty()

try:
    with st.spinner(f"{ticker} 데이터 불러오는 중…"):
        close, spot, r, q, opt, r_src = get_market(ticker, years, horizon)
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

rets = np.log(close).diff().dropna()
garch = GarchVolForecaster().fit(rets.values)
rv_h = garch.forecast_vol(horizon)

# 옵션 IV: 실제 옵션 만기 기간에 맞춰 GARCH 예측도 다시 계산 (기간 일치)
if manual_iv:
    market_iv, opt_days, iv_note = iv_input, horizon, "직접 입력"
    opt_T = horizon / 252
elif opt:
    market_iv = opt["iv"]
    opt_days = max(1, round(opt["cal_days"] * 252 / 365))
    opt_T = opt["cal_days"] / 365
    iv_note = f"만기 {opt['expiry']} · 행사가 {opt['strike']:g} · 가격 {opt['source']}"
else:
    market_iv, opt_days, iv_note, opt_T = None, horizon, "옵션 데이터 없음", horizon / 252
rv_opt = garch.forecast_vol(opt_days)

logic = VolatilityTradingLogic(vrp=vrp, threshold=threshold)
signal = logic.generate_signal(rv_opt, market_iv) if market_iv is not None else None
if signal is not None:   # "SHORT_VOL (변동성 매도)" → 코드 / 한글 이름
    sig_code, sig_name = VolatilityTradingLogic.LABELS[signal].rstrip(")").split(" (")


# ---------------- 상단 요약 카드 ----------------
def show_hero(dir_metric):
    name = dict(matches).get(ticker, ticker)
    with hero_slot.container():
        ui.hero("Volatility · Direction · KOSPI", name, f"{spot:,.2f}", close.pct_change().iloc[-1],
                [("기준일", close.index[-1].date()), ("학습 데이터", f"{years}년"), ("무위험 금리", f"{r:.2%}")],
                [(f"예측 변동성 · {horizon}일", f"{rv_h:.1%}",
                  f"1년 기준 환산 · 평소 수준 {np.sqrt(garch.long_run_var * 252):.1%}", False),
                 ("시장 기대 변동성 (IV)", f"{market_iv:.1%}" if market_iv is not None else "—", iv_note, False),
                 ("변동성 신호", sig_name if signal is not None else "—",
                  f"{sig_code} · edge {logic.edge(rv_opt, market_iv):+.1%}" if market_iv is not None else "IV 필요",
                  signal is not None and signal != 0),
                 dir_metric],
                "모든 수치는 그 시점까지의 데이터로만 계산한 추정치이며, 투자 조언이 아닙니다.")


show_hero((f"{dir_h}일 뒤 상승 확률", "…", "방향 모델 학습 중", False))

try:
    with st.spinner("방향 예측 모델 학습 중…"):
        p_up, pick, now_all, contrib, dir_asof, wf, ev, close_long, alt_status = get_direction(ticker, dir_h, cost_bps)
except Exception as e:
    p_up, ev, dir_err = None, None, e
# 확률 정확도(Brier)가 기본 확률보다 좋고, 적중률 우위가 통계적으로 유의할 때만 '검증 통과'
reliable = ev is not None and ev["skill"] > 0 and ev["z"] > 2

show_hero((f"{dir_h}일 뒤 상승 확률", f"{p_up:.0%}" if p_up is not None else "—",
           ("과거 검증 통과" if reliable else "과거 상승 비율" if pick == "base" else "과거 검증 미통과 · 참고용")
           if p_up is not None else "데이터 부족", reliable))

tab1, tab_dir, tab_pick, tab_snap, tab2, tab_regime, tab3, tab4 = st.tabs(
    ["주가 예측 범위", "방향 예측", "코스피 종목 고르기", "시장 심리", "변동성", "시장 국면",
     "신호 & 헤지 손익", "예측 정확도 검증"])

# ---------------- 1. 주가 예측 범위 ----------------
with tab1:
    ui.section("Price Range Forecast", f"앞으로 {horizon}거래일, 주가가 움직일 범위",
               "최근 변동성을 바탕으로 주가가 얼마나 크게 움직일 수 있는지 보여줍니다. "
               "진한 띠 안에 머물 확률은 약 68%, 연한 띠는 약 95%입니다. 폭락·폭등이 정규분포보다 자주 나오는 점을 "
               "t분포 GARCH로 반영해 1만 번 시뮬레이션했습니다. 오를지 내릴지는 <b>방향 예측</b> 탭에서 봅니다.")
    cum_sd = np.sqrt(np.cumsum(garch.daily_var_path(horizon)))     # k일 누적 로그수익률 표준편차
    fut_idx = pd.bdate_range(close.index[-1], periods=horizon + 1)[1:]
    sims = garch.simulate(horizon)                                  # (경로, 일) 누적 로그수익률
    qs = spot * np.exp(np.quantile(sims, [0.025, 0.16, 0.84, 0.975], axis=0))
    bands = {0.68: (qs[1], qs[2]), 0.95: (qs[0], qs[3])}

    show = close.iloc[-min(len(close), max(252, 3 * horizon)):]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=show.index, y=show.values, name="종가", line=dict(color=ui.INK)))
    for p, color in ((0.95, ui.BAND_OUTER), (0.68, ui.BAND_INNER)):
        lo, hi = bands[p]
        x = [close.index[-1], *fut_idx]
        fig.add_trace(go.Scatter(x=x, y=[spot, *hi], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=[spot, *lo], fill="tonexty", fillcolor=color, line=dict(width=0),
                                 name=f"{p:.0%} 범위"))
    fig.add_trace(go.Scatter(x=[close.index[-1], fut_idx[-1]], y=[spot, spot], name="중앙값",
                             line=dict(color=ui.NEUTRAL, dash="dot")))
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.08))
    ui.chart(fig)

    marks = sorted({k for k in (5, 21, 63, 126, 252) if k <= horizon} | {horizon})
    st.dataframe(pd.DataFrame({
        "기간(거래일)": marks,
        "날짜": [fut_idx[k - 1].date() for k in marks],
        "68% 하단": [bands[0.68][0][k - 1] for k in marks],
        "68% 상단": [bands[0.68][1][k - 1] for k in marks],
        "95% 하단": [bands[0.95][0][k - 1] for k in marks],
        "95% 상단": [bands[0.95][1][k - 1] for k in marks],
        "보통 움직임 폭(±1σ)": [f"±{s:.1%}" for s in cum_sd[np.array(marks) - 1]],
    }).style.format({c: "{:,.2f}" for c in ("68% 하단", "68% 상단", "95% 하단", "95% 상단")}),
        hide_index=True, width="stretch")

    st.subheader("손실 위험 (VaR · ES)")
    hist_h = np.log(close).diff(horizon).dropna().values           # 학습 기간의 실제 h일 수익률 (겹침)
    risk = {"1d95": var_es(sims[:, 0], 0.95), "1d99": var_es(sims[:, 0], 0.99), "h95": var_es(sims[:, -1], 0.95)}
    past = {"1d95": var_es(rets.values, 0.95), "1d99": var_es(rets.values, 0.99),
            "h95": var_es(hist_h, 0.95) if len(hist_h) >= 100 else (np.nan, np.nan)}
    v1, v2, v3, v4 = st.columns(4)
    for col, key, label in ((v1, "1d95", "하루 VaR 95%"), (v2, "1d99", "하루 VaR 99%"), (v3, "h95", f"{horizon}거래일 VaR 95%")):
        col.metric(label, f"−{risk[key][0]:.1%}",
                   f"과거 실제 −{past[key][0]:.1%}" if np.isfinite(past[key][0]) else "과거 표본 부족", delta_color="off")
    v4.metric(f"{horizon}거래일 ES 95%", f"−{risk['h95'][1]:.1%}",
              f"과거 실제 −{past['h95'][1]:.1%}" if np.isfinite(past["h95"][1]) else "과거 표본 부족", delta_color="off",
              help="최악 5% 경우의 평균 손실")
    st.caption(f"VaR 95%는 '100번 중 95번은 손실이 이보다 작다'는 뜻입니다. 예를 들어 1,000만 원어치를 들고 있다면 "
               f"하루에 약 {risk['1d95'][0] * 1e7:,.0f}원 넘게 잃을 확률이 5%입니다. ES는 그 나머지 최악 5% 경우의 평균 손실입니다. "
               f"큰 값은 t분포 GARCH(꼬리 두께 ν = {garch.nu:.1f}) 시뮬레이션, '과거 실제'는 학습 기간({years}년)의 실제 수익률로 계산했습니다.")

# ---------------- 방향 예측 ----------------
with tab_dir:
    if p_up is None:
        st.error(f"방향 예측을 할 수 없습니다 (데이터 부족 등): {dir_err}")
    else:
        view = "상승" if p_up > 0.5 else "하락"
        strength = abs(p_up - 0.5)
        level = ("모델 신호 없음, 과거 상승 비율" if pick == "base" else
                 "강함" if strength >= 0.15 else "보통" if strength >= 0.07 else "약함(거의 동전 던지기)")
        ui.section("Direction Forecast", f"{dir_h}거래일 뒤, 지금보다 오를까?")
        ui.verdict(f"기준일 {dir_asof.date()} · 사용 모델 {dr.MODEL_LABELS[pick]}", f"{view} {p_up:.1%}", "상승 확률",
                   [(f"강도 {level}", "neutral"),
                    ("과거 검증 통과", "approve") if reliable else ("과거 검증 미통과", "elective")],
                   "최대 10년치 하루 주가로 여러 모델을 시험해, 최근 3년 동안 가장 잘 맞힌 모델을 골랐습니다.")
        ui.badges("모델에 들어간 데이터", [("가격 특징 · 사용", "approve")] + [
            (f"{k} · {'사용' if v.startswith('사용') else '키 없음' if v == '키 없음' else '오류' if v.startswith('오류') else '없음'}",
             "approve" if v.startswith("사용") else "neutral" if v == "키 없음" else
             "reject" if v.startswith("오류") else "neutral") for k, v in alt_status.items()])
        if any(v == "키 없음" for v in alt_status.values()):
            st.caption("키 없음 = API 키가 없어 꺼짐. `api_keys.env` 파일에 `DART_API_KEY=…`, `ECOS_API_KEY=…`를 넣으면 켜집니다.")
        errs = {k: v for k, v in alt_status.items() if v.startswith("오류")}
        if errs:
            st.caption(" / ".join(f"{k}: {v}" for k, v in errs.items()))

        if pick == "base":
            st.info("지금은 어떤 모델도 **이 종목이 과거에 오른 비율**보다 잘 맞히지 못했습니다. "
                    "그래서 그 비율을 그대로 상승 확률로 보여줍니다. 근거 없는 예측을 내지 않기 위한 장치입니다.")
        if reliable:
            st.success(f"과거 검증에서 이 모델은 '항상 오른다'고 보는 것보다 확실히 잘 맞혔습니다 "
                       f"(확률 정확도 {ev['skill']:+.1%} 개선, 우연일 가능성 낮음 z = {ev['z']:.1f}). "
                       "그래도 앞으로도 그럴 거라는 보장은 없습니다.")
        else:
            st.warning(f"과거 검증에서 이 모델은 '항상 오른다'고 보는 것보다 **확실히 낫지 않았습니다** "
                       f"(z = {ev['z']:.1f}, 확률 정확도 {ev['skill']:+.1%}). 참고로만 보세요. "
                       "공개된 주가만으로 방향을 맞히기는 원래 매우 어렵습니다.")

        left, right = st.columns(2)
        with left:
            st.markdown("**어떤 지표가 상승·하락 쪽으로 기울였나** (오른쪽으로 길수록 상승 쪽, 간단한 모델 기준)")
            cs = contrib.sort_values()
            fig = go.Figure(go.Bar(x=cs.values, y=cs.index, orientation="h",
                                   marker_color=[ui.POS if v > 0 else ui.NEG for v in cs.values]))
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="영향 크기")
            ui.chart(fig)
        if ev is not None:
            with right:
                st.markdown("**과거 성적표** (매 시점 그때까지의 데이터만 써서 예측한 결과)")
                m1, m2 = st.columns(2)
                m1.metric("적중률", f"{ev['hit']:.1%}")
                m2.metric("'항상 오른다'고 했을 때", f"{ev['base']:.1%}",
                          f"차이 {ev['hit'] - ev['base']:+.1%}p", delta_color="off")
                m3, m4 = st.columns(2)
                m3.metric("확률 오차 (Brier, 낮을수록 좋음)", f"{ev['brier']:.3f}",
                          f"기준 {ev['brier_base']:.3f}", delta_color="off")
                m4.metric("검증 표본", f"{ev['n']:,}일", f"서로 겹치지 않는 구간 ≈ {ev['n_eff']:.0f}개", delta_color="off")

            st.markdown("**후보 모델 비교** — 모두 같은 조건으로 시험했습니다. 이 표를 보고 나중에 1등을 고르면 성적이 부풀려지므로, "
                        "앱은 매 시점 그때까지의 성적만 보고 고른 **자동 선택** 결과를 씁니다.")
            st.dataframe(pd.DataFrame([{
                "모델": dr.MODEL_LABELS.get(k, "자동 선택"),
                "오늘 상승 확률": now_all.get(k, p_up),
                "적중률": s["hit"], "'항상 오른다' 대비": s["hit"] - s["base"],
                "구분력 AUC (0.5=무작위)": s["auc"], "확률 정확도 개선": s["skill"],
            } for k, s in ev["table"].items()]).style.format({
                "오늘 상승 확률": "{:.1%}", "적중률": "{:.1%}", "'항상 오른다' 대비": "{:+.1%}",
                "구분력 AUC (0.5=무작위)": "{:.3f}", "확률 정확도 개선": "{:+.1%}"}),
                hide_index=True, width="stretch")
            share = ", ".join(f"{dr.MODEL_LABELS[k]} {v:.0%}" for k, v in ev["chosen_share"].items())
            st.caption(f"검증 기간 동안 자동 선택된 비율: {share}. "
                       "구분력(AUC)은 오를 때와 내릴 때를 얼마나 잘 가르는지(0.5면 동전 던지기), "
                       "확률 정확도 개선은 과거 상승 비율을 그대로 쓸 때보다 확률이 몇 % 더 정확한지입니다.")

            st.markdown(f"**상승 확률이 50%를 넘을 때만 들고 있었다면?** 단순 보유와 비교 "
                        f"(거래비용 {cost_bps}bp, 주식을 들고 있던 기간 {ev['time_in_market']:.0%}, "
                        f"위험 대비 수익(샤프) {ev['sharpe']:.2f} vs {ev['sharpe_bh']:.2f}, "
                        f"최대 낙폭 {ev['mdd']:.0%} vs {ev['mdd_bh']:.0%})")
            eq = ev["equity"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=eq.index, y=eq["전략"], name="50% 넘을 때만 보유", line=dict(color=ui.BLUE)))
            fig.add_trace(go.Scatter(x=eq.index, y=eq["단순 보유"], name="단순 보유", line=dict(color=ui.NEUTRAL)))
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                              yaxis_title="누적 (시작=1)", legend=dict(orientation="h", y=1.1))
            ui.chart(fig)

            st.markdown("**상승 확률 추이와 주가** (같은 기간, 위·아래로 나눠 표시)")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.5, 0.5])
            fig.add_trace(go.Scatter(x=wf.index, y=wf["auto"], name="상승 확률 (자동 선택)",
                                     line=dict(color=ui.BLUE)), row=1, col=1)
            fig.add_hline(y=0.5, line=dict(color=ui.NEUTRAL, dash="dot"), row=1, col=1)
            px_view = close_long[close_long.index >= wf.index[0]]
            fig.add_trace(go.Scatter(x=px_view.index, y=px_view.values, name="종가",
                                     line=dict(color=ui.INK, width=1.5)), row=2, col=1)
            fig.update_yaxes(tickformat=".0%", title_text="상승 확률", row=1, col=1)
            fig.update_yaxes(title_text="종가", row=2, col=1)
            fig.update_layout(height=420, hovermode="x unified", legend=dict(orientation="h", y=1.08))
            ui.chart(fig)

# ---------------- 코스피 종목 고르기 ----------------
with tab_pick:
    ui.section("KOSPI Ranking", "코스피 대형주 중 상대적으로 더 오를 종목",
               "한 종목의 방향 대신, 코스피 대형주 약 200개를 서로 비교해 순위를 매깁니다. "
               "사이드바의 종목 설정과 무관하게 동작합니다.")
    info = uv.data_info()

    def run_update():
        bar = st.progress(0.0, text="시작…")
        uv.update_data(progress=lambda p, m: bar.progress(min(max(p, 0.0), 1.0), text=m))
        st.cache_data.clear()
        st.rerun()

    if info is None:
        st.info("아직 데이터가 없습니다. 처음 받을 때 코스피 전 종목 주가와, 한 번이라도 200위 안에 든 종목의 "
                "6년치 수급·실적·공시·재무제표를 받느라 **20~40분** 걸립니다. "
                "(터미널에서 `python universe.py update`로 받아도 됩니다.)")
        if st.button("데이터 받기", type="primary"):
            run_update()
    else:
        c1, c2, c3 = st.columns([2, 1, 1])
        c1.caption(f"데이터 갱신: {info['updated']} · 코스피 {info.get('n_all', '?')}종목 중 "
                   f"과거 200위 이내 진입 {info['n']}종목 · {info['years']}년")
        hold = c2.selectbox("종목 교체 주기", [63, 21], format_func=lambda x: f"{x}거래일 (≈{x // 21}개월)")
        pick_cost = c3.number_input("거래비용 (매수·매도 각각, bp)", 0.0, 100.0, 15.0, 5.0,
                                    help="수수료+세금+체결 가격 차이. 국내 주식은 팔 때 세금이 있어 "
                                         "10~25bp(0.10~0.25%)가 현실적입니다.")
        if st.button("데이터 업데이트 (새 날짜만 받음)"):
            run_update()
        try:
            with st.spinner("과거 시점마다 모델을 학습·검증하는 중… (처음 1~3분)"):
                U = get_universe(hold, pick_cost, info["updated"])
        except Exception as e:
            st.error(f"계산 실패: {e}")
            U = None

        if U is not None:
            auto = U["table"].loc["auto"]
            ok = auto["초과수익 t값"] > 2 and auto["IC t값"] > 2
            pick_label = "고르지 않음 — 최근 성적이 좋은 모델이 없어, 전체 종목에 똑같이 나눠 담기를 권장" if U["pick"] == "none" \
                else uv.MODEL_LABELS[U["pick"]]
            ui.verdict(f"{U['date'].date()} 기준", pick_label, "",
                       [("과거 검증 통과", "approve") if ok else ("평균보다 낫다고 보기 어려움", "elective")])
            if ok:
                st.success(f"과거 검증에서 상위 20% 종목이 비용을 빼고도 전체 평균보다 확실히 더 올랐습니다 "
                           f"(1년 기준 {auto['상위20% 초과수익(연, 비용후)']:+.1%}, 우연일 가능성 낮음 t = {auto['초과수익 t값']:.1f}, "
                           f"순위 정확도 t = {auto['IC t값']:.1f}). 그래도 앞으로의 성과를 보장하지 않습니다.")
            else:
                st.warning(f"과거 검증에서 상위 20% 종목이 전체 평균보다 더 올랐다고 확신하기 어렵습니다 "
                           f"(1년 기준 {auto['상위20% 초과수익(연, 비용후)']:+.1%}, t = {auto['초과수익 t값']:.1f}, "
                           f"순위 정확도 t = {auto['IC t값']:.1f}; t가 2를 넘어야 우연이 아니라고 봅니다). 아래 순위는 참고로만 보세요.")

            cv = U["curves"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cv["auto"].index, y=cv["auto"]["상위 20% (비용 후)"], name="상위 20% 종목 (비용 뺀 후)",
                                     line=dict(color=ui.BLUE, width=2.5)))
            fig.add_trace(go.Scatter(x=cv["_universe"].index, y=cv["_universe"], name="전체 종목 평균",
                                     line=dict(color=ui.NEUTRAL)))
            fig.add_trace(go.Scatter(x=cv["auto"].index, y=cv["auto"]["하위 20%"], name="하위 20% 종목",
                                     line=dict(color=ui.PURPLE, dash="dot")))
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                              yaxis_title="누적 (시작=1)", legend=dict(orientation="h", y=1.1))
            ui.chart(fig)
            st.caption(f"최대 낙폭: 상위 20% {max_drawdown(cv['auto']['상위 20% (비용 후)']):.0%} · "
                       f"전체 종목 평균 {max_drawdown(cv['_universe']):.0%}. "
                       "리밸런싱 시점 기준이라 그 사이에 잠깐 더 빠졌던 하락은 덜 잡힙니다.")

            st.markdown("**모델 비교** — 모두 같은 조건으로 시험했습니다. 이 표를 보고 나중에 1등을 고르면 성적이 부풀려지므로, "
                        "앱은 매 시점 그때까지의 성적만 보고 고른 **자동 선택**을 씁니다.")
            st.dataframe(U["table"].style.format({
                "평균 IC": "{:+.3f}", "IC t값": "{:+.1f}", "IC>0 비율": "{:.0%}",
                "상위20% 초과수익(연, 비용후)": "{:+.1%}", "초과수익 t값": "{:+.1f}",
                "상위−하위 스프레드(연)": "{:+.1%}", "최대 낙폭(비용후)": "{:.0%}",
                "평균 회전율": "{:.0%}", "리밸런싱 수": "{:.0f}"}, na_rep="—"),
                hide_index=True, width="stretch")
            share = U["chosen"].value_counts(normalize=True)
            st.caption("자동 선택 비율: " + ", ".join(
                f"{uv.MODEL_LABELS.get(k, '고르지 않음')} {v:.0%}" for k, v in share.items()) +
                ". IC(순위 정확도)는 점수 순위와 실제 수익률 순위가 얼마나 일치하는지입니다. "
                "0이면 무관하고 0.05 정도면 실무에서 의미 있는 수준입니다. t값이 2를 넘으면 우연일 가능성이 낮다고 봅니다.")

            left, right = st.columns(2)
            fmt = {c: "{:.0%}" for c in U["top"].columns if c not in ("순위", "종목명", "점수")}
            fmt["점수"] = "{:+.3f}"
            with left:
                st.markdown(f"**상위 20종목** (전체 {U['n_univ']}개 중)")
                st.dataframe(U["top"].style.format(fmt), width="stretch")
            with right:
                st.markdown("**하위 20종목**")
                st.dataframe(U["bottom"].style.format(fmt), width="stretch")
            st.caption("신호 열은 전체 종목 중 상대 위치입니다 (100% = 가장 큼). 예: 외국인 순매수 강도 90%면 상위 10%.")

            ic = U["ic"].dropna()
            fig = go.Figure(go.Bar(x=ic.index, y=ic.values,
                                   marker_color=[ui.POS if v > 0 else ui.NEG for v in ic.values]))
            fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="교체 시점별 순위 정확도(IC)")
            ui.chart(fig)

            with st.expander("데이터·한계"):
                cov = pd.Series({uv.FEATURES[c][0]: U["coverage"].get(c, 0) for c in uv.FEATURES})
                st.markdown("**신호별 데이터 보유율** (30% 미만이면 모델에서 제외)")
                st.dataframe(cov.to_frame("보유율").style.format("{:.0%}"), width="stretch")
                st.markdown("""
- **종목 범위:** 현재 상장된 코스피 보통주 전체 중, 각 과거 시점의 거래대금 상위 200개 (공식 코스피200 구성과는 다름).
  '지금 큰 종목'으로 과거를 고르는 편향은 없앴지만, 그 사이 **상장폐지된 종목은 주가가 없어 빠져 있어**
  과거 성과가 약간 좋게 보일 수 있습니다.
- **가치 지표:** DART 재무제표(분기·사업보고서)의 자본총계·순이익을 **공시 다음 거래일부터** 씁니다.
  시가총액은 '그날 주가 × 현재 상장주식수'로 근사해서, 과거 증자·소각으로 주식수가 바뀐 종목은 오차가 있습니다.
- **비용:** 리밸런싱마다 바뀐 종목 비율 × 양방향 비용을 뺐습니다. 세금·호가 차이가 더 크면 결과가 나빠집니다.
- 투자 조언이 아닙니다.
""")

# ---------------- 시장 심리 스냅샷 ----------------
with tab_snap:
    ui.section("Market Sentiment", "옵션·컨센서스·공시·뉴스 현재 값")
    st.info("아래 **옵션 풋/콜·스큐, 실적 전망 변화, 뉴스 감성**은 무료로는 지금 값만 구할 수 있어, "
            "과거에 잘 맞았는지 확인할 수 없습니다. 그래서 예측 모델에는 넣지 않고 참고용으로만 보여줍니다. "
            "(실적 서프라이즈, 애널리스트 등급·목표가, 외국인·기관 수급, 시장 SKEW는 과거 기록이 있어 방향 예측 모델에 들어갑니다.)")
    with st.spinner("옵션·추정치·뉴스 불러오는 중…"):
        snap = get_snapshot(ticker, spot, r, q)

    st.subheader("옵션 시장")
    o = snap["options"]
    if o is None:
        st.caption("이 종목은 옵션 데이터가 없습니다 (한국 종목 등).")
    else:
        a, b, c, d = st.columns(4)
        a.metric("풋/콜 거래량 비율", f"{o['pc_volume']:.2f}", f"풋 {o['put_volume']:,.0f} / 콜 {o['call_volume']:,.0f}",
                 delta_color="off")
        b.metric("풋/콜 미결제약정 비율", f"{o['pc_oi']:.2f}")
        c.metric("스큐 (하락 대비 옵션 − 상승 옵션의 IV)", f"{o['skew']:+.1%}" if np.isfinite(o["skew"]) else "—",
                 f"풋 {o['iv_put90']:.1%} / 콜 {o['iv_call110']:.1%}", delta_color="off")
        d.metric("현재가 근처 옵션의 IV", f"{o['iv_atm']:.1%}" if np.isfinite(o["iv_atm"]) else "—", f"만기 {o['expiry']} ({o['days']}일)",
                 delta_color="off")
        st.caption("풋은 하락에 대비하는 옵션, 콜은 상승에 거는 옵션입니다. 풋/콜 비율이 1보다 크면 하락을 걱정하는 사람이 많다는 뜻이지만, "
                   "너무 높으면 오히려 바닥 신호로 보기도 합니다. 스큐가 클수록 시장이 하락 위험을 더 비싸게 쳐준다는 뜻입니다. "
                   "만기 70일 이내 옵션을 합산했고, 장 마감 후엔 거래량이 0일 수 있습니다.")

    st.subheader("애널리스트 실적(EPS) 전망")
    rv = snap["revisions"]
    code = ad.krx_code(ticker)
    if code:
        st.caption("한국 종목은 네이버 증권의 실적과 전망치를 보여줍니다. **(E)**가 붙은 열은 애널리스트들의 평균 전망치입니다. "
                   "금액 단위: 억원, 비율: %")
        for label, period in (("분기", "quarter"), ("연간", "annual")):
            try:
                fin = ad.naver_financials(code, period)
            except Exception:
                fin = None
            if fin is not None and not fin.empty:
                st.markdown(f"**{label}**")
                st.dataframe(fin.style.format("{:,.2f}", na_rep="—"), width="stretch")
            else:
                st.caption(f"{label} 재무 데이터를 불러오지 못했습니다.")
    elif rv is None or rv.empty:
        st.caption("추정치 데이터가 없습니다.")
    else:
        fmt = {c: "{:+.2%}" for c in rv.columns if "대비" in c}
        fmt["현재 추정 EPS"] = "{:,.2f}"
        st.dataframe(rv.style.format(fmt, na_rep="—"), width="stretch")
        st.caption("전망치가 올라가는 중(+)이면 애널리스트들이 실적 기대를 높이고 있다는 뜻입니다. "
                   "과거 전망 변경 이력은 무료로 구할 수 없어, 과거 검증은 등급·목표가 변경 데이터로 대신합니다.")

    if code and kr_data.get_key("DART_API_KEY"):
        st.subheader("최근 공시 (DART)")
        try:
            dis = kr_data.dart_disclosures(code, close.index[0])
            recent = dis.sort_values("date", ascending=False).head(30).copy()
            recent["분류"] = recent["report"].map(kr_data.classify).map(
                lambda k: kr_data.DART_FEATURES.get(k, "기타" if k else "정정").split("(")[0])
            recent["링크"] = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=" + recent["rcept_no"]
            st.dataframe(recent[["date", "분류", "report", "링크"]], hide_index=True, width="stretch",
                         column_config={"date": st.column_config.DateColumn("접수일"), "report": "보고서명",
                                        "링크": st.column_config.LinkColumn("원문", display_text="열기")})
        except Exception as e:
            st.caption(f"공시를 불러오지 못했습니다: {e}")

    if code:
        st.subheader("외국인·기관 수급 (최근 60거래일)")
        try:
            fl = ad.load_investor_flows(code, close.index[-60]).iloc[-60:]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=fl.index, y=fl["frgn"], name="외국인 순매수(주)", marker_color=ui.BLUE))
            fig.add_trace(go.Bar(x=fl.index, y=fl["inst"], name="기관 순매수(주)", marker_color=ui.AMBER))
            fig.update_layout(barmode="group", height=300, margin=dict(l=10, r=10, t=30, b=10),
                              hovermode="x unified", legend=dict(orientation="h", y=1.1))
            ui.chart(fig)
        except Exception as e:
            st.caption(f"수급 데이터를 불러오지 못했습니다: {e}")

    st.subheader("뉴스 감성")
    nw = snap["news"]
    if nw is None or nw.empty:
        st.caption("뉴스를 불러오지 못했습니다.")
    else:
        recent = nw[nw["time"] >= nw["time"].max() - pd.Timedelta(days=7)]
        a, b, c = st.columns(3)
        a.metric("최근 7일 평균 감성", f"{recent['score'].mean():+.2f}", "−1 매우 부정 ~ +1 매우 긍정", delta_color="off")
        b.metric("긍정 / 부정 헤드라인", f"{(nw['score'] > 0).sum()} / {(nw['score'] < 0).sum()}",
                 f"중립 {(nw['score'] == 0).sum()}", delta_color="off")
        c.metric("수집 기간", f"{(nw['time'].max() - nw['time'].min()).days + 1}일", f"헤드라인 {len(nw)}개",
                 delta_color="off")
        daily = nw.groupby(nw["time"].dt.date)["score"].mean()
        fig = go.Figure(go.Bar(x=[d.strftime("%m-%d") for d in daily.index], y=daily.values,
                               marker_color=[ui.POS if v > 0 else ui.NEG if v < 0 else ui.NEUTRAL for v in daily.values]))
        fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="일평균 감성",
                          xaxis_type="category", yaxis_range=[-1, 1])
        ui.chart(fig)
        st.dataframe(nw[["time", "score", "title", "source", "link"]], hide_index=True, width="stretch",
                     column_config={"time": st.column_config.DatetimeColumn("시각", format="MM-DD HH:mm"),
                                    "score": st.column_config.NumberColumn("감성", format="%+.2f"),
                                    "title": "헤드라인", "source": "언론사",
                                    "link": st.column_config.LinkColumn("링크", display_text="열기")})
        st.caption("감성 점수는 제목에 나온 긍정·부정 단어를 세는 단순한 방식이라, 반어나 맥락은 이해하지 못합니다. "
                   "과거 뉴스 기록을 무료로 구할 수 없어 예측에 도움이 되는지는 확인하지 못했습니다.")

# ---------------- 2. 변동성 ----------------
with tab2:
    ui.section("Volatility", "실제 변동성 · 모델 추정 · 시장 기대 비교",
               "회색은 최근 21일 동안 실제로 움직인 폭, 파랑은 GARCH 모델이 추정한 변동성과 앞으로의 예측(점선), "
               "주황 점선은 옵션 가격에 담긴 시장의 기대(IV)입니다. 모두 1년 기준으로 환산했습니다.")
    rv21 = rets.rolling(21).std() * np.sqrt(252)
    cv = pd.Series(garch.cond_vol, index=rets.index)
    fut_vol = np.sqrt(garch.daily_var_path(horizon) * 252)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rv21.index, y=rv21, name="실제 변동성 (최근 21일)", line=dict(color=ui.NEUTRAL)))
    fig.add_trace(go.Scatter(x=cv.index, y=cv, name="모델 추정 (GARCH)", line=dict(color=ui.BLUE)))
    fig.add_trace(go.Scatter(x=fut_idx, y=fut_vol, name="모델 예측 (GARCH)", line=dict(color=ui.BLUE, dash="dash")))
    if market_iv is not None:
        fig.add_hline(y=market_iv, line=dict(color=ui.AMBER, dash="dot"), annotation_text=f"시장 기대 IV {market_iv:.1%}",
                      annotation_font_color=ui.INK_80)
    fig.update_layout(height=420, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified", legend=dict(orientation="h", y=1.08))
    ui.chart(fig)
    a, b, c, d, e = st.columns(5)
    a.metric("α (충격 반응)", f"{garch.alpha:.3f}", help="큰 하루 변동이 다음 날 변동성을 얼마나 키우는지")
    b.metric("β (지속성)", f"{garch.beta:.3f}", help="높아진 변동성이 얼마나 오래 유지되는지")
    c.metric("α+β", f"{garch.persistence:.3f}", help="1에 가까울수록 변동성 충격이 오래 지속")
    d.metric("반감기", f"{np.log(0.5) / np.log(garch.persistence):.0f}일" if garch.persistence < 1 else "∞")
    e.metric("꼬리 두께 (ν)", f"{garch.nu:.1f}",
             help="충격 분포(t분포)의 자유도. 작을수록 폭락·폭등이 자주 나옵니다. 정규분포는 ∞, 주식은 보통 4~10")

# ---------------- 시장 국면 ----------------
with tab_regime:
    ui.section("Market Regime", "지금은 안정 국면일까, 불안 국면일까",
               "주가가 조용히 움직이는 '안정 국면'과 크게 출렁이는 '불안 국면'을 2상태 국면 전환 모델(Markov switching)로 나눕니다. "
               "확률은 그날까지의 데이터만으로 계산해서, 과거 구간에도 미래 정보가 섞이지 않습니다.")
    try:
        with st.spinner("국면 모델 추정 중…"):
            R = get_regime(ticker, cost_bps)
    except Exception as e:
        R = None
        st.error(f"국면 분석을 할 수 없습니다 (데이터 부족 등): {e}")
    if R is not None:
        p_now, stt = R["p_now"], R["states"]
        high = p_now >= 0.5
        ui.verdict(f"기준일 {R['asof'].date()} · 최대 10년치 일봉", rg.STATE_LABELS[int(high)], "",
                   [(f"불안 국면 확률 {p_now:.0%}", "elective" if high else "approve")],
                   f"안정 국면은 연 변동성 약 {stt['연 변동성'].iloc[0]:.0%}, 불안 국면은 약 {stt['연 변동성'].iloc[1]:.0%}로 추정됩니다. "
                   f"한 번 들어가면 평균 {stt['평균 지속 기간(거래일)'].iloc[0]:.0f}거래일 / "
                   f"{stt['평균 지속 기간(거래일)'].iloc[1]:.0f}거래일 이어졌습니다.")

        ph = R["p_high"].dropna()
        px_r = R["close"].reindex(ph.index)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.6, 0.4])
        fig.add_trace(go.Scatter(x=px_r.index, y=px_r.values, name="종가", line=dict(color=ui.INK, width=1.5)), row=1, col=1)
        on = (ph >= 0.5).values                                   # 불안 국면 구간을 가격 차트에 옅게 칠함 (선을 먼저 그려야 칸이 생김)
        edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
        for a0, a1 in zip(edges[::2], edges[1::2]):
            fig.add_vrect(x0=ph.index[a0], x1=ph.index[a1 - 1], fillcolor="rgba(183,110,0,0.14)", line_width=0,
                          layer="below", row=1, col=1)
        fig.add_trace(go.Scatter(x=ph.index, y=ph.values, name="불안 국면 확률", fill="tozeroy",
                                 line=dict(color=ui.AMBER, width=1), fillcolor="rgba(183,110,0,0.18)"), row=2, col=1)
        fig.add_hline(y=0.5, line=dict(color=ui.NEUTRAL, dash="dot"), row=2, col=1)
        fig.update_yaxes(title_text="종가", row=1, col=1)
        fig.update_yaxes(tickformat=".0%", range=[0, 1], title_text="불안 국면 확률", row=2, col=1)
        fig.update_layout(height=480, hovermode="x unified", legend=dict(orientation="h", y=1.08))
        ui.chart(fig)
        st.caption("주황 구간 = 불안 국면 확률 50% 이상. 모델은 1년마다 그때까지의 데이터로만 다시 추정했고, 처음 3년은 학습 기간이라 비어 있습니다.")

        st.markdown("**국면별 특징** (전체 기간으로 추정)")
        st.dataframe(stt.style.format({"연 평균 수익률": "{:+.1%}", "연 변동성": "{:.1%}", "평균 지속 기간(거래일)": "{:.0f}"}),
                     width="stretch")

        st.markdown(f"**불안 국면일 때 현금으로 피했다면?** 단순 보유와 비교 (거래비용 {cost_bps}bp, "
                    f"주식을 들고 있던 기간 {R['time_in_market']:.0%}, 국면이 바뀌어 사고판 횟수 {R['switches']}번)")
        st.dataframe(R["table"].style.format({"연 수익률": "{:+.1%}", "연 변동성": "{:.1%}", "샤프": "{:.2f}",
                                              "최대 낙폭": "{:.0%}"}), width="stretch")
        eq = R["equity"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq.index, y=eq["불안 국면엔 현금"], name="불안 국면엔 현금", line=dict(color=ui.BLUE)))
        fig.add_trace(go.Scatter(x=eq.index, y=eq["단순 보유"], name="단순 보유", line=dict(color=ui.NEUTRAL)))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                          yaxis_title="누적 (시작=1)", legend=dict(orientation="h", y=1.1))
        ui.chart(fig)
        st.caption("확률이 50% 이상인 날 종가에 팔고 다음 날부터 현금(이자 0으로 계산), 50% 아래로 내려오면 다시 삽니다. "
                   "국면 분류는 보통 수익을 늘리기보다 큰 하락을 줄이는 도구입니다. 상승장 초입을 놓치는 대가가 따릅니다.")

# ---------------- 3. 신호 & 헤지 손익 ----------------
with tab3:
    ui.section("Signal & Hedge P&L", "변동성 매수·매도 신호와 델타헤지 손익",
               "옵션 가격에 담긴 시장의 기대 변동성(IV)이 모델 예측보다 충분히 높으면 옵션이 비싸다고 보고 '변동성 매도', "
               "충분히 낮으면 싸다고 보고 '변동성 매수' 신호를 냅니다.")
    if market_iv is None:
        st.warning("이 종목은 옵션 데이터가 없어 신호를 낼 수 없습니다. 사이드바에서 '시장 기대 변동성(IV) 직접 입력'을 켜세요.")
    else:
        edge = logic.edge(rv_opt, market_iv)
        ui.verdict(f"edge {edge:+.2%} · 관망 구간 ±{threshold:.1%}", sig_name, "", [(sig_code, "neutral")])
        st.markdown(f"""
| 항목 | 값 |
|---|---|
| 시장 기대 변동성 IV ({iv_note}) | **{market_iv:.2%}** |
| GARCH 예측 RV (옵션 만기 {opt_days}거래일 평균) | {rv_opt:.2%} |
| + VRP | {vrp:.2%} |
| **edge = IV − (RV + VRP)** | **{logic.edge(rv_opt, market_iv):+.2%}** (관망 구간 ±{threshold:.1%}) |
| 무위험 금리 ({r_src}) / 배당수익률 | {r:.2%} / {q:.2%} |
""")
        position = signal if signal != 0 else st.radio(
            "지금은 관망 신호입니다. 손익을 시험해 볼 방향을 고르세요:", [1, -1],
            format_func=lambda p: VolatilityTradingLogic.LABELS[p], horizontal=True)

        eng = GammaScalpingEngine(spot, spot, opt_T, r, market_iv, q)
        gk = eng.straddle_greeks()
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("스트래들 이론 가격", f"{gk['price']:.2f}", help="현재가 근처의 콜과 풋을 함께 산 가격")
        g2.metric("델타", f"{gk['delta']:+.3f}", help="주가가 1 오를 때 스트래들 가격이 변하는 양")
        g3.metric("감마", f"{gk['gamma']:.4f}", help="주가가 1 오를 때 델타가 변하는 양")
        g4.metric("하루 시간 가치 감소 (세타)", f"{gk['theta_per_day']:+.3f}")

        st.markdown(f"**델타헤지 손익 시뮬레이션** — 스트래들 1개를 사거나 판 뒤 매일 주식으로 방향 위험을 없앴을 때, "
                    f"만기까지의 손익을 상황별로 {n_paths}번씩 가상 실험했습니다 (거래비용 {cost_bps}bp).")
        rng = np.random.default_rng(0)
        scenarios = [("GARCH 예측대로", rv_opt), ("IV 수준", market_iv),
                     ("IV × 1.5 (급등장)", 1.5 * market_iv), ("IV × 0.6 (조용한 장)", 0.6 * market_iv)]
        scen_color = dict(zip([s[0] for s in scenarios], [ui.BLUE, ui.NEUTRAL, ui.PURPLE, ui.AMBER]))
        rows, hist = [], go.Figure()
        for name, vol in scenarios:
            paths = simulate_gbm_paths(spot, vol, r, opt_T, opt_days, n_paths, rng)
            pnl = eng.hedged_pnl(paths, position=position, cost_bps=cost_bps)
            rows.append({"시나리오": name, "실제 변동성": f"{vol:.1%}", "평균 손익": pnl.mean(),
                         "승률": f"{np.mean(pnl > 0):.0%}", "최악": pnl.min(), "최선": pnl.max(),
                         "옵션 가격 대비 평균 손익": f"{pnl.mean() / gk['price']:+.1%}"})
            hist.add_trace(go.Histogram(x=pnl, name=name, opacity=0.55, nbinsx=60, marker_color=scen_color[name]))
        st.dataframe(pd.DataFrame(rows).style.format({"평균 손익": "{:+.2f}", "최악": "{:+.2f}", "최선": "{:+.2f}"}),
                     hide_index=True, width="stretch")
        hist.update_layout(barmode="overlay", height=360, xaxis_title="만기 손익 (주가 단위)",
                           margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.1))
        ui.chart(hist)
        if position == -1:
            st.warning("변동성 매도는 예측이 맞으면 꾸준히 벌지만, 실제 변동성이 시장 기대(IV)를 넘으면 손실이 크게 날 수 있습니다. "
                       "포지션 크기를 제한하고 손절 기준을 꼭 정하세요.")

    st.subheader("과거 성적 — 이 신호대로 거래했다면?")
    vix_sym = md.VOL_INDEX.get(ticker.upper())
    if vix_sym is None:
        st.caption("과거 IV가 필요해서 변동성 지수가 있는 종목만 검증할 수 있습니다: SPY·VOO·IVV (VIX), QQQ (VXN), IWM (RVX).")
    else:
        vix_name = vix_sym.lstrip("^")
        st.markdown(f"과거 {vix_name}(30일 만기 옵션에 담긴 시장 기대 변동성)에서 아래 차이를 뺀 값을 그날의 ATM IV로 보고, "
                    "한 달(21거래일)마다 그때까지의 데이터로만 GARCH 예측 → 신호 → ATM 스트래들 진입 → 매일 델타헤지 → 만기 정산을 "
                    "반복합니다. 사이드바의 VRP·관망 구간·거래비용(주식 헤지)을 그대로 씁니다.")
        o1, o2 = st.columns(2)
        iv_gap = o1.number_input(f"{vix_name}와 실제 ATM IV의 차이 (%p)", 0.0, 5.0, 2.0, 0.5,
                                 help=f"{vix_name}는 외가격 옵션까지 섞어 계산해서 ATM 옵션 IV보다 보통 1~3%p 높습니다. "
                                      "IV = 지수 − 이 값으로 계산합니다.") / 100
        opt_spread = o2.number_input("옵션 체결 비용 (스트래들 가격 대비 %)", 0.0, 10.0, 1.0, 0.5,
                                     help="중간 가격보다 불리하게 사고파는 비용(호가 차이의 절반). 진입할 때 한 번 냅니다. "
                                          "거래가 많은 대형 ETF 옵션은 작고, 거래가 적은 옵션은 큽니다.") / 100
        if st.button("과거 성적 계산", type="primary", key="vol_bt"):
            try:
                with st.spinner("최대 10년치를 한 달씩 다시 예측하고, 차이·진입일을 바꿔 가며 점검하는 중…"):
                    res = get_vol_backtest(ticker, vrp, threshold, cost_bps, r, q, iv_gap, opt_spread)
            except Exception as e:
                res = "error"
                st.error(f"계산 실패: {e}")
            if res is None:
                st.warning(f"{vix_name} 데이터를 불러오지 못했습니다.")
            elif res != "error":
                vbt, vtab, vcurves = res["bt"], res["table"], res["curves"]
                sg, sh, lg = vtab.loc["signal"], vtab.loc["short"], vtab.loc["long"]
                if sg["거래 수"] == 0:
                    tags = [("신호가 한 번도 나지 않음", "neutral")]
                else:
                    tags = [("이익이 우연이 아님 (t > 2)", "approve") if sg["t값"] > 2 else ("이익이 우연일 수 있음", "elective"),
                            ("항상 매도보다 나음", "approve") if sg["샤프(연)"] > sh["샤프(연)"] else ("항상 매도보다 못함", "elective")]
                ui.verdict(f"{vbt.index[0].date()} ~ {vbt.index[-1].date()} · 진입 {len(vbt)}번 · "
                           f"IV = {vix_name} − {iv_gap * 100:g}%p · 옵션 체결 비용 {opt_spread:.1%}",
                           f"샤프 {sg['샤프(연)']:.2f}", "GARCH 신호대로 (연)", tags,
                           f"비교: 항상 변동성 매도 샤프 {sh['샤프(연)']:.2f}, 항상 매수 {lg['샤프(연)']:.2f}. "
                           f"신호는 매수 {int((vbt['signal'] == 1).sum())}번 · 매도 {int((vbt['signal'] == -1).sum())}번 · "
                           f"관망 {int((vbt['signal'] == 0).sum())}번이었습니다.")
                vfmt = {"거래 수": "{:.0f}", "승률": "{:.0%}", "평균 손익(주가 대비)": "{:+.2%}",
                        "평균 손익(프리미엄 대비)": "{:+.1%}", "t값": "{:+.1f}", "샤프(연)": "{:.2f}", "최대 낙폭": "{:.0%}",
                        "최악 1회": "{:+.1%}"}
                st.dataframe(vtab.set_index("전략").style.format(vfmt, na_rep="—"), width="stretch")
                fig = go.Figure()
                for name, color in zip(vcurves.columns, (ui.BLUE, ui.AMBER, ui.PURPLE)):
                    fig.add_trace(go.Scatter(x=vcurves.index, y=vcurves[name], name=name, line=dict(color=color)))
                fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                                  yaxis_title="누적 (시작=1, 주가 1주당 스트래들 1개)", legend=dict(orientation="h", y=1.1))
                ui.chart(fig)
                k1, k2, k3 = st.columns(3)
                k1.metric("과거 평균 VRP", f"{(vbt['iv'] - vbt['realized']).mean():+.1%}",
                          "IV − 이후 21일 실제 변동성", delta_color="off")
                vrmse = lambda c: np.sqrt(np.mean((vbt[c] - vbt["realized"]) ** 2))
                k2.metric("GARCH 예측 오차", f"{vrmse('garch'):.1%}", "실제 변동성 대비 RMSE", delta_color="off")
                k3.metric("IV 예측 오차", f"{vrmse('iv'):.1%}", "실제 변동성 대비 RMSE", delta_color="off")
                st.caption(f"이 화면의 IV = {vix_name} − {iv_gap * 100:g}%p입니다. 실제 차이는 시기마다 달라서 아래 '차이를 바꿔 보면' 표로 함께 "
                           "확인하세요. 금리·배당은 현재 값으로 고정했습니다. 평균 VRP를 보고 사이드바 VRP를 맞춘 뒤 다시 돌리면 "
                           "과거를 보고 맞춘 것이라 성적이 부풀려집니다.")

                # ---- 현실성 점검: IV 차이와 진입일
                rfmt = {"진입 수": "{:.0f}", "항상 매도 평균 손익(주가 대비)": "{:+.2%}", "항상 매도 t값": "{:+.1f}",
                        "항상 매도 샤프": "{:.2f}", "항상 매도 최대 낙폭": "{:.0%}", "신호대로 샤프": "{:.2f}", "신호대로 t값": "{:+.1f}"}
                st.markdown(f"**현실성 점검 ① {vix_name}와 ATM IV의 차이를 바꿔 보면** (다른 설정은 그대로)")
                st.dataframe(res["by_gap"].drop(columns="첫 진입일").style.format(rfmt, na_rep="—"), width="stretch")
                st.caption("한 달짜리 ATM 스트래들은 IV가 1%p 낮으면 받는 가격이 주가의 약 0.23% 줄어듭니다. "
                           "그래서 차이를 크게 볼수록 '항상 매도'의 이익이 줄어듭니다. 0%p 줄이 지수를 그대로 쓴 예전 결과입니다.")
                st.markdown("**현실성 점검 ② 진입일을 바꿔 보면** (첫 진입을 며칠씩 늦춰 같은 계산을 반복)")
                st.dataframe(res["by_offset"].style.format(rfmt, na_rep="—"), width="stretch")
                so_sh = res["by_offset"]["항상 매도 샤프"].astype(float)
                sg_sh = res["by_offset"]["신호대로 샤프"].astype(float)
                st.caption(f"진입일에 따라 항상 매도 샤프는 {so_sh.min():.2f} ~ {so_sh.max():.2f}, 신호대로 샤프는 "
                           f"{sg_sh.min():.2f} ~ {sg_sh.max():.2f} 사이였습니다. 폭이 넓을수록 결과에 진입일 운이 많이 섞여 있다는 뜻입니다.")

                # ---- (1) IV를 예측에 같이 쓰기
                vbt2 = add_combined_forecast(vbt)
                ferr, n_err = forecast_errors(vbt2, "IV")
                st.markdown(f"**{vix_name}를 예측에 같이 쓰면 더 정확할까?** — 같은 {n_err}번의 진입에서, 앞으로 21일 실제 변동성을 "
                            "얼마나 맞혔는지 비교")
                tcol = "GARCH 대비 개선 t값"                 # GARCH 자기 자신은 비교 대상이 없음 → '—' (빈 값은 Streamlit이 None으로 표시)
                fshow = ferr.assign(**{tcol: ferr[tcol].map(lambda v: "—" if pd.isna(v) else f"{v:+.1f}")})
                st.dataframe(fshow.style.format({"예측 오차(RMSE)": "{:.1%}", "평균 치우침": "{:+.1%}"}), width="stretch",
                             column_config={tcol: st.column_config.TextColumn(tcol, alignment="right")})
                best = ferr["예측 오차(RMSE)"].astype(float).idxmin()
                st.caption(f"가장 정확했던 예측: **{best}**. 결합 예측은 '실제 변동성 ≈ a + b·GARCH + c·IV'의 a·b·c를 "
                           f"매번 이미 결과가 나온 과거 진입들로만 다시 추정해 만들었습니다(처음 24번은 추정용이라 비교에서 제외). "
                           "평균 치우침이 +면 실제보다 높게 예측했다는 뜻이고, 개선 t값이 2를 넘어야 GARCH보다 낫다는 게 우연이 아닙니다.")

                # ---- (2) 매도만: 앞 기간으로 고르고 뒤 기간에서 검증
                so = short_only_split(vbt2, iv_name="IV")
                st.markdown("**매도만 하는 규칙 — 앞 기간으로 고르고 뒤 기간에서 검증**")
                if so is None:
                    st.caption("표본이 부족해 이 검증을 할 수 없습니다 (결합 예측이 있는 진입이 24번 이상 필요).")
                else:
                    tt = so["test"]
                    ch, al, sg2 = tt.loc["고른 규칙"], tt.loc["항상 매도"], tt.loc["GARCH 신호대로 (사이드바 설정)"]
                    (a0, a1), (b0, b1) = so["train_period"], so["test_period"]
                    tags = [("뒤 기간 이익이 우연이 아님 (t > 2)", "approve") if ch["t값"] > 2 else ("뒤 기간 이익이 우연일 수 있음", "elective")]
                    if so["chosen"] != "항상 매도":
                        tags.append(("뒤 기간에 항상 매도보다 나음", "approve") if ch["샤프(연)"] > al["샤프(연)"]
                                    else ("뒤 기간에 항상 매도보다 못함", "elective"))
                    ui.verdict(f"앞 기간 {a0.date()} ~ {a1.date()} 성적으로 고른 규칙", so["chosen"], "", tags,
                               f"뒤 기간 {b0.date()} ~ {b1.date()} ({len(so['curves'])}번)의 샤프: 고른 규칙 {ch['샤프(연)']:.2f}, "
                               f"항상 매도 {al['샤프(연)']:.2f}, GARCH 신호대로 {sg2['샤프(연)']:.2f}. "
                               "뒤 기간은 규칙을 고를 때 전혀 쓰지 않았으므로 '처음 보는 데이터'에서의 성적입니다.")
                    st.dataframe(tt.style.format(vfmt, na_rep="—"), width="stretch")
                    fig = go.Figure()
                    for name, color in zip(so["curves"].columns, (ui.BLUE, ui.AMBER, ui.NEUTRAL)):
                        fig.add_trace(go.Scatter(x=so["curves"].index, y=so["curves"][name], name=name, line=dict(color=color)))
                    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                                      yaxis_title="뒤 기간 누적 (시작=1)", legend=dict(orientation="h", y=1.12))
                    ui.chart(fig)
                    with st.expander("앞 기간 후보 규칙별 성적"):
                        st.dataframe(so["candidates"].sort_values("앞 기간 샤프", ascending=False).style.format(
                            {"앞 기간 샤프": "{:.2f}", "앞 기간 거래 수": "{:.0f}", "앞 기간 평균 손익(주가 대비)": "{:+.2%}"},
                            na_rep="—"), width="stretch")
                    st.caption(f"규칙: 'IV − 예측 변동성'이 기준보다 크면 그달 ATM 스트래들을 팔고(매일 델타헤지), 아니면 쉽니다. "
                               "후보는 예측 2가지(GARCH, 결합) × 기준 8가지와 '항상 매도'이고, 앞 기간 샤프가 가장 높은 것을 골랐습니다. "
                               "뒤 기간이 3~4년 정도라 표본이 작으니 t값을 함께 보세요.")

# ---------------- 4. 백테스트 ----------------
with tab4:
    ui.section("Backtest", "GARCH 예측은 과거에 얼마나 맞았나",
               f"과거 여러 시점에서 그때까지의 데이터만으로 {horizon}일 변동성을 다시 예측하고, 실제로 움직인 폭과 비교합니다. "
               "비교 대상은 '최근 변동성이 그대로 이어진다'는 단순 예측입니다. 범위와 VaR는 t분포 GARCH 시뮬레이션으로 판정합니다.")
    if st.button("검증 실행", type="primary"):
        with st.spinner("과거 여러 시점에서 다시 예측하는 중…"):
            bt = run_backtest(rets.values, horizon)
        if bt.empty:
            st.warning("데이터가 부족합니다. 학습 데이터 기간을 늘리거나 예측 기간을 줄이세요.")
        else:
            rmse = lambda c: np.sqrt(np.mean((bt[c] - bt["realized"]) ** 2))
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("모델 예측 오차", f"{rmse('garch'):.2%}", help="RMSE, 작을수록 정확")
            m2.metric("단순 예측 오차", f"{rmse('naive'):.2%}",
                      f"모델이 {'더 정확' if rmse('garch') < rmse('naive') else '덜 정확'}", delta_color="off")
            m3.metric("68% 범위 안에 든 비율", f"{bt['in1'].mean():.0%}", "목표 68%", delta_color="off")
            m4.metric("95% 범위 안에 든 비율", f"{bt['in2'].mean():.0%}", "목표 95%", delta_color="off")
            m5.metric("VaR 95%를 넘은 손실", f"{bt['var_breach'].mean():.0%}", "목표 5%", delta_color="off")
            dates = rets.index[bt["origin"]]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=bt["realized"], name="실제 변동성", line=dict(color=ui.INK)))
            fig.add_trace(go.Scatter(x=dates, y=bt["garch"], name="모델 예측", line=dict(color=ui.BLUE)))
            fig.add_trace(go.Scatter(x=dates, y=bt["naive"], name="단순 예측", line=dict(color=ui.NEUTRAL, dash="dot")))
            fig.update_layout(height=380, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=30, b=10),
                              hovermode="x unified", legend=dict(orientation="h", y=1.1))
            ui.chart(fig)
            st.caption(f"10거래일 간격으로 {len(bt)}번 검증했습니다. 검증 기간이 서로 겹쳐 있어 결과를 너무 믿지는 마세요.")

ui.footer("교육·연구용 고지:", "이 프로그램의 예측과 신호는 과거 데이터로 계산한 참고 자료이며 투자 조언이 아닙니다. "
          "과거 검증 결과가 미래 성과를 보장하지 않습니다.",
          "옵션 데이터(yfinance)는 늦게 들어오거나 장 마감 후엔 부정확할 수 있습니다. 옵션 가격 계산(블랙-숄즈)은 만기 전에도 행사할 수 있는 미국식 옵션에는 근사치입니다.")
