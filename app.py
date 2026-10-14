"""변동성 예측 & 트레이딩 신호 프로그램 (Streamlit).

실행:  run.bat  또는  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import norm

import alt_data as ad
import direction as dr
import kr_data
import market_data as md
import ui
import universe as uv
from quant_engine import (GarchVolForecaster, GammaScalpingEngine, VolatilityTradingLogic,
                          simulate_gbm_paths)

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


@st.cache_data(show_spinner=False)
def run_backtest(returns, h, step=10):
    """과거 여러 시점에서 '그때까지 데이터만으로' 예측 → 이후 실제와 비교 (look-ahead 없음)."""
    r = np.asarray(returns)
    start = max(250, len(r) - 500)
    rows = []
    for o in range(start, len(r) - h + 1, step):
        g = GarchVolForecaster().fit(r[max(0, o - 750):o])
        f = g.forecast_vol(h)
        real = np.sqrt(np.mean(r[o:o + h] ** 2) * 252)
        naive = np.sqrt(np.mean(r[o - max(h, 21):o] ** 2) * 252)
        move = r[o:o + h].sum() / (f * np.sqrt(h / 252))      # 표준화된 실제 움직임
        rows.append((o, f, naive, real, abs(move) <= 1, abs(move) <= 2))
    return pd.DataFrame(rows, columns=["origin", "garch", "naive", "realized", "in1", "in2"])


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
    horizon = st.slider("예측 기간 (거래일)", 5, 252, 63, help="21≈1개월, 63≈3개월")
    years = st.slider("학습 데이터 (년)", 2, 10, 3)
    dir_h = st.select_slider("방향 예측 기간 (거래일)", [5, 10, 21, 42, 63], 21, help="이 기간 뒤 주가가 지금보다 높을지 예측")
    st.subheader("신호 파라미터")
    vrp = st.number_input("VRP (변동성 위험 프리미엄, %p)", 0.0, 20.0, 2.0, 0.5) / 100
    threshold = st.number_input("관망 구간 ±threshold (%p)", 0.0, 20.0, 2.0, 0.5) / 100
    manual_iv = st.checkbox("IV 직접 입력 (옵션 데이터 없을 때)")
    iv_input = st.number_input("시장 IV (%)", 1.0, 300.0, 20.0, 0.5, disabled=not manual_iv) / 100
    st.subheader("헤지 시뮬레이션")
    cost_bps = st.number_input("거래비용 (bp)", 0.0, 50.0, 1.0, 0.5)
    n_paths = st.select_slider("경로 수", [200, 500, 1000, 2000], 500)
    st.button("🔄 새로고침", on_click=st.cache_data.clear)

hero_slot = st.empty()

try:
    with st.spinner(f"{ticker} 데이터 불러오는 중…"):
        close, spot, r, q, opt, r_src = get_market(ticker, years, horizon)
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

with hero_slot.container():
    ui.hero("변동성 예측 & 트레이딩 신호",
            "GARCH(1,1)로 앞으로 주가가 움직일 폭을 예측하고, 옵션 내재변동성(IV)과 비교해 변동성 매수·매도 신호를 냅니다. "
            "방향 예측과 코스피 종목 고르기도 함께 검증합니다.",
            [("티커", ticker), ("기준일", close.index[-1].date()), ("현재가", f"{spot:,.2f}")])

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

try:
    with st.spinner("방향 예측 모델 학습 중…"):
        p_up, pick, now_all, contrib, dir_asof, wf, ev, close_long, alt_status = get_direction(ticker, dir_h, cost_bps)
except Exception as e:
    p_up, ev, dir_err = None, None, e
# 확률 정확도(Brier)가 기본 확률보다 좋고, 적중률 우위가 통계적으로 유의할 때만 '검증 통과'
reliable = ev is not None and ev["skill"] > 0 and ev["z"] > 2

# ---------------- 요약 지표 ----------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("현재가", f"{spot:,.2f}", f"{close.pct_change().iloc[-1]:+.2%} (전일 대비)")
c2.metric(f"예측 변동성 ({horizon}일 평균, 연율)", f"{rv_h:.1%}",
          f"장기 평균 {np.sqrt(garch.long_run_var * 252):.1%}", delta_color="off")
c3.metric("시장 IV", f"{market_iv:.1%}" if market_iv is not None else "—", iv_note, delta_color="off")
c4.metric("신호", VolatilityTradingLogic.LABELS[signal] if signal is not None else "—",
          f"edge {logic.edge(rv_opt, market_iv):+.1%}" if market_iv is not None else "IV 필요", delta_color="off")
c5.metric(f"{dir_h}일 뒤 상승 확률", f"{p_up:.0%}" if p_up is not None else "—",
          ("검증 통과" if reliable else "과거 상승 비율" if pick == "base" else "신뢰도 낮음")
          if p_up is not None else "데이터 부족", delta_color="off")

tab1, tab_dir, tab_pick, tab_snap, tab2, tab3, tab4 = st.tabs(
    ["🎯 주가 예측 범위", "🧭 방향 예측", "🏆 코스피 종목 고르기", "📰 시장 심리", "📊 변동성",
     "💹 신호 & 헤지 손익", "🧪 예측 정확도 검증"])

# ---------------- 1. 주가 예측 범위 ----------------
with tab1:
    st.info("GARCH 변동성으로 '얼마나 크게 움직일 수 있는지'를 확률 범위로 보여줍니다. 중앙값은 현재가이며, "
            "오를지/내릴지는 **🧭 방향 예측** 탭에서 따로 봅니다.")
    cum_sd = np.sqrt(np.cumsum(garch.daily_var_path(horizon)))     # k일 누적 로그수익률 표준편차
    fut_idx = pd.bdate_range(close.index[-1], periods=horizon + 1)[1:]
    bands = {p: (spot * np.exp(-norm.ppf(0.5 + p / 2) * cum_sd), spot * np.exp(norm.ppf(0.5 + p / 2) * cum_sd))
             for p in (0.68, 0.95)}

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
    st.plotly_chart(fig, width="stretch")

    marks = sorted({k for k in (5, 21, 63, 126, 252) if k <= horizon} | {horizon})
    st.dataframe(pd.DataFrame({
        "기간(거래일)": marks,
        "날짜": [fut_idx[k - 1].date() for k in marks],
        "68% 하단": [bands[0.68][0][k - 1] for k in marks],
        "68% 상단": [bands[0.68][1][k - 1] for k in marks],
        "95% 하단": [bands[0.95][0][k - 1] for k in marks],
        "95% 상단": [bands[0.95][1][k - 1] for k in marks],
        "예상 변동폭(±1σ)": [f"±{s:.1%}" for s in cum_sd[np.array(marks) - 1]],
    }).style.format({c: "{:,.2f}" for c in ("68% 하단", "68% 상단", "95% 하단", "95% 상단")}),
        hide_index=True, width="stretch")

# ---------------- 방향 예측 ----------------
with tab_dir:
    if p_up is None:
        st.error(f"방향 예측을 할 수 없습니다 (데이터 부족 등): {dir_err}")
    else:
        view = "상승" if p_up > 0.5 else "하락"
        strength = abs(p_up - 0.5)
        level = ("모델 신호 없음, 과거 상승 비율" if pick == "base" else
                 "강함" if strength >= 0.15 else "보통" if strength >= 0.07 else "약함(거의 동전 던지기)")
        st.subheader(f"{dir_h}거래일 뒤 예측: **{view}** — 상승 확률 {p_up:.1%} (강도: {level}"
                     f"{'' if reliable else ' · 과거 검증 미통과'})")
        st.caption(f"기준일 {dir_asof.date()} · 최대 10년치 일봉 · "
                   f"사용 모델: **{dr.MODEL_LABELS[pick]}** (최근 3년 실전 성적이 가장 좋은 후보를 자동 선택)")
        st.caption("모델에 들어간 데이터: 가격 특징 · " + " · ".join(
            f"{k} {'✅' if v.startswith('사용') else '🔑' if v == '키 없음' else '❌'}" for k, v in alt_status.items()))
        if any(v == "키 없음" for v in alt_status.values()):
            st.caption("🔑 = API 키가 없어 꺼짐. `vol_predictor/api_keys.env` 파일에 `DART_API_KEY=…`, `ECOS_API_KEY=…`를 넣으면 켜집니다.")
        errs = {k: v for k, v in alt_status.items() if v.startswith("오류")}
        if errs:
            st.caption("⚠️ " + " / ".join(f"{k}: {v}" for k, v in errs.items()))

        if pick == "base":
            st.info("지금은 어떤 모델도 최근 실전 성적이 '과거 상승 비율을 그대로 쓰는 것'보다 좋지 않아서, "
                    "상승 확률로 **이 종목의 과거 상승 비율**을 보여주고 있습니다. 근거 없는 신호를 내지 않기 위한 안전장치입니다.")
        if reliable:
            st.success(f"과거 검증에서 기본 확률보다 확률 정확도가 높고(Brier 개선 {ev['skill']:+.1%}), "
                       f"적중률 우위도 통계적으로 유의했습니다 (z = {ev['z']:.1f}). 그래도 미래에도 그럴 거란 보장은 없습니다.")
        else:
            st.warning(f"과거 검증에서 '항상 상승'보다 **유의미하게 낫지 않았습니다** (z = {ev['z']:.1f}, "
                       f"Brier 개선 {ev['skill']:+.1%}). 참고용으로만 보세요. 공개 일봉 데이터로 주가 방향은 거의 예측되지 않는 게 정상입니다.")

        left, right = st.columns(2)
        with left:
            st.markdown("**지표별 방향 신호** (오른쪽 = 상승 쪽, 선형 모델 기준 설명용)")
            cs = contrib.sort_values()
            fig = go.Figure(go.Bar(x=cs.values, y=cs.index, orientation="h",
                                   marker_color=[ui.POS if v > 0 else ui.NEG for v in cs.values]))
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="기여도 (로그오즈)")
            st.plotly_chart(fig, width="stretch")
        if ev is not None:
            with right:
                st.markdown("**과거 성적표 — 자동 선택 모델** (각 시점에서 그때까지의 데이터로만 학습·선택)")
                m1, m2 = st.columns(2)
                m1.metric("적중률", f"{ev['hit']:.1%}")
                m2.metric("'항상 상승' 적중률", f"{ev['base']:.1%}",
                          f"차이 {ev['hit'] - ev['base']:+.1%}p", delta_color="off")
                m3, m4 = st.columns(2)
                m3.metric("Brier 점수 (낮을수록 좋음)", f"{ev['brier']:.3f}",
                          f"기준 {ev['brier_base']:.3f}", delta_color="off")
                m4.metric("검증 표본", f"{ev['n']:,}일", f"실질 독립 표본 ≈ {ev['n_eff']:.0f}", delta_color="off")

            st.markdown("**후보 모델 비교** — 모두 같은 워크포워드 조건. ⚠️ 이 표에서 사후에 1등을 고르면 과적합이 되므로, "
                        "앱은 매 시점 '그때까지의 성적'으로만 고른 **자동 선택** 결과를 씁니다.")
            st.dataframe(pd.DataFrame([{
                "모델": dr.MODEL_LABELS.get(k, "⭐ 자동 선택"),
                "오늘 상승 확률": now_all.get(k, p_up),
                "적중률": s["hit"], "'항상 상승' 대비": s["hit"] - s["base"],
                "AUC (0.5=무작위)": s["auc"], "Brier 개선": s["skill"],
            } for k, s in ev["table"].items()]).style.format({
                "오늘 상승 확률": "{:.1%}", "적중률": "{:.1%}", "'항상 상승' 대비": "{:+.1%}",
                "AUC (0.5=무작위)": "{:.3f}", "Brier 개선": "{:+.1%}"}),
                hide_index=True, width="stretch")
            share = ", ".join(f"{dr.MODEL_LABELS[k]} {v:.0%}" for k, v in ev["chosen_share"].items())
            st.caption(f"검증 기간 동안 자동 선택된 비율: {share}. "
                       "AUC는 상승/하락을 얼마나 잘 구분하는지(0.5면 동전 던지기), Brier 개선은 확률 정확도가 기본 확률보다 몇 % 좋은지입니다.")

            st.markdown(f"**'상승 확률 > 50%일 때만 보유' 전략 vs 단순 보유** (거래비용 {cost_bps}bp, "
                        f"보유 비중 {ev['time_in_market']:.0%}, 샤프 {ev['sharpe']:.2f} vs {ev['sharpe_bh']:.2f})")
            eq = ev["equity"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=eq.index, y=eq["전략"], name="모델 전략", line=dict(color=ui.BLUE)))
            fig.add_trace(go.Scatter(x=eq.index, y=eq["단순 보유"], name="단순 보유", line=dict(color=ui.NEUTRAL)))
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                              yaxis_title="누적 (시작=1)", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, width="stretch")

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
            st.plotly_chart(fig, width="stretch")

# ---------------- 코스피 종목 고르기 ----------------
with tab_pick:
    st.markdown("한 종목의 방향 대신, **코스피 대형주 약 200개를 서로 비교해 앞으로 상대적으로 더 오를 종목**을 고릅니다. "
                "(위 티커 설정과 무관하게 동작합니다.)")
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
        hold = c2.selectbox("보유 기간(리밸런싱 주기)", [63, 21], format_func=lambda x: f"{x}거래일 (≈{x // 21}개월)")
        pick_cost = c3.number_input("거래비용 (편도, bp)", 0.0, 100.0, 15.0, 5.0,
                                    help="수수료+세금+슬리피지. 국내 주식은 매도세가 있어 편도 10~25bp 정도가 현실적")
        if st.button("🔄 데이터 업데이트 (새 날짜만 받음)"):
            run_update()
        try:
            with st.spinner("워크포워드 학습·검증 중… (처음 1~3분)"):
                U = get_universe(hold, pick_cost, info["updated"])
        except Exception as e:
            st.error(f"계산 실패: {e}")
            U = None

        if U is not None:
            auto = U["table"].loc["auto"]
            ok = auto["초과수익 t값"] > 2 and auto["IC t값"] > 2
            pick_label = "고르지 않음 (모든 모델 최근 성적 ≤ 0 → 전체 동일가중 권장)" if U["pick"] == "none" \
                else uv.MODEL_LABELS[U["pick"]]
            st.subheader(f"{U['date'].date()} 기준 · 사용 모델: {pick_label}")
            if ok:
                st.success(f"과거 검증에서 자동 선택 전략이 비용을 빼고도 전체 평균보다 통계적으로 유의하게 더 올랐습니다 "
                           f"(연 {auto['상위20% 초과수익(연, 비용후)']:+.1%}, t = {auto['초과수익 t값']:.1f}; "
                           f"IC t = {auto['IC t값']:.1f}). 그래도 미래 성과를 보장하지 않습니다.")
            else:
                st.warning(f"과거 검증에서 자동 선택 전략의 초과수익이 통계적으로 유의하지 않았습니다 "
                           f"(연 {auto['상위20% 초과수익(연, 비용후)']:+.1%}, t = {auto['초과수익 t값']:.1f}; "
                           f"IC t = {auto['IC t값']:.1f}). 아래 순위는 참고용입니다.")

            cv = U["curves"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cv["auto"].index, y=cv["auto"]["상위 20% (비용 후)"], name="상위 20% (자동 선택, 비용 후)",
                                     line=dict(color=ui.BLUE, width=2.5)))
            fig.add_trace(go.Scatter(x=cv["_universe"].index, y=cv["_universe"], name="전체 동일가중",
                                     line=dict(color=ui.NEUTRAL)))
            fig.add_trace(go.Scatter(x=cv["auto"].index, y=cv["auto"]["하위 20%"], name="하위 20%",
                                     line=dict(color=ui.CORAL, dash="dot")))
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                              yaxis_title="누적 (시작=1)", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, width="stretch")

            st.markdown("**모델 비교** — 모두 같은 워크포워드 조건. ⚠️ 이 표에서 사후에 1등을 고르면 과적합이므로, "
                        "앱은 매 시점 '그때까지의 성적'으로만 고른 **자동 선택**을 씁니다.")
            st.dataframe(U["table"].style.format({
                "평균 IC": "{:+.3f}", "IC t값": "{:+.1f}", "IC>0 비율": "{:.0%}",
                "상위20% 초과수익(연, 비용후)": "{:+.1%}", "초과수익 t값": "{:+.1f}",
                "상위−하위 스프레드(연)": "{:+.1%}", "평균 회전율": "{:.0%}", "리밸런싱 수": "{:.0f}"}, na_rep="—"),
                hide_index=True, width="stretch")
            share = U["chosen"].value_counts(normalize=True)
            st.caption("자동 선택 비율: " + ", ".join(
                f"{uv.MODEL_LABELS.get(k, '고르지 않음')} {v:.0%}" for k, v in share.items()) +
                ". IC = 점수 순위와 실제 수익률 순위의 상관(0이면 무관, 0.05면 실무에선 의미 있는 수준).")

            left, right = st.columns(2)
            fmt = {c: "{:.0%}" for c in U["top"].columns if c not in ("순위", "종목명", "점수")}
            fmt["점수"] = "{:+.3f}"
            with left:
                st.markdown(f"**상위 20종목** (전체 {U['n_univ']}개 중)")
                st.dataframe(U["top"].style.format(fmt), width="stretch")
            with right:
                st.markdown("**하위 20종목**")
                st.dataframe(U["bottom"].style.format(fmt), width="stretch")
            st.caption("신호 열은 종목 간 백분위(100% = 그 신호가 가장 큰 종목)입니다. 예: 외국인 순매수 강도 90% = 상위 10% 순매수.")

            ic = U["ic"].dropna()
            fig = go.Figure(go.Bar(x=ic.index, y=ic.values,
                                   marker_color=[ui.POS if v > 0 else ui.NEG for v in ic.values]))
            fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="리밸런싱별 IC")
            st.plotly_chart(fig, width="stretch")

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
    st.info("**실적 서프라이즈 · 애널리스트 등급/목표가 · 외국인/기관 수급 · 시장 SKEW**는 과거 기록이 있어서 "
            "🧭 방향 예측 모델에 들어가 검증됩니다. 아래 **종목 옵션 풋/콜·스큐, EPS 추정치 변경, 뉴스 감성**은 "
            "무료로는 '현재 값'만 구할 수 있어 과거 검증이 불가능하므로, 모델에 넣지 않고 참고용으로만 보여줍니다.")
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
        c.metric("IV 스큐 (90% 풋 − 110% 콜)", f"{o['skew']:+.1%}" if np.isfinite(o["skew"]) else "—",
                 f"풋 {o['iv_put90']:.1%} / 콜 {o['iv_call110']:.1%}", delta_color="off")
        d.metric("ATM IV", f"{o['iv_atm']:.1%}" if np.isfinite(o["iv_atm"]) else "—", f"만기 {o['expiry']} ({o['days']}일)",
                 delta_color="off")
        st.caption("풋/콜 비율이 높을수록(보통 1 이상) 하락 대비 수요가 많다는 뜻이지만, 지나치게 높으면 역발상으로 바닥 신호로 보기도 합니다. "
                   "스큐가 클수록 시장이 하락 위험에 더 높은 값을 매기고 있다는 뜻입니다. 70일 이내 만기 합산, 장 마감 후엔 거래량이 0일 수 있습니다.")

    st.subheader("애널리스트 EPS 추정치")
    rv = snap["revisions"]
    code = ad.krx_code(ticker)
    if code:
        st.caption("한국 종목은 네이버 증권의 실적·컨센서스를 보여줍니다. **(E)** 열이 애널리스트 컨센서스 추정치입니다. "
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
        st.caption("추정치가 올라가는 중(+)이면 애널리스트들이 실적 전망을 높이고 있다는 뜻입니다. "
                   "과거 추정치 변경 이력은 무료로 없어서, 과거 검증은 등급·목표가 변경 데이터로 대신합니다.")

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
            fig.add_trace(go.Bar(x=fl.index, y=fl["inst"], name="기관 순매수(주)", marker_color=ui.MINT))
            fig.update_layout(barmode="group", height=300, margin=dict(l=10, r=10, t=30, b=10),
                              hovermode="x unified", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, width="stretch")
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
        st.plotly_chart(fig, width="stretch")
        st.dataframe(nw[["time", "score", "title", "source", "link"]], hide_index=True, width="stretch",
                     column_config={"time": st.column_config.DatetimeColumn("시각", format="MM-DD HH:mm"),
                                    "score": st.column_config.NumberColumn("감성", format="%+.2f"),
                                    "title": "헤드라인", "source": "언론사",
                                    "link": st.column_config.LinkColumn("링크", display_text="열기")})
        st.caption("감성 점수는 제목에 나온 금융 긍정/부정 단어를 세는 단순 사전 방식입니다. 반어·맥락은 이해하지 못합니다. "
                   "과거 뉴스 기록이 무료로 없어 예측력 검증이 불가능합니다.")

# ---------------- 2. 변동성 ----------------
with tab2:
    rv21 = rets.rolling(21).std() * np.sqrt(252)
    cv = pd.Series(garch.cond_vol, index=rets.index)
    fut_vol = np.sqrt(garch.daily_var_path(horizon) * 252)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rv21.index, y=rv21, name="21일 실현변동성", line=dict(color=ui.NEUTRAL)))
    fig.add_trace(go.Scatter(x=cv.index, y=cv, name="GARCH 조건부 변동성", line=dict(color=ui.BLUE)))
    fig.add_trace(go.Scatter(x=fut_idx, y=fut_vol, name="GARCH 예측 (일별)", line=dict(color=ui.MINT, dash="dash")))
    if market_iv is not None:
        fig.add_hline(y=market_iv, line=dict(color=ui.CORAL, dash="dot"), annotation_text=f"시장 IV {market_iv:.1%}")
    fig.update_layout(height=420, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified", legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")
    a, b, c, d = st.columns(4)
    a.metric("α (충격 반응)", f"{garch.alpha:.3f}")
    b.metric("β (지속성)", f"{garch.beta:.3f}")
    c.metric("α+β", f"{garch.persistence:.3f}", help="1에 가까울수록 변동성 충격이 오래 지속")
    d.metric("반감기", f"{np.log(0.5) / np.log(garch.persistence):.0f}일" if garch.persistence < 1 else "∞")

# ---------------- 3. 신호 & 헤지 손익 ----------------
with tab3:
    if market_iv is None:
        st.warning("이 종목은 옵션 데이터가 없어 신호를 낼 수 없습니다. 사이드바에서 'IV 직접 입력'을 켜세요.")
    else:
        st.markdown(f"""
| 항목 | 값 |
|---|---|
| 시장 IV ({iv_note}) | **{market_iv:.2%}** |
| GARCH 예측 RV (옵션 만기 {opt_days}거래일 평균) | {rv_opt:.2%} |
| + VRP | {vrp:.2%} |
| **edge = IV − (RV + VRP)** | **{logic.edge(rv_opt, market_iv):+.2%}** (관망 구간 ±{threshold:.1%}) |
| 무위험이자율 ({r_src}) / 배당수익률 | {r:.2%} / {q:.2%} |
""")
        st.subheader(f"신호: {VolatilityTradingLogic.LABELS[signal]}")
        position = signal if signal != 0 else st.radio(
            "관망 신호입니다. 시뮬레이션할 포지션을 고르세요:", [1, -1],
            format_func=lambda p: VolatilityTradingLogic.LABELS[p], horizontal=True)

        eng = GammaScalpingEngine(spot, spot, opt_T, r, market_iv, q)
        gk = eng.straddle_greeks()
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("ATM 스트래들 이론가", f"{gk['price']:.2f}")
        g2.metric("델타", f"{gk['delta']:+.3f}")
        g3.metric("감마", f"{gk['gamma']:.4f}")
        g4.metric("세타/일", f"{gk['theta_per_day']:+.3f}")

        st.markdown(f"**델타헤지 손익 몬테카를로** — 스트래들 1단위, 매일 헤지, 비용 {cost_bps}bp, 각 {n_paths}경로")
        rng = np.random.default_rng(0)
        scenarios = [("GARCH 예측대로", rv_opt), ("IV 수준", market_iv),
                     ("IV × 1.5 (급등장)", 1.5 * market_iv), ("IV × 0.6 (조용한 장)", 0.6 * market_iv)]
        scen_color = dict(zip([s[0] for s in scenarios], [ui.BLUE, ui.NEUTRAL, ui.CORAL, ui.MINT]))
        rows, hist = [], go.Figure()
        for name, vol in scenarios:
            paths = simulate_gbm_paths(spot, vol, r, opt_T, opt_days, n_paths, rng)
            pnl = eng.hedged_pnl(paths, position=position, cost_bps=cost_bps)
            rows.append({"시나리오": name, "실현변동성": f"{vol:.1%}", "평균 손익": pnl.mean(),
                         "승률": f"{np.mean(pnl > 0):.0%}", "최악": pnl.min(), "최선": pnl.max(),
                         "프리미엄 대비 평균": f"{pnl.mean() / gk['price']:+.1%}"})
            hist.add_trace(go.Histogram(x=pnl, name=name, opacity=0.55, nbinsx=60, marker_color=scen_color[name]))
        st.dataframe(pd.DataFrame(rows).style.format({"평균 손익": "{:+.2f}", "최악": "{:+.2f}", "최선": "{:+.2f}"}),
                     hide_index=True, width="stretch")
        hist.update_layout(barmode="overlay", height=360, xaxis_title="만기 손익 (주가 단위)",
                           margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(hist, width="stretch")
        if position == -1:
            st.warning("숏 감마는 예측이 맞으면 꾸준히 벌지만, 실현변동성이 IV를 넘으면 손실이 커집니다. 포지션 한도·손절 필수.")

# ---------------- 4. 백테스트 ----------------
with tab4:
    st.markdown(f"과거 각 시점에서 **그 시점까지의 데이터만으로** GARCH를 다시 적합해 {horizon}일 변동성을 예측하고, "
                "이후 실제로 실현된 변동성과 비교합니다. 비교 기준(naive)은 '최근 실현변동성이 그대로 간다'는 예측입니다.")
    if st.button("검증 실행", type="primary"):
        with st.spinner("여러 시점에서 GARCH 재적합 중…"):
            bt = run_backtest(rets.values, horizon)
        if bt.empty:
            st.warning("데이터가 부족합니다. 학습 데이터 기간을 늘리거나 예측 기간을 줄이세요.")
        else:
            rmse = lambda c: np.sqrt(np.mean((bt[c] - bt["realized"]) ** 2))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("GARCH 오차 (RMSE)", f"{rmse('garch'):.2%}")
            m2.metric("Naive 오차 (RMSE)", f"{rmse('naive'):.2%}",
                      f"GARCH가 {'더 정확' if rmse('garch') < rmse('naive') else '덜 정확'}", delta_color="off")
            m3.metric("68% 범위 적중률", f"{bt['in1'].mean():.0%}", "이론값 68%", delta_color="off")
            m4.metric("95% 범위 적중률", f"{bt['in2'].mean():.0%}", "이론값 95%", delta_color="off")
            dates = rets.index[bt["origin"]]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=bt["realized"], name="실제 실현변동성", line=dict(color=ui.INK)))
            fig.add_trace(go.Scatter(x=dates, y=bt["garch"], name="GARCH 예측", line=dict(color=ui.BLUE)))
            fig.add_trace(go.Scatter(x=dates, y=bt["naive"], name="Naive 예측", line=dict(color=ui.NEUTRAL, dash="dot")))
            fig.update_layout(height=380, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=30, b=10),
                              hovermode="x unified", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, width="stretch")
            st.caption(f"검증 시점 {len(bt)}개 (10거래일 간격). 기간이 겹쳐서 표본들이 독립이 아니므로 참고용입니다.")

st.divider()
st.caption("⚠️ 교육·연구용입니다. 투자 조언이 아니며, 옵션 데이터(yfinance)는 지연되거나 장 마감 후엔 부정확할 수 있습니다. "
           "블랙-숄즈는 미국식 옵션에 대한 근사입니다.")
