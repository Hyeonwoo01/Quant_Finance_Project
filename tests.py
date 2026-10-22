"""노트북의 검증(assert)들을 그대로 옮긴 자체 테스트.  실행: python tests.py"""
import numpy as np

from quant_engine import (GarchVolForecaster, GammaScalpingEngine, RiskNeutralPricingEngine,
                          VolatilityTradingLogic, _straddle_delta, simulate_gbm_paths)


def simulate_garch(n, omega, alpha, beta, rng):
    r, s2 = np.empty(n), omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    return r


# 프라이싱 / IV
engine = RiskNeutralPricingEngine(100.0, 100.0, 1.0, 0.05, 'call')
put = RiskNeutralPricingEngine(100.0, 100.0, 1.0, 0.05, 'put')
assert abs(engine.price(0.20) - 10.45) < 0.01
assert abs(engine.implied_volatility(12.00) - 0.2411) < 1e-3
assert abs(engine.price(0.2) - put.price(0.2) - (100 - 100 * np.exp(-0.05))) < 1e-10
assert np.isnan(engine.implied_volatility(101.0)) and np.isnan(engine.implied_volatility(0.5))
for K in (70, 100, 150):
    for T in (0.05, 0.5, 2.0):
        for sg in (0.08, 0.25, 0.9):
            for typ in ('call', 'put'):
                e = RiskNeutralPricingEngine(100.0, float(K), T, 0.03, typ)
                if e.vega(sg) >= 1e-3:
                    assert abs(e.implied_volatility(e.price(sg)) - sg) < 1e-6

# GARCH 파라미터 복원
g = GarchVolForecaster().fit(simulate_garch(6000, 4e-6, 0.08, 0.90, np.random.default_rng(3)))
assert abs(g.persistence - 0.98) < 0.03

# 신호
logic = VolatilityTradingLogic(vrp=0.028, threshold=0.02)
assert logic.generate_signal(0.20, 0.23) == 0
assert logic.generate_signal(0.15, 0.27) == -1
assert logic.generate_signal(0.28, 0.22) == 1

# 그릭스
e0 = GammaScalpingEngine(100.0, 100.0, 1.0, 0.05, 0.20)
h = 1e-5
v_hi = GammaScalpingEngine(100, 100, 1 + h, 0.05, 0.2).straddle_greeks()["price"]
v_lo = GammaScalpingEngine(100, 100, 1 - h, 0.05, 0.2).straddle_greeks()["price"]
assert abs(e0.straddle_greeks()["theta_per_day"] * 252 + (v_hi - v_lo) / (2 * h)) < 1e-6
assert abs(_straddle_delta(100, 100, 1.0, 0.05, 0.2) - e0.straddle_greeks()["delta"]) < 1e-12

# 벡터화된 헤지 손익 == 경로 하나씩 계산한 값, 그리고 부호가 이론대로인지
paths = simulate_gbm_paths(100, 0.30, 0.03, 0.25, 63, 50, np.random.default_rng(1))
sc = GammaScalpingEngine(100, 100, 0.25, 0.03, 0.20)
vec = sc.hedged_pnl(paths, position=1)
assert np.allclose(vec, [sc.hedged_pnl(p, position=1) for p in paths])
assert vec.mean() > 0                                      # 실현 30% > IV 20% → 롱 감마 이익
assert sc.hedged_pnl(paths, position=-1).mean() < 0

# 방향 예측: 로지스틱이 알려진 관계를 복원하는지
import pandas as pd
import direction as dr

rng = np.random.default_rng(7)
Xs = rng.standard_normal((4000, 3))
ys = (rng.random(4000) < 1 / (1 + np.exp(-(1.5 * Xs[:, 0] - 1.0 * Xs[:, 1])))).astype(float)
m = dr.LogisticDirectionModel(l2=1e-6).fit(Xs, ys)
assert abs(m.w[0] - 1.5) < 0.15 and abs(m.w[1] + 1.0) < 0.15 and abs(m.w[2]) < 0.15

# 워크포워드에 미래 정보가 새지 않는지: T 이후 가격을 바꿔도 T까지의 예측은 그대로여야 한다
px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 1500))),
               index=pd.bdate_range("2015-01-01", periods=1500))
wf_a, _ = dr.walk_forward(px, h=21)
T = 1200
px_b = px.copy()
px_b.iloc[T + 1:] *= np.exp(np.cumsum(rng.normal(0, 0.05, len(px) - T - 1)))
wf_b, _ = dr.walk_forward(px_b, h=21)
upto = wf_a.index <= px.index[T]
assert upto.sum() > 100
for col in ("auto", "base", "logit_price", "gbm", "ensemble"):     # 자동 선택 포함 모든 후보
    assert np.allclose(wf_a[col][upto], wf_b[col][upto]), col
ev = dr.evaluate(wf_a, 21, np.log(px).diff())
assert 0 <= ev["hit"] <= 1 and len(ev["equity"]) == len(wf_a)

# 파이프라인이 '실제로 예측 가능한' 데이터에선 신호를 찾아내는지 (모멘텀이 있는 합성 가격)
r = np.zeros(2500)
for t in range(2500):
    r[t] = 0.0002 + 0.03 * r[max(0, t - 21):t].sum() + 0.01 * rng.standard_normal()
px_m = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2015-01-01", periods=2500))
wf_m, _ = dr.walk_forward(px_m, h=5)
ev_m = dr.evaluate(wf_m, 5, np.log(px_m).diff())
assert ev_m["auc"] > 0.6 and ev_m["skill"] > 0.02 and ev_m["hit"] > ev_m["base"] + 0.02, ev_m["table"]["auto"]

# 대체 데이터: 발표된 정보는 '발표 다음 거래일'부터만 쓰여야 한다
import alt_data as ad

idx = pd.bdate_range("2024-01-01", periods=10)       # 1/1(월) ~ 1/12(금)
stamps = pd.DatetimeIndex(["2024-01-03 16:00", "2024-01-05 08:00", "2024-01-06 10:00"], tz="America/New_York")
pos = ad._effective_pos(idx, stamps)
assert [idx[p].date().isoformat() for p in pos] == ["2024-01-04", "2024-01-08", "2024-01-08"]  # 장후·장전·주말 발표 모두 다음 거래일

assert ad.headline_score("Apple beats estimates, shares surge") > 0
assert ad.headline_score("Stock plunges after earnings miss") < 0
assert ad.headline_score("삼성전자 호실적에 신고가") > 0 and ad.headline_score("반도체 부진 우려에 급락") < 0
assert ad.headline_score("Company holds annual meeting") == 0

# alt 특징을 넣어도 워크포워드에 미래 정보가 새지 않는지 (T 이후 alt 값만 바꿔 봄)
alt_a = pd.DataFrame({"eps_surprise": rng.normal(0, 0.1, len(px))}, index=px.index)
alt_b = alt_a.copy()
alt_b.iloc[T + 1:] = 5.0
wa, cols_a = dr.walk_forward(px, 21, alt=alt_a)
wb, _ = dr.walk_forward(px, 21, alt=alt_b)
assert "eps_surprise" in cols_a and "logit_alt" in wa
up_to = wa.index <= px.index[T]
assert np.allclose(wa["auto"][up_to], wb["auto"][up_to]) and np.allclose(wa["logit_alt"][up_to], wb["logit_alt"][up_to])

# 앙상블은 '로지스틱·전체'와 '부스팅'의 평균 — 다시 학습하지 않고 평균만 내도 정의와 같다
assert "logit_all" in wa and np.allclose(wa["ensemble"], (wa["logit_all"] + wa["gbm"]) / 2)
_c = dr.make_candidates(cols_a)
_X = dr.build_features(px, alt=alt_a).values
_y = (rng.random(len(_X)) < 0.5).astype(float)
_fp = dr.fit_predict_all(dr.make_candidates(cols_a), _X[:800], _y[:800], _X[800:900])
assert np.allclose(_fp["ensemble"], _c["ensemble"].fit(_X[:800], _y[:800]).predict_proba(_X[800:900]))
_cols_p = list(dr.build_features(px).columns)                         # 가격 특징만 ('로지스틱·전체' 후보 없음)
_Xp = dr.build_features(px).values
_cp = dr.make_candidates(_cols_p)
assert "logit_all" not in _cp
assert np.allclose(dr.fit_predict_all(dr.make_candidates(_cols_p), _Xp[:800], _y[:800], _Xp[800:900])["ensemble"],
                   _cp["ensemble"].fit(_Xp[:800], _y[:800]).predict_proba(_Xp[800:900]))

# 가정 바꿔 점검: 자동 선택만 다시 하면(재학습 없음) 같은 기간에서 워크포워드와 똑같아야 하고, 표 모양이 맞아야 한다
rs = dr.reselect(wf_a, 21, 756)
assert np.allclose(rs["auto"], wf_a["auto"]) and (rs["chosen"] == wf_a["chosen"]).all()
assert (dr.reselect(wf_a, 21, 10_000, min_scored=10_000)["chosen"] == "base").all()      # 채점 표본 부족 → 기본 확률
rb_m = dr.robustness(wf_m, 5, np.log(px_m).diff(), 1.0)
assert list(rb_m["by_window"].index) == ["최근 1년", "최근 2년", "최근 3년", "최근 5년"] and len(rb_m["by_period"]) == 3
assert rb_m["by_rule"].shape == (3, 4) and list(rb_m["by_rule"].index) == ["45%", "50%", "55%"]
assert (rb_m["by_window"]["확률 정확도 개선"] > 0.01).all()      # 심어 둔 모멘텀은 선택 기간과 상관없이 잡혀야
assert not dr.robustness(wf_a, 21, np.log(px).diff())["by_window"]["검증 통과"].any()   # 순수 무작위 가격은 통과 못 해야
assert dr.evaluate(wf_a, 21, np.log(px).diff(), threshold=1.0)["time_in_market"] == 0.0
assert "gbm" not in dr.make_candidates(cols_a, light=True) and "logit_all" in dr.make_candidates(cols_a, light=True)
wl, _ = dr.walk_forward(px, 21, light=True)
assert "gbm" not in wl and "ensemble" not in wl and np.allclose(wl["logit_price"], wf_a["logit_price"])   # 가벼운 점검도 같은 로지스틱
hz = dr.horizon_check(px_m, horizons=(5, 21))                  # 가벼운 점검(로지스틱)도 심어 둔 신호를 잡아야
assert list(hz.index) == ["5거래일", "21거래일"] and hz.loc["5거래일", "확률 정확도 개선"] > 0
assert "부스팅·앙상블 선택 비율" not in hz
hzf = dr.horizon_check(px_m, horizons=(21,), light=False, refit_every=252)        # 부스팅 포함: 선택 비율 열이 붙어야
assert 0 <= hzf.loc["21거래일", "부스팅·앙상블 선택 비율"] <= 1

# DART 공시: 분류 + '접수일 다음 거래일'부터 반영되는지 (네트워크 없이 공시 목록을 가짜로 주입)
import kr_data

assert kr_data.classify("주요사항보고서(자기주식취득결정)") == "dart_return"
assert kr_data.classify("주요사항보고서(유상증자결정)") == "dart_dilution"
assert kr_data.classify("[기재정정]주요사항보고서(유상증자결정)") is None        # 정정은 세지 않음
assert kr_data.classify("분기보고서 (2026.06)") == "other"
fake = pd.DataFrame({"date": pd.to_datetime(["2024-01-03", "2024-01-05"]),
                     "report": ["주요사항보고서(자기주식취득결정)", "단일판매ㆍ공급계약체결"], "rcept_no": ["1", "2"]})
orig = kr_data.dart_disclosures
kr_data.dart_disclosures = lambda code, start, **kw: fake
try:
    f = kr_data.dart_features("000000", idx)
finally:
    kr_data.dart_disclosures = orig
assert f.loc["2024-01-03", "dart_return"] == 0 and f.loc["2024-01-04", "dart_return"] == 1   # 1/3 접수 → 1/4부터
assert f.loc["2024-01-05", "dart_contract"] == 0 and f.loc["2024-01-08", "dart_contract"] == 1
assert f.loc["2024-01-08", "dart_total"] == 2

# 종목 고르기: 심어 둔 '종목 간' 신호를 찾는지 + T 이후 가격을 바꿔도 T까지의 점수가 그대로인지
import universe as uv

def synth_panel(seed, n_stock=80, n_day=800, shock_after=None):
    g = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_day)
    sig = np.zeros((n_day, n_stock))
    for t in range(1, n_day):
        sig[t] = 0.97 * sig[t - 1] + 0.25 * g.standard_normal(n_stock)       # 천천히 변하는 종목별 신호
    ret = 0.0015 * sig + 0.02 * g.standard_normal((n_day, n_stock)) + 0.01 * g.standard_normal((n_day, 1))
    if shock_after is not None:
        ret[shock_after + 1:] += 0.05 * np.random.default_rng(99).standard_normal((n_day - shock_after - 1, n_stock))
    codes = [f"{i:06d}" for i in range(n_stock)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(ret, axis=0)), index=dates, columns=codes)
    F = {"mom_12_1": pd.DataFrame(np.random.default_rng(seed + 1).standard_normal((n_day, n_stock)), index=dates, columns=codes),
         "frgn_flow_21": pd.DataFrame(sig, index=dates, columns=codes)}
    univ = pd.DataFrame(np.repeat(np.arange(n_day)[:, None] >= 260, n_stock, axis=1), index=dates, columns=codes)
    return {"feat": F, "close": close, "universe": univ, "names": pd.Series(codes, index=codes),
            "coverage": {"mom_12_1": 1.0, "frgn_flow_21": 1.0}}

pa = synth_panel(5)
wf_u = uv.walk_forward(pa, h=21, train_years=1, refit_models=("ridge",))
tab, _ = uv.evaluate(wf_u, cost_bps=0)
assert tab.loc["ridge", "평균 IC"] > 0.1 and tab.loc["ridge", "상위20% 초과수익(연, 비용후)"] > 0, tab
assert tab.loc["composite", "평균 IC"] > 0.05            # 고정 가중도 (외국인 수급 + 방향) 신호를 반영
Tcut = 700                                                # 첫 리밸런싱(512일) 이후 리밸런싱 9회가 비교됨
pb = synth_panel(5, shock_after=Tcut)
wf_b = uv.walk_forward(pb, h=21, train_years=1, refit_models=("ridge",))
cut = pa["close"].index[Tcut]
for m in ("ridge", "composite", "auto"):
    sa, sb = wf_u["scores"][m], wf_b["scores"][m]
    rows = sa.index[sa.index <= cut]
    assert len(rows) > 5 and np.allclose(sa.loc[rows].values, sb.loc[rows].values, equal_nan=True), m

# 거시 지표 시차: 명시 > 티커로 자동 판단 > 모르면 안전하게 하루 늦춤
assert dr.resolve_lag(None, "005930.KS") and dr.resolve_lag(None, "247540.KQ")
assert not dr.resolve_lag(None, "AAPL") and not dr.resolve_lag(None, "SPY")
assert dr.resolve_lag(None, None) and not dr.resolve_lag(False, "005930.KS")
mi = pd.bdate_range("2019-01-01", periods=700)
g2 = np.random.default_rng(11)
macro_t = pd.DataFrame({k: 50 + np.cumsum(g2.normal(0, 1, len(mi))) for k in
                        ("vix", "vix3m", "tnx", "irx", "hyg", "ief", "market")}, index=mi).abs() + 10
px_t = pd.Series(100 * np.exp(np.cumsum(g2.normal(0, 0.01, len(mi)))), index=mi)
f_kr = dr.build_features(px_t, macro_t, ticker="005930.KS")
f_us = dr.build_features(px_t, macro_t, ticker="AAPL")
d0 = f_kr.index[-1]
assert f_kr.loc[d0, "vix"] == np.log(macro_t["vix"]).loc[:d0].iloc[-2]      # 한국: 전날 미국 값
assert f_us.loc[d0, "vix"] == np.log(macro_t["vix"]).loc[d0]                # 미국: 같은 날 값
assert f_kr.equals(dr.build_features(px_t, macro_t, lag_macro=True))

# 종목명 검색: 정확히 일치 > 앞부분 > 포함, 같은 순위면 시가총액 순 / 코드·티커 입력은 그대로
import market_data as md
_fake = pd.DataFrame({"name": ["삼성전자", "삼성전자우", "삼성바이오로직스", "에코프로비엠", "에코프로", "NAVER"],
                      "ticker": ["005930.KS", "005935.KS", "207940.KS", "247540.KQ", "086520.KQ", "035420.KS"],
                      "market": ["코스피", "코스피", "코스피", "코스닥", "코스닥", "코스피"],
                      "cap": [9e14, 5e13, 7e13, 2e13, 1e13, 3e13]})
_orig_sym = md.kr_symbols
md.kr_symbols = lambda *a, **k: _fake
try:
    assert [t for t, _ in md.search_ticker("삼성전자")] == ["005930.KS", "005935.KS"]
    assert [t for t, _ in md.search_ticker("삼성")][:2] == ["005930.KS", "207940.KS"]     # 앞부분 일치, 시총 순
    assert [t for t, _ in md.search_ticker("에코프로")] == ["086520.KQ", "247540.KQ"]      # 정확히 일치가 먼저
    assert md.search_ticker("247540")[0][0] == "247540.KQ"                                # 코드만 → 시장 찾아 붙임
    assert md.search_ticker("005930.ks")[0][0] == "005930.KS"
    assert md.search_ticker("SPY") == [("SPY", "SPY")]
    assert md.search_ticker("naver")[0][0] == "035420.KS"                                 # 영문 이름 한국 종목
    assert md.search_ticker("없는회사") == [] and md.search_ticker("  ") == []
finally:
    md.kr_symbols = _orig_sym

# 가치 지표: 4분기 = 연간 − (1~3분기), 최근 4분기 합(TTM), 공시 다음 거래일부터 반영
fin = pd.DataFrame([
    {"code": "000010", "year": 2023, "q": 1, "rcept_dt": pd.Timestamp("2023-05-15"), "equity": 100.0, "ni": 10.0},
    {"code": "000010", "year": 2023, "q": 2, "rcept_dt": pd.Timestamp("2023-08-14"), "equity": 100.0, "ni": 10.0},
    {"code": "000010", "year": 2023, "q": 3, "rcept_dt": pd.Timestamp("2023-11-14"), "equity": 100.0, "ni": 10.0},
    {"code": "000010", "year": 2023, "q": 4, "rcept_dt": pd.Timestamp("2024-03-15"), "equity": 120.0, "ni": 50.0},
    {"code": "000010", "year": 2024, "q": 1, "rcept_dt": pd.Timestamp("2024-05-15"), "equity": 125.0, "ni": 5.0},
])
vidx = pd.bdate_range("2023-01-02", "2024-06-28")
vclose = pd.DataFrame({"000010": 10.0}, index=vidx)
v = kr_data.value_features(fin, vidx, vclose, pd.Series({"000010": 10.0}))    # 시총 = 100
assert np.isnan(v["ep"].loc["2024-03-15", "000010"])                  # 사업보고서 접수 당일엔 아직 4분기 미확정
assert abs(v["ep"].loc["2024-03-18", "000010"] - 0.50) < 1e-12         # 10+10+10+(50−30) = 50
assert abs(v["bm"].loc["2024-03-18", "000010"] - 1.20) < 1e-12
assert abs(v["ep"].loc["2024-05-16", "000010"] - 0.45) < 1e-12         # 10+10+20+5 = 45
assert abs(v["roe"].loc["2024-05-16", "000010"] - 45 / 125) < 1e-12
assert v["bm"].loc["2023-05-15", "000010"] != v["bm"].loc["2023-05-15", "000010"]   # 첫 공시 전엔 NaN

# ---------------- t분포 GARCH · 위험 지표 ----------------
from quant_engine import GammaScalpingEngine as _GSE, max_drawdown, summarize_vol_backtest, var_es, vol_signal_backtest
import regime as rg


def simulate_garch_t(n, omega, alpha, beta, nu, rng):
    r, s2 = np.empty(n), omega / (1 - alpha - beta)
    for i in range(n):
        r[i] = np.sqrt(s2) * rng.standard_t(nu) * np.sqrt((nu - 2) / nu)
        s2 = omega + alpha * r[i] ** 2 + beta * s2
    return r


gt = GarchVolForecaster().fit(simulate_garch_t(6000, 4e-6, 0.08, 0.90, 5.0, np.random.default_rng(11)))
assert 3.5 < gt.nu < 7.5 and abs(gt.persistence - 0.98) < 0.03, (gt.nu, gt.persistence)
assert GarchVolForecaster(dist="normal").fit(simulate_garch(3000, 4e-6, 0.08, 0.90, np.random.default_rng(3))).nu == np.inf
sim = gt.simulate(21, 20000, seed=1)
assert sim.shape == (20000, 21)
assert abs(sim[:, -1].std() - np.sqrt(gt.daily_var_path(21).sum())) / sim[:, -1].std() < 0.05   # 분산 경로와 일치
gn_ = GarchVolForecaster(dist="normal").fit(simulate_garch(3000, 4e-6, 0.08, 0.90, np.random.default_rng(3)))
z1 = gn_.simulate(1, 200000, seed=2)[:, 0]
assert abs(np.quantile(z1, 0.01) / np.sqrt(gn_.sigma2_next) + 2.326) < 0.03                  # 정규면 1% 분위수 = −2.326σ
assert np.quantile(gt.simulate(1, 200000, seed=2)[:, 0], 0.001) / np.sqrt(gt.sigma2_next) < -3.3   # t는 꼬리가 더 두껍다

v95, es95 = var_es(np.random.default_rng(0).normal(0, 0.01, 400000), 0.95)
assert abs(v95 - (1 - np.exp(-1.645 * 0.01))) < 3e-4 and es95 > v95
assert max_drawdown([1.0, 1.2, 0.9, 1.3, 1.17]) == -0.25 and max_drawdown([1, 2, 3]) == 0.0

# 헤지 손익: 매도 = −(매수 손익) − 거래비용 (비용은 방향과 무관)
_e = _GSE(100, 100, 21 / 252, 0.02, 0.2, 0.01)
_p = simulate_gbm_paths(100, 0.25, 0.02, 21 / 252, 21, 1, np.random.default_rng(5))[0]
_g, _n = _e.hedged_pnl(_p, 1, 0.0), _e.hedged_pnl(_p, 1, 5.0)
assert abs(_e.hedged_pnl(_p, -1, 5.0) - (-_g - (_g - _n))) < 1e-10

# 여러 진입을 한 번에 계산하는 헤지 손익 == 진입 하나씩 엔진으로 계산한 값 (만기·IV가 진입마다 달라도)
from quant_engine import hedged_straddle_pnl, straddle_price
_rng = np.random.default_rng(8)
_P = simulate_gbm_paths(100, 0.3, 0.02, 30 / 365, 21, 6, _rng) * _rng.uniform(0.5, 2.0, (6, 1))
_T, _S = _rng.uniform(25, 35, 6) / 365, _rng.uniform(0.1, 0.5, 6)
for c in (0.0, 3.0):
    _one = [GammaScalpingEngine(p[0], p[0], t, 0.02, sg, 0.01).hedged_pnl(p, 1, c) for p, t, sg in zip(_P, _T, _S)]
    assert np.allclose(hedged_straddle_pnl(_P, _T, 0.02, _S, 0.01, c), _one)
assert np.allclose(straddle_price(_P[:, 0], _P[:, 0], _T, 0.02, _S, 0.01),
                   [GammaScalpingEngine(p[0], p[0], t, 0.02, sg, 0.01).straddle_greeks()["price"] for p, t, sg in zip(_P, _T, _S)])

# 변동성 신호 백테스트: IV가 실제보다 늘 10%p 비싸면 '매도' 신호가 나고 이익이어야 한다 + 미래 정보 누출 없음
_rng = np.random.default_rng(21)
_idx = pd.bdate_range("2015-01-01", periods=1400)
_px = pd.Series(100 * np.exp(np.cumsum(0.2 / np.sqrt(252) * _rng.standard_normal(1400))), index=_idx)
_iv = pd.Series(0.30, index=_idx)
vb = vol_signal_backtest(_px, _iv, vrp=0.0, threshold=0.02, h=21, cost_bps=1.0)
vtab, vcur = summarize_vol_backtest(vb)
assert (vb["signal"] == -1).mean() > 0.9 and vtab.loc["signal", "평균 손익(주가 대비)"] > 0
assert vtab.loc["long", "평균 손익(주가 대비)"] < 0 and len(vcur) == len(vb)
_px2 = _px.copy()
_px2.iloc[1000:] *= np.exp(np.cumsum(0.05 * _rng.standard_normal(400)))      # T=1000 이후 가격만 바꿈
vb2 = vol_signal_backtest(_px2, _iv, vrp=0.0, threshold=0.02, h=21, cost_bps=1.0)
done = [d for i, d in enumerate(vb.index) if _idx.get_loc(d) + 21 < 1000]      # 결과까지 T 이전에 끝난 거래
assert len(done) > 10 and np.allclose(vb.loc[done].values, vb2.loc[done].values)

# 현실성 옵션: 옵션 체결 비용 = 비율 × 스트래들 가격, IV를 낮추면 매도 이익이 줄어듦, GARCH 예측 재사용, 진입일 늦추기
from quant_engine import vol_backtest_robustness
_c1 = vol_signal_backtest(_px, _iv, cost_bps=1.0, option_spread=0.01)
assert np.allclose(_c1["cost"] - vb["cost"], 0.01 * vb["premium"]) and np.allclose(_c1["long"], vb["long"])
_lo = vol_signal_backtest(_px, _iv, cost_bps=1.0, iv_gap=0.02, forecasts=vb["garch"].to_dict())
assert np.allclose(_lo["garch"], vb["garch"]) and np.allclose(_lo["iv"], 0.28) and np.allclose(_lo["iv_index"], 0.30)
assert (-_lo["long"]).mean() < (-vb["long"]).mean() and (_lo["premium"] < vb["premium"]).all()
_late = vol_signal_backtest(_px, _iv, cost_bps=1.0, offset=5)
assert _late.index[0] == _idx[505] and vb.index[0] == _idx[500]
rb = vol_backtest_robustness(_px, _iv, vb, cost_bps=1.0, gaps=(0.0, 0.02), offsets=(0, 5))
gtab, otab = rb["by_gap"], rb["by_offset"]
assert list(gtab.index) == ["0%p", "2%p"] and gtab.loc["0%p", "진입 수"] == len(vb)
assert set(rb["grid"]) == {(0.0, 0), (0.02, 0), (0.0, 5), (0.02, 5)} and rb["grid"][(0.0, 0)] is vb
assert np.allclose(rb["grid"][(0.02, 5)]["garch"], rb["grid"][(0.0, 5)]["garch"])        # 같은 진입일이면 예측 재사용
assert gtab.loc["2%p", "항상 매도 평균 손익(주가 대비)"] < gtab.loc["0%p", "항상 매도 평균 손익(주가 대비)"]
assert otab.loc["5거래일 늦게", "첫 진입일"] == _idx[505].date()

# IV 결합 예측: 과거 결과만 쓰는지 + 실제가 GARCH·IV의 섞임이면 결합이 둘 다보다 정확해야
from quant_engine import add_combined_forecast, forecast_errors, short_only_split
_rng = np.random.default_rng(31)
_k = 120
_g = 0.15 + 0.05 * _rng.standard_normal(_k)
_ivv = 0.18 + 0.05 * _rng.standard_normal(_k)
_bt = pd.DataFrame({"garch": _g, "iv": _ivv, "realized": 0.5 * _g + 0.5 * _ivv - 0.02 + 0.01 * _rng.standard_normal(_k),
                    "signal": 0, "long": 0.0, "cost": 0.0, "premium": 0.05},
                   index=pd.bdate_range("2015-01-01", periods=_k, freq="21B"))
cb = add_combined_forecast(_bt)
assert cb["combo"].iloc[:24].isna().all() and cb["combo"].iloc[24:].notna().all()
_bt2 = _bt.copy()
_bt2.iloc[80:, _bt2.columns.get_loc("realized")] = 9.9             # 80번째 이후 결과만 바꿈
cb2 = add_combined_forecast(_bt2)
assert np.allclose(cb["combo"].iloc[24:81], cb2["combo"].iloc[24:81]) and np.allclose(cb["iv_adj"].iloc[24:81], cb2["iv_adj"].iloc[24:81])
fe, n_fe = forecast_errors(cb, "VIX")
assert n_fe == _k - 24 and fe["예측 오차(RMSE)"].idxmin() == "GARCH + VIX 결합", fe
assert fe.loc["GARCH + VIX 결합", "GARCH 대비 개선 t값"] > 2 and np.isnan(fe.loc["GARCH", "GARCH 대비 개선 t값"])

# '매도만' 규칙: IV가 예측보다 비쌀 때만 팔면 이익인 데이터에서 그 규칙을 골라야 하고, 뒤 기간은 선택에 안 쓰여야 한다
_spread = _rng.uniform(-0.04, 0.06, _k)
_bs = pd.DataFrame({"garch": _g, "iv": _g + _spread, "realized": _g + 0.005 * _rng.standard_normal(_k),
                    "signal": 0, "cost": 0.0, "premium": 0.05}, index=_bt.index)
_bs["long"] = 0.5 * (_bs["realized"] - _bs["iv"])                  # 롱 스트래들 손익 ≈ 베가 × (실제 − IV)
so = short_only_split(add_combined_forecast(_bs), iv_name="VIX")
assert so["chosen"] != "항상 매도" and so["chosen"].endswith(("기준 -1%p", "기준 +0%p", "기준 +1%p")), so["chosen"]   # 참 기준선 ≈ 0
assert so["test"].loc["고른 규칙", "샤프(연)"] > so["test"].loc["항상 매도", "샤프(연)"]
_bs2 = _bs.copy()
_half = 24 + (_k - 24) // 2
_bs2.iloc[_half:, _bs2.columns.get_loc("long")] *= -1                # 뒤 기간 손익만 뒤집음
assert short_only_split(add_combined_forecast(_bs2), iv_name="VIX")["chosen"] == so["chosen"]
assert short_only_split(add_combined_forecast(_bs.iloc[:40]), iv_name="VIX") is None   # 표본 부족

# '매도만' 견고성: 모든 조합에서 통하는 데이터면 고정 규칙이 늘 항상 매도보다 낫고, 뒤 기간을 뒤집으면 못해야 한다
from quant_engine import short_only_robustness
rob = short_only_robustness({(0.0, 0): _bs, (0.01, 0): _bs, (0.0, 4): _bs.iloc[2:]}, rule=so["chosen"], iv_name="VIX")
assert len(rob) == 3 and (rob["이 규칙 샤프"] > rob["항상 매도 샤프"]).all() and (rob["다시 고른 규칙"] != "항상 매도").all()
rob2 = short_only_robustness({(0.0, 0): _bs2}, rule=so["chosen"], iv_name="VIX")
assert (rob2["이 규칙 샤프"] < rob2["항상 매도 샤프"]).all()
assert "이 규칙 샤프" not in short_only_robustness({(0.0, 0): _bs}, rule=None, iv_name="VIX")

# ---------------- 시장 국면 ----------------
_rng = np.random.default_rng(7)
_n = 3000
_s = np.zeros(_n, int)
for t in range(1, _n):
    _s[t] = _rng.random() < (0.01 if _s[t - 1] == 0 else 0.97)
_x = np.array([0.0006, -0.001])[_s] + np.array([0.006, 0.02])[_s] * _rng.standard_normal(_n)
m = rg.RegimeSwitchingModel().fit(_x)
assert abs(m.sd[0] - 0.006) < 0.001 and abs(m.sd[1] - 0.02) < 0.003 and abs(m.P[0, 0] - 0.99) < 0.01, (m.sd, m.P)
assert ((m.filter(_x) >= 0.5) == _s).mean() > 0.9
# 필터 확률은 그날까지의 데이터만 쓴다: 뒤쪽 수익률을 바꿔도 앞쪽 확률은 그대로
_x2 = _x.copy()
_x2[2000:] = _rng.standard_normal(1000) * 0.05
assert np.allclose(m.filter(_x)[:2000], m.filter(_x2)[:2000])
# 워크포워드: T 이후 가격을 바꿔도 T까지의 확률은 그대로
_c = pd.Series(100 * np.exp(np.cumsum(_x)), index=pd.bdate_range("2012-01-02", periods=_n))
_c2 = _c.copy()
_c2.iloc[2100:] = _c.iloc[2100] * np.exp(np.cumsum(_x2[2100:]))
pa, _ = rg.walk_forward(_c)
pb, _ = rg.walk_forward(_c2)
assert pa.iloc[:756].isna().all() and np.allclose(pa.iloc[756:2099], pb.iloc[756:2099])
out = rg.analyze(_c, 1.0)
assert out["table"].loc["불안 국면엔 현금", "최대 낙폭"] > out["table"].loc["단순 보유", "최대 낙폭"]   # 심어 둔 국면이면 낙폭이 줄어야
print("모든 테스트 통과")
