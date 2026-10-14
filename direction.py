"""주가 방향(상승/하락) 예측: 가격·거시 특징 + 여러 후보 모델 + 워크포워드 자동 선택.

정확도 개선의 원칙
1. 정보 추가: 종목 가격 특징 외에 시장 전체 상태(VIX·기간구조·금리곡선·신용·지수 추세).
2. 후보 모델 여러 개: 로지스틱(가격/전체), 그래디언트 부스팅, 앙상블, 그리고 '기본 확률(base rate)'.
3. 자동 선택은 '그 시점까지 이미 채점된 과거 예측 성적'만 보고 고른다 (중첩 워크포워드).
   → 결과 표에서 사후에 제일 좋은 모델을 고르는 것과 달리 과적합된 숫자가 아니다.
   → 아무 모델도 기본 확률보다 못하면 기본 확률을 고르므로 '항상 상승'보다 크게 나빠지지 않는다.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import alt_data

# 이 환경(sklearn 1.9 + Python 3.15)에서 학습마다 나오는 무해한 경고 — 로그가 넘치지 않게 끈다
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

PRICE_FEATURES = {
    "mom_5": "1주 모멘텀", "mom_21": "1개월 모멘텀", "mom_63": "3개월 모멘텀", "mom_252": "1년 모멘텀",
    "vol_21": "1개월 변동성", "vol_ratio": "변동성 변화(21/63일)",
    "ma50_gap": "50일선 이격도", "ma200_gap": "200일선 이격도", "rsi14": "RSI(14)",
}
MACRO_FEATURES = {
    "vix": "VIX 수준", "vix_term": "VIX 기간구조(VIX/VIX3M)", "vix_chg": "VIX 1개월 변화",
    "curve": "금리곡선(10년−3개월)", "credit": "신용 스프레드 변화(HYG/IEF)",
    "mkt_mom_63": "지수 3개월 모멘텀", "mkt_ma200_gap": "지수 200일선 이격도", "rel_strength": "지수 대비 상대강도",
    "skew_idx": "CBOE SKEW (꼬리위험 수요)", "skew_chg": "SKEW 1개월 변화", "vvix": "VVIX (변동성의 변동성)",
    # 한국 종목 전용
    "fx_chg_21": "원/달러 1개월 변화(원화 약세 +)", "fx_vol_21": "원/달러 1개월 변동성",
    "kq_ks_rel": "코스닥−코스피 1개월 상대강도(위험선호)",
    "kr_curve": "국고채 금리곡선(10년−3년)", "kr_credit": "회사채 신용스프레드(AA−−국고3년)",
    "kr3y_chg": "국고3년 금리 3개월 변화", "kr_credit_chg": "신용스프레드 1개월 변화",
}
ALT_FEATURES = alt_data.ALT_FEATURES
FEATURE_LABELS = {**PRICE_FEATURES, **MACRO_FEATURES, **ALT_FEATURES}
MODEL_LABELS = {"base": "기본 확률 (항상 과거 상승비율)", "logit_price": "로지스틱 · 가격 특징",
                "logit_alt": "로지스틱 · 실적/애널리스트/수급만", "logit_all": "로지스틱 · 전체 특징",
                "gbm": "그래디언트 부스팅 · 전체", "ensemble": "앙상블 (로지스틱+부스팅)"}


# ---------------- 특징 ----------------
def build_features(close, macro=None, lag_macro=False, alt=None):
    """t일 종가까지의 정보만 쓰는 특징들 (미래 정보 없음).

    macro: market_data.load_macro()의 DataFrame. lag_macro=True면 하루 늦춰 붙인다
    (한국 종목처럼 미국 지표보다 먼저 장이 끝나는 경우, 같은 날짜 미국 값은 아직 모르는 정보).
    alt: alt_data.build_alt_features()의 DataFrame (이미 공개 시점에 맞춰져 있음).
    """
    lc = np.log(close)
    r = lc.diff()
    f = pd.DataFrame(index=close.index)
    for k in (5, 21, 63, 252):
        f[f"mom_{k}"] = lc - lc.shift(k)
    f["vol_21"] = r.rolling(21).std() * np.sqrt(252)
    f["vol_ratio"] = r.rolling(21).std() / r.rolling(63).std()
    f["ma50_gap"] = lc - lc.rolling(50).mean()
    f["ma200_gap"] = lc - lc.rolling(200).mean()
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    if macro is not None and not macro.empty:
        m = pd.DataFrame(index=macro.index)
        m["vix"] = np.log(macro["vix"])
        m["vix_term"] = macro["vix"] / macro["vix3m"]
        m["vix_chg"] = np.log(macro["vix"] / macro["vix"].shift(21))
        m["curve"] = macro["tnx"] - macro["irx"]
        cr = np.log(macro["hyg"] / macro["ief"])
        m["credit"] = cr - cr.shift(63)
        lm = np.log(macro["market"])
        m["mkt_mom_63"] = lm - lm.shift(63)
        m["mkt_ma200_gap"] = lm - lm.rolling(200).mean()
        if "skew" in macro:          # 옵션 시장 전체의 꼬리위험 가격 (종목별 과거 풋/콜·스큐는 무료로 없음)
            m["skew_idx"] = macro["skew"]
            m["skew_chg"] = macro["skew"] - macro["skew"].shift(21)
            m["vvix"] = np.log(macro["vvix"])
        if "usdkrw" in macro:        # 외국인 수급은 환율과 함께 움직이는 경향
            lfx = np.log(macro["usdkrw"])
            m["fx_chg_21"] = lfx - lfx.shift(21)
            m["fx_vol_21"] = lfx.diff().rolling(21).std() * np.sqrt(252)
        if "kosdaq" in macro and "kospi" in macro:
            lq, lk = np.log(macro["kosdaq"]), np.log(macro["kospi"])
            m["kq_ks_rel"] = (lq - lq.shift(21)) - (lk - lk.shift(21))
        if "kr3y" in macro and macro["kr3y"].notna().any():      # 한국은행 ECOS (키가 있을 때)
            m["kr_curve"] = macro["kr10y"] - macro["kr3y"]
            m["kr_credit"] = macro["krAA"] - macro["kr3y"]
            m["kr3y_chg"] = macro["kr3y"] - macro["kr3y"].shift(63)
            m["kr_credit_chg"] = m["kr_credit"] - m["kr_credit"].shift(21)
        if lag_macro:
            m = m.shift(1)
        m = m.reindex(f.index, method="ffill", limit=5)
        f = f.join(m)
        f["rel_strength"] = f["mom_63"] - f["mkt_mom_63"]
    if alt is not None and not alt.empty:
        f = f.join(alt.reindex(f.index))
    f = f.replace([np.inf, -np.inf], np.nan)
    # 데이터가 끝까지 비어 있는 열(이 종목엔 해당 없음)은 버리고, 나머지는 모든 값이 있는 날만 쓴다
    return f.loc[:, f.notna().any()].dropna()


def forward_log_return(close, h):
    """t일 → t+h일 로그수익률 (마지막 h일은 아직 모름 → NaN)."""
    return np.log(close.shift(-h) / close)


# ---------------- 모델 ----------------
class LogisticDirectionModel:
    """표준화 + L2 정규화 로지스틱 회귀."""

    def __init__(self, l2=1.0):
        self.l2 = l2
        self.mu = self.sd = self.w = None
        self.b = 0.0

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        Z = (X - self.mu) / self.sd
        n, k = Z.shape

        def loss(p):
            w, b = p[:k], p[k]
            z = Z @ w + b
            nll = np.sum(np.logaddexp(0, z) - y * z) / n + 0.5 * self.l2 * w @ w / n
            pr = expit(z)
            g = np.append(Z.T @ (pr - y) / n + self.l2 * w / n, np.mean(pr - y))
            return nll, g

        res = minimize(loss, np.zeros(k + 1), jac=True, method="L-BFGS-B")
        self.w, self.b = res.x[:k], res.x[k]
        return self

    def standardize(self, X):
        return (np.asarray(X, float) - self.mu) / self.sd

    def predict_proba(self, X):
        return expit(self.standardize(X) @ self.w + self.b)

    def contributions(self, x_row):
        """각 특징이 현재 예측에 기여한 정도(로그오즈 단위)."""
        return self.standardize(np.atleast_2d(x_row))[0] * self.w


class _Base:
    def fit(self, X, y):
        self.p = float(np.clip(np.mean(y), 0.01, 0.99))
        return self

    def predict_proba(self, X):
        return np.full(len(X), self.p)


class _Logit:
    def __init__(self, cols, l2):
        self.cols, self.m = cols, LogisticDirectionModel(l2)

    def fit(self, X, y):
        self.m.fit(X[:, self.cols], y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X[:, self.cols])


class _GBM:
    def fit(self, X, y):
        # 겹치는 h일 라벨 때문에 표본이 실제보다 적다 → 얕은 트리 + 큰 잎 + 강한 정규화로 과적합 억제
        self.m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=150,
                                                min_samples_leaf=150, l2_regularization=1.0, random_state=0)
        self.m.fit(X, y.astype(int))
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)[:, 1]


class _Ensemble:
    def __init__(self, parts):
        self.parts = parts

    def fit(self, X, y):
        for p in self.parts:
            p.fit(X, y)
        return self

    def predict_proba(self, X):
        return np.mean([p.predict_proba(X) for p in self.parts], axis=0)


def make_candidates(columns):
    price_idx = [i for i, c in enumerate(columns) if c in PRICE_FEATURES]
    alt_idx = [i for i, c in enumerate(columns) if c in ALT_FEATURES]
    all_idx = list(range(len(columns)))
    cands = {"base": _Base(), "logit_price": _Logit(price_idx, 1.0)}
    if alt_idx:                                 # 새 데이터만으로 신호가 있는지 따로 확인
        cands["logit_alt"] = _Logit(alt_idx, 1.0)
    if len(all_idx) > len(price_idx):           # 거시/대체 데이터가 있을 때만
        cands["logit_all"] = _Logit(all_idx, 10.0)
    cands["gbm"] = _GBM()
    cands["ensemble"] = _Ensemble([_Logit(all_idx, 10.0), _GBM()])
    return cands


def _logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ---------------- 워크포워드 ----------------
def walk_forward(close, h, macro=None, lag_macro=False, alt=None, min_train=500, refit_every=21,
                 select_window=756, min_scored=252):
    """각 시점 t에서 't까지 알 수 있던 데이터'로만 모든 후보를 학습하고 예측한다.

    'auto' 열: 블록 시작 t에서, 결과가 이미 확정된(행 ≤ t−h) 과거 예측들 중 최근 select_window개의
    로그손실이 가장 낮은 후보를 고른다. 채점된 예측이 min_scored개 미만이면 기본 확률을 쓴다.
    """
    feat = build_features(close, macro, lag_macro, alt)
    fwd = forward_log_return(close, h).reindex(feat.index)
    X, y = feat.values, (fwd.values > 0).astype(float)
    n = len(feat)
    names = list(make_candidates(feat.columns))
    probs = {k: np.full(n, np.nan) for k in names}
    auto, chosen = np.full(n, np.nan), np.full(n, None, dtype=object)

    for t in range(min_train + h, n, refit_every):
        end = min(t + refit_every, n)
        tr = slice(0, t - h + 1)
        for k, model in make_candidates(feat.columns).items():
            probs[k][t:end] = model.fit(X[tr], y[tr]).predict_proba(X[t:end])

        known = np.arange(n) <= t - h                       # 결과가 확정된 행만 채점
        scored = known & ~np.isnan(probs["base"])
        idx = np.where(scored)[0][-select_window:]
        if len(idx) >= min_scored:
            pick = min(names, key=lambda k: _logloss(probs[k][idx], y[idx]))
        else:
            pick = "base"
        auto[t:end], chosen[t:end] = probs[pick][t:end], pick

    out = pd.DataFrame({**{k: v for k, v in probs.items()}, "auto": auto, "chosen": chosen,
                        "fwd": fwd.values}, index=feat.index)
    return out.dropna(subset=["auto"]), list(feat.columns)


def score(prob, fwd, h):
    known = fwd.notna() & prob.notna()
    p, up = prob[known].values, (fwd[known].values > 0).astype(float)
    base = float(up.mean())
    hit = float(np.mean((p > 0.5) == up))
    n_eff = max(len(p) / h, 1.0)                     # h일 수익률이 겹치므로 실질 독립 표본 수
    z = (hit - base) / np.sqrt(max(base * (1 - base), 1e-9) / n_eff)
    brier = float(np.mean((p - up) ** 2))
    brier_base = float(np.mean((base - up) ** 2))
    try:
        auc = float(roc_auc_score(up, p)) if 0 < up.sum() < len(up) and np.ptp(p) > 0 else 0.5
    except ValueError:
        auc = 0.5
    return {"hit": hit, "base": base, "z": float(z), "auc": auc, "brier": brier, "brier_base": brier_base,
            "skill": 1 - brier / brier_base if brier_base > 0 else 0.0, "n": int(len(p)), "n_eff": n_eff}


def evaluate(wf, h, daily_returns, cost_bps=0.0):
    """모든 후보와 자동 선택을 채점 + 자동 선택의 '상승 확률>50%면 보유' 전략 vs 단순 보유."""
    cols = [c for c in MODEL_LABELS if c in wf] + ["auto"]
    table = {c: score(wf[c], wf["fwd"], h) for c in cols}

    pos = (wf["auto"] > 0.5).astype(float)
    nxt = daily_returns.shift(-1).reindex(wf.index).fillna(0.0)    # t일 종가에 결정 → t+1일 수익률
    turnover = pos.diff().abs().fillna(pos.iloc[0])
    strat = pos * nxt - turnover * cost_bps * 1e-4
    equity = pd.DataFrame({"전략": np.exp(strat.cumsum()), "단순 보유": np.exp(nxt.cumsum())}, index=wf.index)
    ann = lambda x: (x.mean() * 252, x.std() * np.sqrt(252))
    (sm, ss), (bm, bs) = ann(strat), ann(nxt)
    return {**table["auto"], "table": table, "equity": equity, "time_in_market": float(pos.mean()),
            "sharpe": sm / ss if ss > 0 else 0.0, "sharpe_bh": bm / bs if bs > 0 else 0.0,
            "chosen_share": wf["chosen"].value_counts(normalize=True)}


def predict_now(close, h, wf, macro=None, lag_macro=False, alt=None, select_window=756):
    """결과가 확정된 모든 과거로 학습 → 오늘의 P(상승).

    고르는 모델은 워크포워드에서 최근 select_window개 채점 성적이 가장 좋은 후보.
    """
    feat = build_features(close, macro, lag_macro, alt)
    fwd = forward_log_return(close, h).reindex(feat.index)
    known = fwd.notna().values
    X, y = feat.values, (fwd.values > 0).astype(float)
    cands = make_candidates(feat.columns)
    now = {k: float(m.fit(X[known], y[known]).predict_proba(X[-1:])[0]) for k, m in cands.items()}

    scored = wf.dropna(subset=["fwd"]).iloc[-select_window:]
    yy = (scored["fwd"].values > 0).astype(float)
    pick = min(cands, key=lambda k: _logloss(scored[k].values, yy)) if len(scored) >= 252 else "base"

    expl = LogisticDirectionModel(10.0).fit(X[known], y[known])     # 설명용 (선형이라 기여도 분해 가능)
    contrib = pd.Series(expl.contributions(X[-1]), index=[FEATURE_LABELS[c] for c in feat.columns])
    return now[pick], pick, now, contrib, feat.index[-1]
