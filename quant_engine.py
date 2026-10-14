"""노트북(Quant_Finance_Project_fixed.ipynb)의 핵심 클래스를 프로그램용 모듈로 옮긴 것.

프라이싱/IV → GARCH 변동성 예측 → 신호 → 델타헤지(감마 스캘핑) 손익.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, minimize
from scipy.signal import lfilter


class RiskNeutralPricingEngine:
    """유러피안 옵션의 블랙-숄즈 가격·그릭스·IV (위험중립 측도, 연속 배당수익률 q 지원).

    S: 현재가, K: 행사가, T: 잔존만기(년), r: 무위험이자율, q: 연속 배당수익률.
    미국식 개별주식 옵션(조기행사·이산배당)에는 근사치일 뿐이다.
    """

    def __init__(self, S, K, T, r, option_type='call', q=0.0):
        self.S, self.K, self.T, self.r, self.q = S, K, T, r, q
        self.option_type = option_type.lower()
        if self.option_type not in ('call', 'put'):
            raise ValueError("option_type은 'call' 또는 'put'이어야 합니다.")

    def _d1_d2(self, sigma):
        vol_t = sigma * np.sqrt(self.T)
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * sigma**2) * self.T) / vol_t
        return d1, d1 - vol_t

    def price(self, sigma):
        fs, dk = self.S * np.exp(-self.q * self.T), self.K * np.exp(-self.r * self.T)
        if self.T <= 0 or sigma <= 0:  # 만기이거나 변동성 0이면 내재가치
            return max(fs - dk, 0.0) if self.option_type == 'call' else max(dk - fs, 0.0)
        d1, d2 = self._d1_d2(sigma)
        if self.option_type == 'call':
            return fs * norm.cdf(d1) - dk * norm.cdf(d2)
        return dk * norm.cdf(-d2) - fs * norm.cdf(-d1)

    def delta(self, sigma):
        d1, _ = self._d1_d2(sigma)
        dfq = np.exp(-self.q * self.T)
        return dfq * norm.cdf(d1) if self.option_type == 'call' else -dfq * norm.cdf(-d1)

    def gamma(self, sigma):
        d1, _ = self._d1_d2(sigma)
        return np.exp(-self.q * self.T) * norm.pdf(d1) / (self.S * sigma * np.sqrt(self.T))

    def vega(self, sigma):
        """sigma 1.00(=100%p) 변화당 가격 변화."""
        if self.T <= 0:
            return 0.0
        d1, _ = self._d1_d2(sigma)
        return self.S * np.exp(-self.q * self.T) * np.sqrt(self.T) * norm.pdf(d1)

    def theta(self, sigma):
        """연 단위 세타 (시간 경과에 따른 가격 변화, 보통 음수)."""
        d1, d2 = self._d1_d2(sigma)
        fs, dk = self.S * np.exp(-self.q * self.T), self.K * np.exp(-self.r * self.T)
        decay = -fs * norm.pdf(d1) * sigma / (2 * np.sqrt(self.T))
        if self.option_type == 'call':
            return decay - self.r * dk * norm.cdf(d2) + self.q * fs * norm.cdf(d1)
        return decay + self.r * dk * norm.cdf(-d2) - self.q * fs * norm.cdf(-d1)

    def price_bounds(self):
        """무차익 가격 범위(하한, 상한). 이 밖의 가격에는 IV가 존재하지 않는다."""
        fs, dk = self.S * np.exp(-self.q * self.T), self.K * np.exp(-self.r * self.T)
        return (max(fs - dk, 0.0), fs) if self.option_type == 'call' else (max(dk - fs, 0.0), dk)

    def implied_volatility(self, market_price, tol=1e-8, max_iter=50, lo=1e-4, hi=5.0):
        """뉴턴-랩슨(빠름) → 실패 시 브렌트(항상 수렴). 해가 없으면 nan."""
        lower, upper = self.price_bounds()
        if self.T <= 0 or not (lower < market_price < upper):
            return np.nan

        sigma = 0.20
        for _ in range(max_iter):
            diff = self.price(sigma) - market_price
            v = self.vega(sigma)
            if v < 1e-12:
                break
            step = diff / v
            if abs(step) < tol:
                return sigma
            new = sigma - step
            if not (lo < new < hi):
                break
            sigma = new
        try:
            return brentq(lambda s: self.price(s) - market_price, lo, hi, xtol=tol)
        except ValueError:
            return np.nan


class GarchVolForecaster:
    """GARCH(1,1) 가우시안 준최우추정 + 옵션 잔존만기 구간의 '평균' 변동성 예측.

    입력: 일간 로그수익률(소수 단위). 출력: 연율화된 소수 변동성 → IV와 바로 비교 가능.
    """

    def __init__(self, trading_days=252):
        self.trading_days = trading_days
        self.omega = self.alpha = self.beta = self.sigma2_next = None
        self.cond_vol = None     # 적합 구간의 조건부 변동성(연율) — 차트용

    def fit(self, daily_log_returns):
        r = np.asarray(daily_log_returns, float)
        x = (r - r.mean()) * 100.0          # 수치 안정성을 위해 % 단위로 스케일
        n, var0 = len(x), x.var()

        def cond_var(p):
            omega, alpha, beta = p
            s2 = np.empty(n)
            s2[0] = var0
            s2[1:], _ = lfilter([1.0], [1.0, -beta], omega + alpha * x[:-1] ** 2, zi=[beta * var0])
            return s2

        def nll(p):
            s2 = cond_var(p)
            return 0.5 * np.sum(np.log(s2) + x**2 / s2)

        res = minimize(nll, x0=[0.05 * var0, 0.05, 0.90], method='SLSQP',
                       bounds=[(1e-8, None), (1e-6, 0.999), (1e-6, 0.999)],
                       constraints=[{'type': 'ineq', 'fun': lambda p: 0.9999 - p[1] - p[2]}])
        omega, alpha, beta = res.x
        s2 = cond_var(res.x)
        self.omega, self.alpha, self.beta = omega / 1e4, alpha, beta
        self.sigma2_next = (omega + alpha * x[-1] ** 2 + beta * s2[-1]) / 1e4
        self.cond_vol = np.sqrt(s2 / 1e4 * self.trading_days)
        return self

    @property
    def persistence(self):
        return self.alpha + self.beta

    @property
    def long_run_var(self):
        return self.omega / (1.0 - self.alpha - self.beta)

    def daily_var_path(self, horizon_days):
        """향후 1..horizon_days 각 날의 예측 일간 분산."""
        k = np.arange(horizon_days)
        return self.long_run_var + self.persistence ** k * (self.sigma2_next - self.long_run_var)

    def forecast_vol(self, horizon_days):
        """향후 horizon_days 동안의 평균 변동성(연율, 소수)."""
        return float(np.sqrt(self.daily_var_path(horizon_days).mean() * self.trading_days))


class VolatilityTradingLogic:
    """edge = IV − (예측 RV + vrp).  edge > threshold → 변동성 매도, edge < −threshold → 변동성 매수."""
    LABELS = {1: "LONG_VOL (변동성 매수)", -1: "SHORT_VOL (변동성 매도)", 0: "HOLD (관망)"}

    def __init__(self, vrp=0.0, threshold=0.02):
        self.vrp, self.threshold = vrp, threshold

    @staticmethod
    def estimate_vrp(past_iv, past_realized_vol):
        return float(np.mean(np.asarray(past_iv, float) - np.asarray(past_realized_vol, float)))

    def edge(self, expected_rv, market_iv):
        return market_iv - (expected_rv + self.vrp)

    def generate_signal(self, expected_rv, market_iv):
        e = self.edge(expected_rv, market_iv)
        if e > self.threshold:
            return -1
        if e < -self.threshold:
            return 1
        return 0


def _straddle_delta(S, K, T, r, sigma, q=0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * (2 * norm.cdf(d1) - 1)


class GammaScalpingEngine:
    """ATM 스트래들(콜+풋)의 그릭스와 델타헤지 경로 손익.

    position = +1: 롱 스트래들(롱 감마) / -1: 숏 스트래들(숏 감마). 신호 부호를 그대로 쓴다.
    """

    def __init__(self, S, K, T, r, sigma, q=0.0):
        self.S, self.K, self.T, self.r, self.sigma, self.q = S, K, T, r, sigma, q
        self.call = RiskNeutralPricingEngine(S, K, T, r, 'call', q)
        self.put = RiskNeutralPricingEngine(S, K, T, r, 'put', q)

    def straddle_greeks(self):
        s = self.sigma
        return {"delta": self.call.delta(s) + self.put.delta(s),
                "gamma": self.call.gamma(s) + self.put.gamma(s),
                "vega": self.call.vega(s) + self.put.vega(s),
                "theta_per_day": (self.call.theta(s) + self.put.theta(s)) / 252,
                "price": self.call.price(s) + self.put.price(s)}

    def hedged_pnl(self, paths, position=1, cost_bps=0.0):
        """경로(들)를 따라 매 스텝 델타헤지했을 때의 만기 손익(스트래들 1단위).

        paths: 길이 n+1 배열 하나, 또는 (경로수, n+1) 배열 — 여러 경로를 한 번에 벡터 계산.
        """
        S = np.asarray(paths, float)
        single = S.ndim == 1
        S = np.atleast_2d(S)
        n = S.shape[1] - 1
        dt, cost = self.T / n, cost_bps * 1e-4
        K, r, q, sg = self.K, self.r, self.q, self.sigma

        shares = -position * _straddle_delta(S[:, 0], K, self.T, r, sg, q)
        cash = -position * self.straddle_greeks()["price"] - shares * S[:, 0] - cost * np.abs(shares) * S[:, 0]
        for t in range(1, n + 1):
            cash = cash * np.exp(r * dt) + shares * S[:, t - 1] * q * dt
            if t < n:
                new = -position * _straddle_delta(S[:, t], K, self.T - t * dt, r, sg, q)
                trade = new - shares
                cash -= trade * S[:, t] + cost * np.abs(trade) * S[:, t]
                shares = new
        payoff = np.abs(S[:, n] - K)
        out = cash + shares * S[:, n] + position * payoff
        return float(out[0]) if single else out


def simulate_gbm_paths(S0, vol, r, T, n_steps, n_paths, rng):
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    inc = (r - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z
    return S0 * np.exp(np.concatenate([np.zeros((n_paths, 1)), np.cumsum(inc, axis=1)], axis=1))
