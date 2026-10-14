"""디자인 시스템 (Mintlify 스타일): 색 토큰, 전역 CSS, Plotly 차트 템플릿, 상단 히어로.

- 글꼴: Inter(본문·제목) + Geist Mono(숫자·코드)
- 버튼은 검은 알약(pill), 민트(#00d4a4)는 포커스·활성 표시에만 아껴 씀
- 카드: 흰 바탕 + 1px 헤어라인 + 12px 모서리, 그림자 거의 없음
- 차트 범주색은 validate_palette.js로 검증한 3색 (블루 → 민트 → 코랄, 고정 순서)
"""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---------- 색 토큰 ----------
INK, CHARCOAL, SLATE, STEEL, STONE, MUTED = "#0a0a0a", "#1c1c1e", "#3a3a3c", "#5a5a5c", "#888888", "#a8a8aa"
CANVAS, SURFACE, SURFACE_SOFT, HAIRLINE, HAIRLINE_SOFT = "#ffffff", "#f7f7f7", "#fafafa", "#e5e5e5", "#ededed"
BRAND_GREEN, BRAND_GREEN_DEEP = "#00d4a4", "#00b48a"

# 차트 전용 (흰 배경 기준 검증 통과: 명도대·채도·색각이상 구분·대비 3:1)
BLUE, MINT, CORAL = "#3772cf", "#009a78", "#f55a3c"
NEUTRAL = STONE                      # 비교 기준선(전체 평균, 단순 보유 등) — 범주색이 아님
POS, NEG = "#1ba673", "#d45656"      # 양/음 극성 (기여도, IC 막대)
BAND_INNER, BAND_OUTER = "rgba(0,154,120,0.22)", "rgba(0,154,120,0.10)"   # 예측 범위 띠

FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Malgun Gothic', sans-serif"
MONO = "'Geist Mono', 'SF Mono', Menlo, Consolas, monospace"

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {{ font-family: {FONT}; }}
.stApp {{ background: {CANVAS}; color: {CHARCOAL}; }}
.block-container {{ padding-top: 2rem; max-width: 1280px; }}

/* 제목: 600 굵기 + 크기에 비례한 음수 자간 */
h1 {{ font-weight: 600 !important; letter-spacing: -1px; color: {INK}; }}
h2 {{ font-weight: 600 !important; letter-spacing: -0.5px; color: {INK}; }}
h3, h4 {{ font-weight: 600 !important; color: {INK}; letter-spacing: 0; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: {STEEL} !important; font-size: 13px; }}
code {{ font-family: {MONO}; font-size: 13px; background: {SURFACE}; color: {CHARCOAL};
        border: 1px solid {HAIRLINE}; border-radius: 4px; padding: 1px 6px; }}

/* 사이드바: 옅은 표면 + 대문자형 소제목 */
section[data-testid="stSidebar"] {{ background: {SURFACE_SOFT}; border-right: 1px solid {HAIRLINE}; }}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{
    font-size: 11px !important; font-weight: 600 !important; letter-spacing: .5px; text-transform: uppercase;
    color: {STEEL} !important; padding: 16px 0 4px; margin: 0; }}

/* 지표 = 카드 */
[data-testid="stMetric"] {{ background: {CANVAS}; border: 1px solid {HAIRLINE}; border-radius: 12px; padding: 14px 16px; }}
[data-testid="stMetricLabel"] p {{ font-size: 13px; color: {STEEL}; font-weight: 500; }}
[data-testid="stMetricValue"] {{ font-size: clamp(17px, 1.6vw, 28px); font-weight: 600; letter-spacing: -0.5px;
                                 color: {INK}; font-variant-numeric: tabular-nums; }}
/* 좁은 화면(사이드바 + 카드 5개)에서 값·설명이 '…'으로 잘리지 않고 줄바꿈되도록 — 안쪽 markdown <p>까지 */
[data-testid="stMetric"] [data-testid="stMarkdownContainer"], [data-testid="stMetric"] p {{
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
    overflow-wrap: anywhere; }}
[data-testid="stMetricDelta"] {{ font-size: 12px; }}
[data-testid="stMetricDelta"] svg {{ display: none; }}           /* 방향 의미 없는 보조설명에 화살표 숨김 */

/* 탭: 밑줄형, 비활성 steel / 활성 ink */
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {HAIRLINE}; }}
.stTabs [data-baseweb="tab"] {{ font-size: 14px; font-weight: 500; color: {STEEL}; padding: 12px 14px; }}
.stTabs [aria-selected="true"] {{ color: {INK}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: {INK}; height: 2px; }}

/* 버튼: 알약 */
.stButton > button, .stDownloadButton > button {{ border-radius: 9999px; font-weight: 500; font-size: 14px;
    padding: 6px 18px; border: 1px solid {HAIRLINE}; }}
.stButton > button[kind="primary"] {{ background: {INK}; color: #fff; border-color: {INK}; }}
.stButton > button[kind="primary"]:active {{ background: {CHARCOAL}; }}

/* 입력: 8px 모서리, 포커스는 브랜드 민트 */
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="base-input"] {{ border-radius: 8px !important; }}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within {{
    border-color: {BRAND_GREEN} !important; box-shadow: 0 0 0 1px {BRAND_GREEN} !important; }}

/* 알림·확장·표: 12px/8px 모서리 */
[data-testid="stAlert"] {{ border-radius: 12px; }}
[data-testid="stExpander"] details {{ border-radius: 12px; border: 1px solid {HAIRLINE_SOFT}; }}
[data-testid="stDataFrame"] {{ border: 1px solid {HAIRLINE}; border-radius: 8px; overflow: hidden; }}
[data-testid="stPlotlyChart"] {{ border: 1px solid {HAIRLINE}; border-radius: 12px; padding: 8px 4px 0; }}
hr {{ border-color: {HAIRLINE}; }}

/* 상단 히어로 (대기 그라데이션은 여기 한 번만) */
.mint-hero {{ background: linear-gradient(180deg, #87a8c8 0%, #f5e9d8 100%); border-radius: 16px;
    padding: 40px 40px 32px; margin-bottom: 8px; }}
.mint-hero h1 {{ font-size: 44px; line-height: 1.05; letter-spacing: -1.5px; margin: 0 0 12px; color: {INK}; padding: 0; }}
.mint-hero p {{ font-size: 17px; line-height: 1.5; color: {SLATE}; margin: 0; max-width: 760px; }}
.mint-pills {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 20px; }}
.mint-pill {{ background: rgba(255,255,255,.75); border: 1px solid rgba(10,10,10,.08); border-radius: 9999px;
    padding: 4px 12px; font-size: 13px; font-weight: 500; color: {CHARCOAL}; }}
.mint-pill b {{ color: {INK}; font-weight: 600; }}
.mint-dot {{ display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: {BRAND_GREEN};
    margin-right: 6px; vertical-align: 1px; }}
@media (max-width: 768px) {{ .mint-hero {{ padding: 28px 20px 24px; }} .mint-hero h1 {{ font-size: 32px; }} }}
</style>
"""


def apply():
    """페이지마다 한 번: 전역 CSS + Plotly 기본 템플릿."""
    st.markdown(_CSS, unsafe_allow_html=True)
    _register_plotly()


def hero(title, subtitle, pills=()):
    """상단 히어로 밴드. pills: [(라벨, 값), ...] — 첫 번째에 민트 점(실시간 데이터 표시)."""
    chips = "".join(
        f'<span class="mint-pill">{"<span class=mint-dot></span>" if i == 0 else ""}{k} <b>{v}</b></span>'
        for i, (k, v) in enumerate(pills))
    st.markdown(f'<div class="mint-hero"><h1>{title}</h1><p>{subtitle}</p>'
                f'{f"<div class=mint-pills>{chips}</div>" if chips else ""}</div>', unsafe_allow_html=True)


def _register_plotly():
    if "mintlify" in pio.templates:
        pio.templates.default = "mintlify"
        return
    axis = dict(gridcolor=HAIRLINE_SOFT, linecolor=HAIRLINE, zerolinecolor=HAIRLINE, tickcolor=HAIRLINE,
                tickfont=dict(color=STEEL, size=12), title=dict(font=dict(color=STEEL, size=12)), showline=False)
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=FONT, color=CHARCOAL, size=13),
        colorway=[BLUE, MINT, CORAL],
        paper_bgcolor=CANVAS, plot_bgcolor=CANVAS,
        xaxis=axis, yaxis=axis,
        legend=dict(font=dict(color=SLATE, size=12), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=CANVAS, bordercolor=HAIRLINE, font=dict(family=FONT, color=INK, size=12)),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    t.data.scatter = [go.Scatter(line=dict(width=2))]
    t.data.bar = [go.Bar(marker=dict(line=dict(width=0)))]
    pio.templates["mintlify"] = t
    pio.templates.default = "mintlify"
