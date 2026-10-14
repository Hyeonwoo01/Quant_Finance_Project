# Quant Finance Project

변동성 트레이딩 파이프라인(블랙-숄즈 IV → GARCH 변동성 예측 → 신호 → 델타헤지)을 Streamlit 앱으로 만든 프로젝트입니다.
모든 예측은 **그 시점까지의 데이터로만 학습하는 워크포워드 방식**으로 검증하고, 검증 결과를 화면에 함께 보여줍니다.

## 기능

| 탭 | 내용 |
|---|---|
| 🎯 주가 예측 범위 | GARCH 변동성으로 향후 68%·95% 가격 범위 |
| 🧭 방향 예측 | 가격·거시·실적·수급·공시 특징 + 여러 모델을 자동 선택 (기본 확률보다 낫지 않으면 기본 확률 사용) |
| 🏆 코스피 종목 고르기 | 코스피 대형주 약 200개의 상대 순위 (모멘텀·저변동성·수급·실적·자사주·가치 지표), 생존 편향 완화 |
| 📰 시장 심리 | 옵션 풋/콜·스큐, 컨센서스, DART 공시, 뉴스 감성 (현재 값, 참고용) |
| 📊 변동성 / 💹 신호 & 헤지 손익 | 실현·조건부·내재 변동성 비교, 변동성 매수/매도 신호, 스트래들 델타헤지 몬테카를로 |
| 🧪 예측 정확도 검증 | GARCH vs 단순 예측의 과거 오차 |

## 실행

Windows: `run.bat` 더블클릭 (처음 실행 시 가상환경을 만들고 패키지를 설치합니다).

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

### API 키 (선택)

한국 종목의 DART 공시·재무제표와 한국은행 금리를 쓰려면 `api_keys.env.example`을 `api_keys.env`로 복사해 키를 넣으세요.
키는 [OpenDART](https://opendart.fss.or.kr)와 [한국은행 ECOS](https://ecos.bok.or.kr/api/)에서 무료로 발급됩니다.
`api_keys.env`는 `.gitignore`에 들어 있어 저장소에 올라가지 않습니다.

### 코스피 종목 고르기 데이터

앱의 🏆 탭에서 **데이터 받기**를 누르거나 `python universe.py update`를 실행하세요 (처음 20~40분).

## 파일

| 파일 | 역할 |
|---|---|
| `quant_engine.py` | 블랙-숄즈 가격·그릭스·IV, GARCH(1,1), 신호, 감마 스캘핑 |
| `direction.py` | 단일 종목 방향 예측 + 워크포워드 자동 선택 |
| `universe.py` | 코스피 종목 고르기 (데이터 수집, 특징 패널, 워크포워드, 평가) |
| `market_data.py`, `alt_data.py`, `kr_data.py` | Yahoo·네이버·DART·ECOS 데이터 |
| `ui.py`, `.streamlit/config.toml` | 디자인 시스템 (Mintlify 스타일) |
| `tests.py` | 가격·IV·GARCH 검증, 미래 정보 누출 테스트 등 (`python tests.py`) |

## 주의

교육·연구용입니다. 투자 조언이 아니며, 과거 검증 결과가 미래 성과를 보장하지 않습니다.
무료 데이터 소스(Yahoo Finance, 네이버 증권 비공식 API 등)는 예고 없이 바뀌거나 지연될 수 있습니다.
