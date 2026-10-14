@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv (
    echo [설치] 가상환경을 만들고 패키지를 설치합니다...
    python -m venv .venv
    .venv\Scripts\python -m pip install -r requirements.txt
)
.venv\Scripts\python -m streamlit run app.py
