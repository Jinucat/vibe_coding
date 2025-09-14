@echo off
echo 🚀 AI MSA 서비스 로컬 실행 스크립트
echo ===================================

echo 📦 Python 가상환경을 활성화합니다...
call python -m venv venv
call venv\Scripts\activate

echo 📦 의존성을 설치합니다...
pip install -r requirements.txt

echo.
echo 🚀 모든 서비스를 시작합니다...
python start_services.py

pause
