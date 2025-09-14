@echo off
echo 🐳 AI MSA 서비스 Docker 실행 스크립트
echo =====================================

echo 📦 Docker Compose로 모든 서비스를 시작합니다...
docker-compose up -d

echo.
echo ⏳ 서비스가 시작되는 동안 잠시 기다립니다...
timeout /t 10 /nobreak > nul

echo.
echo 🎉 모든 서비스가 시작되었습니다!
echo.
echo 📋 서비스 정보:
echo   - API Gateway: http://localhost:8000
echo   - Face Detection: http://localhost:8001
echo   - Object Detection: http://localhost:8002
echo   - Image Similarity: http://localhost:8003
echo.
echo 📖 API 문서: http://localhost:8000/docs
echo.
echo 🧪 테스트 실행: python test_api.py
echo.
echo ⏹️  서비스를 중지하려면: docker-compose down
echo.
pause
