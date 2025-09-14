# AI MSA 서비스

이 프로젝트는 AI 기능을 마이크로서비스 아키텍처로 재구성한 것입니다.

## 🏗️ 아키텍처

```
┌─────────────────┐
│   API Gateway   │ ← 포트 8000
│   (FastAPI)     │
└─────────┬───────┘
          │
    ┌─────┼─────┐
    │     │     │
    ▼     ▼     ▼
┌──────┐ ┌──────┐ ┌──────────┐
│얼굴검출│ │객체검출│ │이미지유사도│
│서비스 │ │서비스 │ │  서비스   │
│:8001  │ │:8002  │ │  :8003   │
└──────┘ └──────┘ └──────────┘
```

## 🚀 서비스 구성

### 1. 얼굴 검출 서비스 (Face Detection Service)
- **포트**: 8001
- **모델**: BlazeFace (blaze_face_short_range.tflite)
- **기능**: 얼굴 검출, 키포인트 추출
- **엔드포인트**:
  - `POST /detect` - 파일 업로드로 얼굴 검출
  - `POST /detect_base64` - Base64 이미지로 얼굴 검출

### 2. 객체 검출 서비스 (Object Detection Service)
- **포트**: 8002
- **모델**: EfficientDet (efficientdet_lite0.tflite)
- **기능**: 객체 검출, 분류
- **엔드포인트**:
  - `POST /detect` - 파일 업로드로 객체 검출
  - `POST /detect_base64` - Base64 이미지로 객체 검출

### 3. 이미지 유사도 서비스 (Image Similarity Service)
- **포트**: 8003
- **모델**: MobileNet V3 (mobilenet_v3_small.tflite)
- **기능**: 이미지 임베딩, 유사도 계산
- **엔드포인트**:
  - `POST /embed` - 이미지 임베딩 계산
  - `POST /compare` - 두 이미지 유사도 비교
  - `POST /compare_multiple` - 여러 이미지 유사도 비교

### 4. API Gateway
- **포트**: 8000
- **기능**: 모든 서비스 통합, 라우팅, CORS 처리
- **엔드포인트**:
  - `GET /` - 서비스 정보
  - `GET /health` - 헬스 체크
  - `POST /api/v1/face-detection/*` - 얼굴 검출 API
  - `POST /api/v1/object-detection/*` - 객체 검출 API
  - `POST /api/v1/image-similarity/*` - 이미지 유사도 API
  - `POST /api/v1/analyze` - 통합 분석 API

## 🛠️ 설치 및 실행

### 방법 1: Docker Compose (권장)

```bash
# 모든 서비스 실행
docker-compose up -d

# 또는 Windows에서
run_docker.bat
```

### 방법 2: 로컬 실행

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt

# 모든 서비스 실행
python start_services.py

# 또는 Windows에서
run_local.bat
```

## 📖 API 사용법

### 1. API 문서 확인
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 2. 헬스 체크
```bash
curl http://localhost:8000/health
```

### 3. 얼굴 검출
```bash
curl -X POST "http://localhost:8000/api/v1/face-detection/detect" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@image.jpg"
```

### 4. 객체 검출
```bash
curl -X POST "http://localhost:8000/api/v1/object-detection/detect" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@image.jpg"
```

### 5. 이미지 유사도 비교
```bash
curl -X POST "http://localhost:8000/api/v1/image-similarity/compare" \
     -H "Content-Type: multipart/form-data" \
     -F "file1=@image1.jpg" \
     -F "file2=@image2.jpg"
```

### 6. 통합 분석
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@image.jpg"
```

## 🧪 테스트

```bash
# API 테스트 실행
python test_api.py
```

## 📁 프로젝트 구조

```
msa_ai_services/
├── api-gateway/                 # API Gateway
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
├── services/
│   ├── face-detection-service/  # 얼굴 검출 서비스
│   │   ├── app.py
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── object-detection-service/ # 객체 검출 서비스
│   │   ├── app.py
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── image-similarity-service/ # 이미지 유사도 서비스
│       ├── app.py
│       ├── requirements.txt
│       └── Dockerfile
├── models/                      # AI 모델 파일들
│   ├── blaze_face_short_range.tflite
│   ├── efficientdet_lite0.tflite
│   └── mobilenet_v3_small.tflite
├── docker-compose.yml           # Docker Compose 설정
├── requirements.txt             # 전체 의존성
├── start_services.py            # 로컬 실행 스크립트
├── test_api.py                  # API 테스트 스크립트
├── run_docker.bat              # Docker 실행 스크립트 (Windows)
├── run_local.bat               # 로컬 실행 스크립트 (Windows)
└── README.md                   # 이 파일
```

## 🔧 설정

### 환경 변수
- `FACE_SERVICE_URL`: 얼굴 검출 서비스 URL
- `OBJECT_SERVICE_URL`: 객체 검출 서비스 URL
- `SIMILARITY_SERVICE_URL`: 이미지 유사도 서비스 URL
- `MODEL_PATH`: 각 서비스의 모델 파일 경로

### 포트 설정
- API Gateway: 8000
- 얼굴 검출 서비스: 8001
- 객체 검출 서비스: 8002
- 이미지 유사도 서비스: 8003

## 🚀 배포

### Docker로 배포
```bash
# 이미지 빌드
docker-compose build

# 서비스 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 서비스 중지
docker-compose down
```

### 개별 서비스 실행
```bash
# 얼굴 검출 서비스만 실행
cd services/face-detection-service
python app.py

# 객체 검출 서비스만 실행
cd services/object-detection-service
python app.py

# 이미지 유사도 서비스만 실행
cd services/image-similarity-service
python app.py

# API Gateway만 실행
cd api-gateway
python app.py
```

## 📊 모니터링

### 헬스 체크
- 전체 서비스: `GET /health`
- 개별 서비스: `GET /{service}/health`

### 로그 확인
```bash
# Docker Compose 로그
docker-compose logs -f

# 개별 서비스 로그
docker-compose logs -f face-detection-service
docker-compose logs -f object-detection-service
docker-compose logs -f image-similarity-service
docker-compose logs -f api-gateway
```

## 🔍 문제 해결

### 일반적인 문제들

1. **모델 파일을 찾을 수 없음**
   - `models/` 디렉토리에 모델 파일이 있는지 확인
   - 환경 변수 `MODEL_PATH` 설정 확인

2. **서비스 연결 실패**
   - 모든 서비스가 실행 중인지 확인
   - 포트가 사용 중이지 않은지 확인
   - Docker 네트워크 설정 확인

3. **메모리 부족**
   - Docker 메모리 제한 증가
   - 모델 로딩 최적화

### 로그 확인
```bash
# 전체 로그
docker-compose logs

# 특정 서비스 로그
docker-compose logs face-detection-service
```

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 지원

문제가 발생하면 이슈를 생성해 주세요.