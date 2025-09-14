# AI Inference APIs

Hugging Face Transformers를 사용한 텍스트 분류 및 MediaPipe를 사용한 이미지 임베딩 유사도 계산을 위한 FastAPI 기반 웹 API 서버들입니다.

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
conda create -n vc python==3.10 -y
conda activate vc

# PyTorch 설치 (CUDA 11.8)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 2. CUDA 설치 (GPU 사용시)
- [NVIDIA CUDA 11.8 다운로드](https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_522.06_windows.exe)

### 3. 설치 확인
```python
import torch
print(torch.cuda.is_available())  # True면 GPU 사용 가능
```

## 📖 사용법

### 🖥️ CLI 사용법 (명령행)
```bash
# 텍스트 분류
python sequence_classification_inference.py --text "This movie is amazing!"

# 이미지 임베딩 유사도 (원본 스크립트)
python embed_sim.py --model models/mobilenet_v3_small.tflite --images img1.jpg img2.jpg
```

### 🌐 API 서버 사용법

#### 1. 텍스트 분류 API (포트 8000)
```bash
# 서버 시작
python api_server.py

# API 문서: http://localhost:8000/docs
# 클라이언트 테스트
python api_client.py
```

#### 2. 이미지 임베딩 API (포트 8001)
```bash
# 서버 시작
python image_embedding_api.py

# API 문서: http://localhost:8001/docs
# 클라이언트 테스트
python image_embedding_client.py
```

### 🔧 API 엔드포인트

#### 텍스트 분류 API (포트 8000)
- `GET /` - 루트 정보
- `GET /health` - 서버 상태 확인
- `POST /classify` - 단일 텍스트 분류
- `POST /classify/batch` - 배치 텍스트 분류
- `GET /models/info` - 사용 가능한 모델 정보

#### 이미지 임베딩 API (포트 8001)
- `GET /` - 루트 정보
- `GET /health` - 서버 상태 확인
- `POST /load-model` - MediaPipe 모델 로드
- `POST /embedding` - 단일 이미지 임베딩 계산
- `POST /similarity/single` - 두 이미지 간 유사도
- `POST /similarity/batch` - 여러 이미지 유사도 행렬
- `POST /upload-images` - 이미지 파일 업로드
- `GET /supported-formats` - 지원 이미지 형식

## 🔧 주요 기능

### 텍스트 분류 API
- **두 가지 추론 방법**: Pipeline 방식과 수동 방식
- **다양한 모델 지원**: DistilBERT, RoBERTa 등
- **GPU 자동 감지**: CUDA 사용 가능시 자동으로 GPU 사용
- **배치 처리**: 여러 텍스트 동시 분류
- **RESTful API**: 표준 HTTP 메서드 사용

### 이미지 임베딩 API
- **MediaPipe 기반**: Google의 고성능 임베딩 모델
- **유사도 계산**: 코사인 유사도 기반 이미지 비교
- **배치 처리**: 여러 이미지 동시 처리
- **파일 업로드**: 이미지 파일 직접 업로드 지원
- **다양한 형식**: JPG, PNG, WebP 등 지원

### 공통 기능
- **자동 문서화**: Swagger UI 제공 (`/docs`)
- **비동기 처리**: FastAPI 기반 고성능 서버
- **CORS 지원**: 웹 애플리케이션 통합 가능
- **헬스 체크**: 서버 상태 모니터링
- **에러 처리**: 안정적인 에러 핸들링

## 📁 파일 구조

```
proj4/
├── sequence_classification_inference.py  # 텍스트 분류 CLI 스크립트
├── embed_sim.py                          # 이미지 임베딩 CLI 스크립트 (원본)
├── api_server.py                         # 텍스트 분류 FastAPI 서버
├── api_client.py                         # 텍스트 분류 API 클라이언트
├── image_embedding_api.py                # 이미지 임베딩 FastAPI 서버
├── image_embedding_client.py             # 이미지 임베딩 API 클라이언트
├── requirements.txt                      # 필요한 패키지 목록
└── README.md                            # 이 파일
```

## 🎯 지원하는 모델

### 텍스트 분류 모델
- `distilbert-base-uncased-finetuned-sst-2-english` (기본값)
- `cardiffnlp/twitter-roberta-base-sentiment-latest`
- `nlptown/bert-base-multilingual-uncased-sentiment`
- 기타 Hugging Face Hub의 모든 sequence classification 모델

### 이미지 임베딩 모델
- MediaPipe TFLite 모델 (예: `mobilenet_v3_small.tflite`)
- [MediaPipe Image Embedder 모델](https://developers.google.com/mediapipe/solutions/vision/image_embedder)
- 사용자 정의 TFLite 임베딩 모델

## 📝 API 사용 예제

### 텍스트 분류 API
```python
import requests

# 단일 텍스트 분류
response = requests.post("http://localhost:8000/classify", json={
    "text": "This movie is amazing!",
    "method": "pipeline"
})
result = response.json()
print(f"라벨: {result['label']}, 점수: {result['score']}")

# 배치 분류
response = requests.post("http://localhost:8000/classify/batch", json={
    "texts": ["Great!", "Terrible!", "Okay."],
    "method": "pipeline"
})
results = response.json()
for i, result in enumerate(results['results']):
    print(f"텍스트 {i+1}: {result['label']} ({result['score']:.4f})")
```

### 이미지 임베딩 API
```python
import requests

# 모델 로드
requests.post("http://localhost:8001/load-model", data={
    "model_path_str": "models/mobilenet_v3_small.tflite"
})

# 두 이미지 간 유사도
response = requests.post("http://localhost:8001/similarity/single", json={
    "image1_path": "image1.jpg",
    "image2_path": "image2.jpg"
})
result = response.json()
print(f"유사도: {result['similarity']:.4f} ({result['percentage']})")

# 여러 이미지 유사도 행렬
response = requests.post("http://localhost:8001/similarity/batch", json={
    "image_paths": ["img1.jpg", "img2.jpg", "img3.jpg"]
})
results = response.json()
print(f"유사도 행렬: {results['similarity_matrix']}")
```

### cURL 예제
```bash
# 텍스트 분류
curl -X POST "http://localhost:8000/classify" \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie is amazing!", "method": "pipeline"}'

# 이미지 유사도
curl -X POST "http://localhost:8001/similarity/single" \
     -H "Content-Type: application/json" \
     -d '{"image1_path": "image1.jpg", "image2_path": "image2.jpg"}'
```

## 📚 참고 자료

- [Hugging Face Sequence Classification 문서](https://huggingface.co/docs/transformers/tasks/sequence_classification)
- [MediaPipe Image Embedder 문서](https://developers.google.com/mediapipe/solutions/vision/image_embedder)
- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [OpenCV Python 문서](https://opencv-python-tutroals.readthedocs.io/)