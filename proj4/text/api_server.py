#!/usr/bin/env python3
"""
FastAPI 기반 Sequence Classification 추론 API 서버
"""

import os
# ---- 로그/프로그레스바/비전 의존 비활성화 (import 전에 설정) ----
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import logging
import warnings
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

# FastAPI 및 관련 라이브러리
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# 딥러닝 관련 라이브러리
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# 로깅 설정
logging.getLogger().setLevel(logging.ERROR)
for name in ["transformers", "huggingface_hub", "urllib3", "filelock", "tqdm"]:
    logging.getLogger(name).setLevel(logging.ERROR)

# Transformers 로깅 끄기
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    try:
        hf_logging.disable_progress_bar()
    except Exception:
        pass
except Exception:
    pass

warnings.filterwarnings("ignore")

# 전역 변수
classifier_pipeline = None
tokenizer = None
model = None
device = "cpu"

# Pydantic 모델 정의
class TextInput(BaseModel):
    text: str = Field(..., description="분류할 텍스트", min_length=1, max_length=1000)
    model_name: Optional[str] = Field(
        default="distilbert-base-uncased-finetuned-sst-2-english",
        description="사용할 모델명"
    )
    method: str = Field(
        default="pipeline",
        description="추론 방법 (pipeline/manual)"
    )

class ClassificationResult(BaseModel):
    label: str = Field(..., description="예측된 라벨")
    score: float = Field(..., description="신뢰도 점수 (0-1)")
    class_id: Optional[int] = Field(None, description="클래스 ID")
    model_name: str = Field(..., description="사용된 모델명")
    method: str = Field(..., description="사용된 추론 방법")

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., description="분류할 텍스트 리스트", min_items=1, max_items=100)
    model_name: Optional[str] = Field(
        default="distilbert-base-uncased-finetuned-sst-2-english",
        description="사용할 모델명"
    )
    method: str = Field(
        default="pipeline",
        description="추론 방법 (pipeline/manual)"
    )

class BatchClassificationResult(BaseModel):
    results: List[ClassificationResult] = Field(..., description="분류 결과 리스트")
    total_count: int = Field(..., description="총 텍스트 수")
    success_count: int = Field(..., description="성공한 분류 수")

class HealthResponse(BaseModel):
    status: str = Field(..., description="서버 상태")
    torch_available: bool = Field(..., description="PyTorch 사용 가능 여부")
    transformers_available: bool = Field(..., description="Transformers 사용 가능 여부")
    device: str = Field(..., description="현재 사용 디바이스")
    model_loaded: bool = Field(..., description="모델 로드 상태")

# 유틸리티 함수들
def normalize_pipeline_device(dev: str) -> int:
    """pipeline device: CPU=-1, GPU=0 (auto 처리 포함)"""
    if dev == "auto":
        return 0 if (TORCH_AVAILABLE and torch.cuda.is_available()) else -1
    if isinstance(dev, str):
        d = dev.lower()
        if d == "cpu":
            return -1
        if d.startswith("cuda"):
            if ":" in d:
                try:
                    return int(d.split(":", 1)[1])
                except ValueError:
                    return 0
            return 0
        return -1
    if isinstance(dev, int):
        return dev
    return -1

def to_torch_device(dev: str) -> str:
    if not TORCH_AVAILABLE:
        return "cpu"
    if dev == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return dev

def load_model_and_tokenizer(model_name: str, device: str):
    """모델/토크나이저 로드"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        dev = to_torch_device(device)
        if TORCH_AVAILABLE:
            model = model.to(dev)
        return tokenizer, model, dev
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 로딩 실패: {str(e)}")

def predict_with_pipeline(text: str, model_name: str, device: str = "auto") -> Dict[str, Any]:
    """Pipeline 추론"""
    try:
        dev_idx = normalize_pipeline_device(device)
        clf = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=dev_idx
        )
        result = clf(text)
        return result[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline 추론 실패: {str(e)}")

def predict_manual(text: str, tokenizer, model, device: str) -> Dict[str, Any]:
    """수동 추론 (PyTorch 필요)"""
    try:
        if not TORCH_AVAILABLE:
            raise HTTPException(status_code=500, detail="manual 모드는 PyTorch가 필요합니다.")
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            idx = logits.argmax(dim=-1).item()
            conf = probs[0][idx].item()
        
        label = model.config.id2label.get(idx, f"Class_{idx}")
        return {"label": label, "score": conf, "class_id": idx}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"수동 추론 실패: {str(e)}")

# 애플리케이션 생명주기 관리
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작시
    global device
    device = to_torch_device("auto")
    print(f"🚀 API 서버 시작 - 디바이스: {device}")
    print(f"🔥 PyTorch 사용 가능: {TORCH_AVAILABLE}")
    print(f"🤖 Transformers 사용 가능: {TRANSFORMERS_AVAILABLE}")
    
    yield
    
    # 종료시
    print("🛑 API 서버 종료")

# FastAPI 앱 생성
app = FastAPI(
    title="Sequence Classification API",
    description="Hugging Face Transformers를 사용한 텍스트 분류 API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 엔드포인트들
@app.get("/", response_model=Dict[str, str])
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Sequence Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스 체크 엔드포인트"""
    return HealthResponse(
        status="healthy",
        torch_available=TORCH_AVAILABLE,
        transformers_available=TRANSFORMERS_AVAILABLE,
        device=device,
        model_loaded=classifier_pipeline is not None or model is not None
    )

@app.post("/classify", response_model=ClassificationResult)
async def classify_text(input_data: TextInput):
    """단일 텍스트 분류"""
    if not TRANSFORMERS_AVAILABLE:
        raise HTTPException(status_code=500, detail="Transformers가 설치되지 않았습니다.")
    
    try:
        if input_data.method == "pipeline":
            result = predict_with_pipeline(input_data.text, input_data.model_name, device)
            return ClassificationResult(
                label=result["label"],
                score=result["score"],
                class_id=None,
                model_name=input_data.model_name,
                method=input_data.method
            )
        else:  # manual
            global tokenizer, model
            if tokenizer is None or model is None:
                tokenizer, model, dev = load_model_and_tokenizer(input_data.model_name, device)
            
            result = predict_manual(input_data.text, tokenizer, model, dev)
            return ClassificationResult(
                label=result["label"],
                score=result["score"],
                class_id=result.get("class_id"),
                model_name=input_data.model_name,
                method=input_data.method
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify/batch", response_model=BatchClassificationResult)
async def classify_texts_batch(input_data: BatchTextInput):
    """배치 텍스트 분류"""
    if not TRANSFORMERS_AVAILABLE:
        raise HTTPException(status_code=500, detail="Transformers가 설치되지 않았습니다.")
    
    results = []
    success_count = 0
    
    try:
        for text in input_data.texts:
            try:
                if input_data.method == "pipeline":
                    result = predict_with_pipeline(text, input_data.model_name, device)
                    results.append(ClassificationResult(
                        label=result["label"],
                        score=result["score"],
                        class_id=None,
                        model_name=input_data.model_name,
                        method=input_data.method
                    ))
                else:  # manual
                    global tokenizer, model
                    if tokenizer is None or model is None:
                        tokenizer, model, dev = load_model_and_tokenizer(input_data.model_name, device)
                    
                    result = predict_manual(text, tokenizer, model, dev)
                    results.append(ClassificationResult(
                        label=result["label"],
                        score=result["score"],
                        class_id=result.get("class_id"),
                        model_name=input_data.model_name,
                        method=input_data.method
                    ))
                success_count += 1
            except Exception as e:
                # 개별 텍스트 실패시에도 계속 진행
                results.append(ClassificationResult(
                    label="ERROR",
                    score=0.0,
                    class_id=None,
                    model_name=input_data.model_name,
                    method=input_data.method
                ))
        
        return BatchClassificationResult(
            results=results,
            total_count=len(input_data.texts),
            success_count=success_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/info")
async def get_models_info():
    """사용 가능한 모델 정보"""
    return {
        "default_model": "distilbert-base-uncased-finetuned-sst-2-english",
        "supported_models": [
            "distilbert-base-uncased-finetuned-sst-2-english",
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "nlptown/bert-base-multilingual-uncased-sentiment"
        ],
        "torch_available": TORCH_AVAILABLE,
        "device": device
    }

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
