#!/usr/bin/env python3
"""
FastAPI 기반 이미지 임베딩 유사도 API 서버
MediaPipe를 사용한 이미지 임베딩 및 유사도 계산
"""

import os
import math
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# FastAPI 및 관련 라이브러리
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# 로깅 설정
import logging
logging.getLogger().setLevel(logging.ERROR)

# ====== 설정 ======
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_EXTENSIONS = IMG_EXTS

# 자동 모델 경로(환경변수 우선)
AUTO_MODEL_PATH = os.getenv("IMAGE_EMBEDDER_PATH", "models/mobilenet_v3_small.tflite")

# 전역 상태
embedder = None
model_path: Optional[Path] = None

# ====== Pydantic 모델 ======
class ImageSimilarityRequest(BaseModel):
    image_paths: List[str] = Field(..., min_items=2, description="이미지 파일 경로 리스트")
    l2_normalize: bool = Field(default=True)
    quantize: bool = Field(default=True)

class ImageSimilarityResponse(BaseModel):
    similarity_matrix: List[List[float]]
    image_names: List[str]
    top_pairs: List[Dict[str, Any]]
    total_images: int

class SingleSimilarityRequest(BaseModel):
    image1_path: str
    image2_path: str
    l2_normalize: bool = Field(default=True)
    quantize: bool = Field(default=True)

class SingleSimilarityResponse(BaseModel):
    similarity: float
    image1_name: str
    image2_name: str
    percentage: str

class EmbeddingRequest(BaseModel):
    image_path: str
    l2_normalize: bool = Field(default=True)
    quantize: bool = Field(default=True)

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    image_name: str
    embedding_dim: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: Optional[str]
    mediapipe_available: bool

# ====== 유틸 ======
def imread_unicode(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None

def resize_for_preview(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if h < w:
        new_h = max(1, math.floor(h / (w / DESIRED_WIDTH)))
        return cv2.resize(image, (DESIRED_WIDTH, new_h))
    else:
        new_w = max(1, math.floor(w / (h / DESIRED_HEIGHT)))
        return cv2.resize(image, (new_w, DESIRED_HEIGHT))

def validate_image_path(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"이미지 파일을 찾을 수 없습니다: {path}")
    if p.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 이미지 형식: {p.suffix}")
    return p

def build_embedder(mpath: Path, l2_normalize: bool = True, quantize: bool = True):
    try:
        base_options = python.BaseOptions(model_asset_path=str(mpath))
        options = vision.ImageEmbedderOptions(
            base_options=base_options, l2_normalize=l2_normalize, quantize=quantize
        )
        return vision.ImageEmbedder.create_from_options(options)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"임베더 생성 실패: {e}")

def make_mp_image(path: Path) -> mp.Image:
    try:
        return mp.Image.create_from_file(str(path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 로드 실패: {e}")

def compute_embedding(_embedder, image_path: Path) -> List[float]:
    try:
        mp_image = make_mp_image(image_path)
        result = _embedder.embed(mp_image)
        return result.embeddings[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"임베딩 계산 실패: {e}")

def compute_similarities(_embedder, image_paths: List[Path]) -> np.ndarray:
    try:
        mp_images = [make_mp_image(p) for p in image_paths]
        emb_results = [ _embedder.embed(im).embeddings[0] for im in mp_images ]
        n = len(image_paths)
        sims = np.eye(n, dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                s = vision.ImageEmbedder.cosine_similarity(emb_results[i], emb_results[j])
                sims[i, j] = sims[j, i] = s
        return sims
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"유사도 계산 실패: {e}")

def get_top_similar_pairs(similarity_matrix: np.ndarray, image_names: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
    pairs = []
    n = len(image_names)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                "image1_index": i,
                "image2_index": j,
                "image1_name": image_names[i],
                "image2_name": image_names[j],
                "similarity": float(similarity_matrix[i, j])
            })
    pairs.sort(key=lambda x: x["similarity"], reverse=True)
    return pairs[:top_k]

# ====== 앱 라이프사이클 ======
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, model_path
    print("🚀 이미지 임베딩 API 서버 시작")
    print(f"📁 MediaPipe 사용 가능: {mp is not None}")
    # 자동 모델 로드
    try:
        cand = Path(AUTO_MODEL_PATH)
        if cand.exists():
            embedder = build_embedder(cand, l2_normalize=True, quantize=True)
            model_path = cand
            print(f"✅ 자동 모델 로드: {cand}")
        else:
            print(f"ℹ️ 자동 모델 로드 생략(파일 없음): {cand}")
    except Exception as e:
        print(f"⚠️ 자동 모델 로드 실패: {e}")
    yield
    print("🛑 이미지 임베딩 API 서버 종료")

# ====== FastAPI 앱 ======
app = FastAPI(
    title="Image Embedding Similarity API",
    description="MediaPipe를 사용한 이미지 임베딩 및 유사도 계산 API",
    version="1.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ====== 엔드포인트 ======
@app.get("/", response_model=Dict[str, str])
async def root():
    return {"message": "Image Embedding Similarity API", "version": "1.1.0", "docs": "/docs", "health": "/health"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=embedder is not None,
        model_path=str(model_path) if model_path else None,
        mediapipe_available=mp is not None
    )

@app.post("/load-model")
async def load_model(
    model_path_str: str = Form(..., description="TFLite 모델 파일 경로"),
    l2_normalize: bool = Form(default=True),
    quantize: bool = Form(default=True)
):
    global embedder, model_path
    mpath = Path(model_path_str)
    if not mpath.exists():
        raise HTTPException(status_code=404, detail=f"모델 파일을 찾을 수 없습니다: {model_path_str}")
    embedder = build_embedder(mpath, l2_normalize, quantize)
    model_path = mpath
    return {"message": "모델 로드 성공", "model_path": str(mpath), "l2_normalize": l2_normalize, "quantize": quantize}

@app.post("/embedding", response_model=EmbeddingResponse)
async def get_embedding(req: EmbeddingRequest):
    if embedder is None:
        raise HTTPException(status_code=400, detail="모델이 로드되지 않았습니다. /load-model을 먼저 호출하세요.")
    image_path = validate_image_path(req.image_path)
    vec = compute_embedding(embedder, image_path)
    return EmbeddingResponse(embedding=vec, image_name=image_path.name, embedding_dim=len(vec))

@app.post("/similarity/single", response_model=SingleSimilarityResponse)
async def compute_single_similarity(req: SingleSimilarityRequest):
    if embedder is None:
        raise HTTPException(status_code=400, detail="모델이 로드되지 않았습니다. /load-model을 먼저 호출하세요.")
    p1 = validate_image_path(req.image1_path)
    p2 = validate_image_path(req.image2_path)
    emb1 = compute_embedding(embedder, p1)
    emb2 = compute_embedding(embedder, p2)
    sim = vision.ImageEmbedder.cosine_similarity(
        vision.Embedding(embedding=emb1), vision.Embedding(embedding=emb2)
    )
    return SingleSimilarityResponse(
        similarity=float(sim), image1_name=p1.name, image2_name=p2.name, percentage=f"{sim*100:.1f}%"
    )

@app.post("/similarity/batch", response_model=ImageSimilarityResponse)
async def compute_batch_similarity(req: ImageSimilarityRequest):
    if embedder is None:
        raise HTTPException(status_code=400, detail="모델이 로드되지 않았습니다. /load-model을 먼저 호출하세요.")
    paths = [validate_image_path(p) for p in req.image_paths]
    sims = compute_similarities(embedder, paths)
    names = [p.name for p in paths]
    top_pairs = get_top_similar_pairs(sims, names)
    return ImageSimilarityResponse(similarity_matrix=sims.tolist(), image_names=names, top_pairs=top_pairs, total_images=len(paths))

# ---- 업로드: keep_temp 지원 추가 ----
@app.post("/upload-images")
async def upload_images(
    files: List[UploadFile] = File(..., description="업로드할 이미지 파일들"),
    keep_temp: bool = Form(False),
    background_tasks: BackgroundTasks = None
):
    if not files:
        raise HTTPException(status_code=400, detail="업로드할 파일이 없습니다.")
    for f in files:
        if not any(f.filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
            raise HTTPException(status_code=400, detail=f"지원하지 않는 파일 형식: {f.filename}")

    temp_dir = Path(tempfile.mkdtemp())
    saved_paths: List[str] = []
    try:
        for f in files:
            fp = temp_dir / f.filename
            with open(fp, "wb") as buf:
                shutil.copyfileobj(f.file, buf)
            saved_paths.append(str(fp))
        # 기본은 삭제, keep_temp면 유지
        if background_tasks and not keep_temp:
            background_tasks.add_task(shutil.rmtree, temp_dir)
        return {
            "message": f"{len(files)}개 파일 업로드 성공",
            "saved_paths": saved_paths,
            "temp_directory": str(temp_dir),
            "kept": bool(keep_temp)
        }
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"파일 업로드 실패: {e}")

# ---- 원스텝: 업로드 + 즉시 유사도 ----
@app.post("/similarity/upload-single", response_model=SingleSimilarityResponse)
async def upload_and_compare_single(
    file1: UploadFile = File(..., description="첫 번째 이미지 파일"),
    file2: UploadFile = File(..., description="두 번째 이미지 파일"),
    l2_normalize: bool = Form(default=True),
    quantize: bool = Form(default=True)
):
    if embedder is None:
        raise HTTPException(status_code=400, detail="모델이 로드되지 않았습니다. /load-model을 먼저 호출하세요.")
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        p1 = tmp_dir / file1.filename
        p2 = tmp_dir / file2.filename
        with open(p1, "wb") as b1: shutil.copyfileobj(file1.file, b1)
        with open(p2, "wb") as b2: shutil.copyfileobj(file2.file, b2)
        # (옵션 플래그는 embedder 생성 시 반영되므로 여기선 정보로만)
        e1 = compute_embedding(embedder, p1)
        e2 = compute_embedding(embedder, p2)
        sim = vision.ImageEmbedder.cosine_similarity(
            vision.Embedding(embedding=e1), vision.Embedding(embedding=e2)
        )
        return SingleSimilarityResponse(
            similarity=float(sim), image1_name=p1.name, image2_name=p2.name, percentage=f"{sim*100:.1f}%"
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

@app.post("/similarity/upload-batch", response_model=ImageSimilarityResponse)
async def upload_and_compare_batch(
    files: List[UploadFile] = File(..., description="여러 이미지 파일"),
):
    if embedder is None:
        raise HTTPException(status_code=400, detail="모델이 로드되지 않았습니다. /load-model을 먼저 호출하세요.")
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="최소 2개 이상의 파일이 필요합니다.")
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        paths: List[Path] = []
        for f in files:
            if not any(f.filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
                raise HTTPException(status_code=400, detail=f"지원하지 않는 파일 형식: {f.filename}")
            p = tmp_dir / f.filename
            with open(p, "wb") as b: shutil.copyfileobj(f.file, b)
            paths.append(p)
        sims = compute_similarities(embedder, paths)
        names = [p.name for p in paths]
        top_pairs = get_top_similar_pairs(sims, names)
        return ImageSimilarityResponse(
            similarity_matrix=sims.tolist(), image_names=names, top_pairs=top_pairs, total_images=len(paths)
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    uvicorn.run(
        "image_embedding_api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
