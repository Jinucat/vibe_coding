from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import base64
from typing import List
import uvicorn
import itertools
from pathlib import Path

app = FastAPI(title="Image Similarity Service", version="1.0.2")

# 전역: MediaPipe ImageEmbedder
embedder = None

def _resolve_model_path() -> str:
    """
    MODEL_PATH 환경변수 우선, 없으면 몇 가지 후보 경로에서 검색.
    """
    env = os.getenv("MODEL_PATH")
    if env and os.path.exists(env):
        return env

    here = Path(__file__).resolve()
    candidates = [
        here.parent / "models" / "mobilenet_v3_small.tflite",
        here.parent.parent / "models" / "mobilenet_v3_small.tflite",
        here.parent.parent.parent / "models" / "mobilenet_v3_small.tflite",
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    raise FileNotFoundError(
        "모델 파일을 찾을 수 없습니다. MODEL_PATH 환경변수로 절대경로를 지정하거나 "
        "services/models 또는 ai_services/models 하위에 mobilenet_v3_small.tflite 를 두세요."
    )

def load_model():
    """TFLite 임베더 모델 로드"""
    global embedder
    model_path = _resolve_model_path()

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ImageEmbedderOptions(
        base_options=base_options,
        l2_normalize=True,   # 벡터 L2 정규화
        quantize=True        # 양자화 결과가 올 수도 있음
    )
    embedder = vision.ImageEmbedder.create_from_options(options)

def _to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="유효하지 않은 이미지입니다.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def compute_embedding(image_rgb: np.ndarray) -> np.ndarray:
    """
    이미지(RGB) → 임베딩 벡터(np.ndarray, float32, L2 정규화).
    mediapipe가 embedding(list/ndarray[float]) 또는 quantized_embedding(bytes)을 줄 수 있음.
    """
    if embedder is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다.")

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = embedder.embed(mp_image)

    if not result.embeddings:
        raise HTTPException(status_code=500, detail="임베딩을 생성하지 못했습니다.")

    emb = result.embeddings[0]

    # ❗️ ndarray를 진리값으로 평가하면 안 됨 → 존재/길이로 체크
    vec_list = getattr(emb, "embedding", None)
    quant = getattr(emb, "quantized_embedding", None)

    vec = None
    if vec_list is not None:
        try:
            if len(vec_list) > 0:
                vec = np.asarray(vec_list, dtype=np.float32)
        except TypeError:
            # len() 불가 타입 방지(희박)
            pass

    if vec is None and quant is not None:
        if len(quant) > 0:
            q = np.frombuffer(quant, dtype=np.uint8).astype(np.float32)
            vec = q

    if vec is None:
        raise HTTPException(status_code=500, detail="임베딩 벡터가 비어 있습니다.")

    # 최종 L2 정규화(안전)
    n = float(np.linalg.norm(vec))
    if not np.isfinite(n) or n == 0.0:
        return vec.astype(np.float32)
    return (vec / n).astype(np.float32)

@app.on_event("startup")
async def startup_event():
    try:
        load_model()
        print("이미지 유사도 모델이 성공적으로 로드되었습니다.")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        raise

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "image-similarity-service"}

@app.post("/embed")
async def get_embedding(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_rgb = _to_rgb(image)

        emb = compute_embedding(image_rgb)  # np.ndarray(float32)
        return {
            "success": True,
            "embedding": emb.tolist(),
            "embedding_size": int(emb.shape[0]),
            "image_info": {"width": image.shape[1], "height": image.shape[0]},
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"임베딩 계산 중 오류가 발생했습니다: {str(e)}")

@app.post("/embed_base64")
async def get_embedding_base64(image_data: dict):
    try:
        image_base64 = image_data.get("image")
        if not image_base64:
            raise HTTPException(status_code=400, detail="이미지 데이터가 필요합니다.")
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_rgb = _to_rgb(image)

        emb = compute_embedding(image_rgb)
        return {
            "success": True,
            "embedding": emb.tolist(),
            "embedding_size": int(emb.shape[0]),
            "image_info": {"width": image.shape[1], "height": image.shape[0]},
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"임베딩 계산 중 오류가 발생했습니다: {str(e)}")

@app.post("/compare")
async def compare_images(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """두 이미지의 유사도(코사인) 비교"""
    try:
        c1 = await file1.read()
        c2 = await file2.read()

        img1 = cv2.imdecode(np.frombuffer(c1, np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.frombuffer(c2, np.uint8), cv2.IMREAD_COLOR)
        img1_rgb = _to_rgb(img1)
        img2_rgb = _to_rgb(img2)

        emb1 = compute_embedding(img1_rgb)
        emb2 = compute_embedding(img2_rgb)

        sim = _cosine(emb1, emb2)
        return {
            "success": True,
            "similarity": float(sim),
            "similarity_percentage": float(sim * 100.0),
            "image1_info": {"width": img1.shape[1], "height": img1.shape[0]},
            "image2_info": {"width": img2.shape[1], "height": img2.shape[0]},
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 비교 중 오류가 발생했습니다: {str(e)}")

@app.post("/compare_base64")
async def compare_images_base64(comparison_data: dict):
    """Base64 두 이미지 유사도 비교"""
    try:
        b64_1 = comparison_data.get("image1")
        b64_2 = comparison_data.get("image2")
        if not b64_1 or not b64_2:
            raise HTTPException(status_code=400, detail="두 이미지 데이터가 모두 필요합니다.")

        if "," in b64_1: b64_1 = b64_1.split(",")[1]
        if "," in b64_2: b64_2 = b64_2.split(",")[1]

        img1 = cv2.imdecode(np.frombuffer(base64.b64decode(b64_1), np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.frombuffer(base64.b64decode(b64_2), np.uint8), cv2.IMREAD_COLOR)
        img1_rgb = _to_rgb(img1)
        img2_rgb = _to_rgb(img2)

        emb1 = compute_embedding(img1_rgb)
        emb2 = compute_embedding(img2_rgb)

        sim = _cosine(emb1, emb2)
        return {
            "success": True,
            "similarity": float(sim),
            "similarity_percentage": float(sim * 100.0),
            "image1_info": {"width": img1.shape[1], "height": img1.shape[0]},
            "image2_info": {"width": img2.shape[1], "height": img2.shape[0]},
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 비교 중 오류가 발생했습니다: {str(e)}")

@app.post("/compare_multiple")
async def compare_multiple_images(files: List[UploadFile] = File(...)):
    """여러 이미지의 유사도 행렬 및 상위 쌍 반환"""
    try:
        if len(files) < 2:
            raise HTTPException(status_code=400, detail="최소 2개의 이미지가 필요합니다.")

        images_rgb: List[np.ndarray] = []
        embeddings: List[np.ndarray] = []

        for i, f in enumerate(files):
            contents = await f.read()
            img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
            img_rgb = _to_rgb(img)
            images_rgb.append(img_rgb)
            embeddings.append(compute_embedding(img_rgb))

        n = len(embeddings)
        sim_mat = np.eye(n, dtype=np.float32)
        for i, j in itertools.combinations(range(n), 2):
            s = _cosine(embeddings[i], embeddings[j])
            sim_mat[i, j] = sim_mat[j, i] = s

        pairs = [
            {
                "image1_index": i,
                "image2_index": j,
                "similarity": float(sim_mat[i, j]),
                "similarity_percentage": float(sim_mat[i, j] * 100.0),
            }
            for i, j in itertools.combinations(range(n), 2)
        ]
        pairs.sort(key=lambda x: x["similarity"], reverse=True)

        return {
            "success": True,
            "image_count": n,
            "similarity_matrix": sim_mat.tolist(),
            "top_pairs": pairs[: min(5, len(pairs))],
            "image_info": [
                {"index": i, "width": images_rgb[i].shape[1], "height": images_rgb[i].shape[0]}
                for i in range(n)
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"다중 이미지 비교 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    # image-similarity-service는 8002 포트로 띄우는 게 맞습니다.
    uvicorn.run(app, host="0.0.0.0", port=8002)
