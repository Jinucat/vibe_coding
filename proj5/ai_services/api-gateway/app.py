# api-gateway/app.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from contextlib import asynccontextmanager
import httpx
import os
import base64
import asyncio
from typing import List, Optional

APP_TITLE = "AI Services API Gateway"
APP_DESC = "AI 서비스들을 통합하는 API Gateway (로컬 개발용 기본값 포함)"
APP_VER = "1.2.0"

# ===== 서비스 URL 기본값 (로컬 개발 우선) =====
FACE_SERVICE_URL = os.getenv("FACE_SERVICE_URL", "http://127.0.0.1:8001")
OBJECT_SERVICE_URL = os.getenv("OBJECT_SERVICE_URL", "http://127.0.0.1:8002")
SIMILARITY_SERVICE_URL = os.getenv("SIMILARITY_SERVICE_URL", "http://127.0.0.1:8003")

# CORS 허용 범위 (로컬 개발은 * 권장 / 운영은 도메인 제한)
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*").split(",")

# httpx 타임아웃(초) — 헬스체크는 더 짧게
HTTPX_TIMEOUT = float(os.getenv("HTTPX_TIMEOUT", "30"))

templates = Jinja2Templates(directory="templates")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 앱 전체에서 재사용할 HTTP 클라이언트
    async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
        app.state.client = client
        yield
        # 종료 시 자동 close

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESC,
    version=APP_VER,
    lifespan=lifespan,
)

# ===== 미들웨어 / 정적 리소스 =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS if ALLOW_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일: /static
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ===== 공통 유틸 =====
def _to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")

async def _svc_get(url: str, timeout: Optional[float] = None) -> httpx.Response:
    client: httpx.AsyncClient = app.state.client
    return await client.get(url, timeout=timeout)

async def _svc_post_json(url: str, payload: dict, timeout: Optional[float] = None) -> httpx.Response:
    client: httpx.AsyncClient = app.state.client
    return await client.post(url, json=payload, timeout=timeout)

# ===== 루트 & 파비콘 =====
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    index_path = os.path.join("templates", "index.html")
    if os.path.isfile(index_path):
        return templates.TemplateResponse("index.html", {"request": request})
    html = f"""
    <html><body style="font-family:Arial; padding:24px; color:#222">
      <h1>{APP_TITLE}</h1>
      <p>문서: <a href="/docs">/docs</a></p>
      <pre>{{
  "services": {{
    "face_detection": "/api/v1/face-detection",
    "object_detection": "/api/v1/object-detection",
    "image_similarity": "/api/v1/image-similarity"
  }}
}}</pre>
    </body></html>
    """
    return HTMLResponse(html)

@app.get("/favicon.ico")
async def favicon():
    # 404 소음 제거용
    return Response(status_code=204)

# ===== 헬스 체크 =====
@app.get("/health")
async def health_check():
    """전체 서비스 헬스 체크(동시요청 + 개별 타임아웃으로 절대 멈추지 않음)"""
    services = {
        "face-detection": FACE_SERVICE_URL,
        "object-detection": OBJECT_SERVICE_URL,
        "image-similarity": SIMILARITY_SERVICE_URL,
    }

    async def check_one(name: str, base: str):
        try:
            # 헬스는 빠르게: 2초 타임아웃
            r = await _svc_get(f"{base}/health", timeout=2.0)
            return name, {
                "status": "healthy" if r.status_code == 200 else "unhealthy",
                "code": r.status_code,
            }
        except Exception as e:
            return name, {"status": "unhealthy", "error": str(e)}

    results = await asyncio.gather(*[check_one(n, u) for n, u in services.items()])
    status_map = dict(results)
    overall = "healthy" if all(v.get("status") == "healthy" for v in status_map.values()) else "unhealthy"
    return {"overall_status": overall, "services": status_map}

# ===== 얼굴 검출 =====
@app.post("/api/v1/face-detection/detect")
async def detect_faces(file: UploadFile = File(...)):
    """이미지 파일 업로드 → 얼굴 서비스 /detect_base64 프록시"""
    try:
        contents = await file.read()
        image_b64 = _to_b64(contents)
        r = await _svc_post_json(f"{FACE_SERVICE_URL}/detect_base64", {"image": image_b64})
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        return r.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"얼굴 검출 중 오류: {e}")

@app.post("/api/v1/face-detection/detect_base64")
async def detect_faces_base64(image_data: dict):
    """Base64 이미지 → 얼굴 서비스 /detect_base64 프록시"""
    try:
        r = await _svc_post_json(f"{FACE_SERVICE_URL}/detect_base64", image_data)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        return r.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"얼굴 검출 중 오류: {e}")

# 별칭(프런트가 /detect/face 경로를 사용할 때 호환)
@app.post("/detect/face")
async def detect_face_alias(file: UploadFile = File(...)):
    return await detect_faces(file=file)

# ===== 객체 검출 =====
@app.post("/api/v1/object-detection/detect")
async def detect_objects(file: UploadFile = File(...)):
    """이미지 파일 업로드 → 객체 서비스 /detect_base64 프록시"""
    try:
        contents = await file.read()
        image_b64 = _to_b64(contents)
        r = await _svc_post_json(f"{OBJECT_SERVICE_URL}/detect_base64", {"image": image_b64})
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        return r.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"객체 검출 중 오류: {e}")

@app.post("/api/v1/object-detection/detect_base64")
async def detect_objects_base64(image_data: dict):
    """Base64 이미지 → 객체 서비스 /detect_base64 프록시"""
    try:
        r = await _svc_post_json(f"{OBJECT_SERVICE_URL}/detect_base64", image_data)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        return r.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"객체 검출 중 오류: {e}")

# 별칭(프런트가 /detect/object 경로를 사용할 때 호환)
@app.post("/detect/object")
async def detect_object_alias(file: UploadFile = File(...)):
    return await detect_objects(file=file)

# ===== 이미지 유사도 =====
@app.post("/api/v1/image-similarity/embed")
async def get_embedding(file: UploadFile = File(...)):
    """이미지 파일 업로드 → 유사도 서비스 /embed_base64 프록시"""
    try:
        contents = await file.read()
        image_b64 = _to_b64(contents)
        r = await _svc_post_json(f"{SIMILARITY_SERVICE_URL}/embed_base64", {"image": image_b64})
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        return r.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"임베딩 계산 중 오류: {e}")

@app.post("/api/v1/image-similarity/compare")
async def compare_images(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """두 이미지 유사도 비교 → 유사도 서비스 /compare_base64 프록시"""
    try:
        b64_1 = _to_b64(await file1.read())
        b64_2 = _to_b64(await file2.read())
        payload = {"image1": b64_1, "image2": b64_2}
        r = await _svc_post_json(f"{SIMILARITY_SERVICE_URL}/compare_base64", payload)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        return r.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 비교 중 오류: {e}")

@app.post("/api/v1/image-similarity/compare_base64")
async def compare_images_base64(comparison_data: dict):
    """Base64 2장 비교 → 유사도 서비스 /compare_base64 프록시"""
    try:
        r = await _svc_post_json(f"{SIMILARITY_SERVICE_URL}/compare_base64", comparison_data)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        return r.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 비교 중 오류: {e}")

@app.post("/api/v1/image-similarity/compare_multiple")
async def compare_multiple_images(files: List[UploadFile] = File(...)):
    """
    여러 이미지 유사도 비교 프록시.
    게이트웨이에서 base64 리스트로 변환해 /compare_multiple_base64 로 전달한다고 가정.
    (백엔드가 multipart(files=[])만 받는다면 이 부분만 변경)
    """
    try:
        if len(files) < 2:
            raise HTTPException(status_code=400, detail="최소 2개의 이미지가 필요합니다.")

        images_b64 = [_to_b64(await f.read()) for f in files]
        payload = {"images": images_b64}

        url = f"{SIMILARITY_SERVICE_URL}/compare_multiple_base64"
        r = await _svc_post_json(url, payload)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        return r.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"다중 이미지 비교 중 오류: {e}")

# 별칭(프런트가 /similarity/compare 경로를 사용할 때 호환)
@app.post("/similarity/compare")
async def similarity_compare_alias(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    return await compare_images(file1=file1, file2=file2)

# ===== 통합 분석 (얼굴 + 객체) — 병렬 수행 =====
@app.post("/api/v1/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """한 장 이미지로 얼굴/객체 검출 병렬 수행 후 결과 합치기"""
    try:
        contents = await file.read()
        image_b64 = _to_b64(contents)

        face_task = _svc_post_json(f"{FACE_SERVICE_URL}/detect_base64", {"image": image_b64})
        obj_task  = _svc_post_json(f"{OBJECT_SERVICE_URL}/detect_base64", {"image": image_b64})

        face_resp, obj_resp = await asyncio.gather(face_task, obj_task)

        face_json = face_resp.json() if face_resp.status_code == 200 else {"error": f"face:{face_resp.status_code}"}
        obj_json  = obj_resp.json()  if obj_resp.status_code  == 200 else {"error": f"object:{obj_resp.status_code}"}

        return {
            "success": True,
            "analysis": {
                "face_detection": face_json,
                "object_detection": obj_json
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 분석 중 오류: {e}")

@app.get("/__version")
def __version():
    return {"ver": APP_VER, "face": FACE_SERVICE_URL, "obj": OBJECT_SERVICE_URL, "sim": SIMILARITY_SERVICE_URL}

@app.get("/__routes")
def __routes():
    return [getattr(r, "path", None) for r in app.routes]


# ===== 로컬 실행 진입점 =====
if __name__ == "__main__":
    import uvicorn
    # 로컬 전용 실행
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
