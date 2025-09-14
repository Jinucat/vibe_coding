from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import base64
from typing import List, Dict, Any
import uvicorn

app = FastAPI(title="Object Detection Service", version="1.0.0")

# 전역 변수로 모델 로드
detector = None

def load_model():
    """모델을 로드합니다."""
    global detector
    model_path = os.getenv('MODEL_PATH', '../../models/efficientdet_lite0.tflite')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        score_threshold=0.5
    )
    detector = vision.ObjectDetector.create_from_options(options)

def process_object_detection(image_data: np.ndarray) -> List[Dict[str, Any]]:
    """객체 검출을 수행합니다."""
    if detector is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다.")
    
    # MediaPipe Image 객체 생성
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_data)
    
    # 객체 검출 수행
    detection_result = detector.detect(mp_image)
    
    objects = []
    height, width = image_data.shape[:2]
    
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        category = detection.categories[0]
        
        object_data = {
            "bounding_box": {
                "x": int(bbox.origin_x),
                "y": int(bbox.origin_y),
                "width": int(bbox.width),
                "height": int(bbox.height)
            },
            "category": {
                "name": category.category_name or "Unknown",
                "confidence": float(category.score)
            }
        }
        
        objects.append(object_data)
    
    return objects

@app.on_event("startup")
async def startup_event():
    """서비스 시작 시 모델을 로드합니다."""
    try:
        load_model()
        print("객체 검출 모델이 성공적으로 로드되었습니다.")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        raise

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "healthy", "service": "object-detection-service"}

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """객체 검출 API"""
    try:
        # 이미지 파일 읽기
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="유효하지 않은 이미지 파일입니다.")
        
        # BGR을 RGB로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 객체 검출 수행
        objects = process_object_detection(image_rgb)
        
        return {
            "success": True,
            "objects_count": len(objects),
            "objects": objects,
            "image_info": {
                "width": image.shape[1],
                "height": image.shape[0]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"객체 검출 중 오류가 발생했습니다: {str(e)}")

@app.post("/detect_base64")
async def detect_objects_base64(image_data: dict):
    """Base64 인코딩된 이미지로 객체 검출"""
    try:
        # Base64 디코딩
        image_base64 = image_data.get("image")
        if not image_base64:
            raise HTTPException(status_code=400, detail="이미지 데이터가 필요합니다.")
        
        # data:image/jpeg;base64, 부분 제거
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        
        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="유효하지 않은 이미지 데이터입니다.")
        
        # BGR을 RGB로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 객체 검출 수행
        objects = process_object_detection(image_rgb)
        
        return {
            "success": True,
            "objects_count": len(objects),
            "objects": objects,
            "image_info": {
                "width": image.shape[1],
                "height": image.shape[0]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"객체 검출 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
