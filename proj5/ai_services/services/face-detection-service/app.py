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

app = FastAPI(title="Face Detection Service", version="1.0.0")

# 전역 변수로 모델 로드
detector = None

def load_model():
    """모델을 로드합니다."""
    global detector
    model_path = os.getenv('MODEL_PATH', '../../models/blaze_face_short_range.tflite')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> tuple:
    """정규화된 좌표를 픽셀 좌표로 변환합니다."""
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or np.isclose(0, value)) and (value < 1 or np.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        return None
    
    x_px = min(int(normalized_x * image_width), image_width - 1)
    y_px = min(int(normalized_y * image_height), image_height - 1)
    return x_px, y_px

def process_face_detection(image_data: np.ndarray) -> List[Dict[str, Any]]:
    """얼굴 검출을 수행합니다."""
    if detector is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다.")
    
    # MediaPipe Image 객체 생성
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_data)
    
    # 얼굴 검출 수행
    detection_result = detector.detect(mp_image)
    
    faces = []
    height, width = image_data.shape[:2]
    
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        face_data = {
            "bounding_box": {
                "x": int(bbox.origin_x),
                "y": int(bbox.origin_y),
                "width": int(bbox.width),
                "height": int(bbox.height)
            },
            "confidence": float(detection.categories[0].score),
            "keypoints": []
        }
        
        # 키포인트 추출
        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(
                keypoint.x, keypoint.y, width, height
            )
            if keypoint_px:
                face_data["keypoints"].append({
                    "x": keypoint_px[0],
                    "y": keypoint_px[1]
                })
        
        faces.append(face_data)
    
    return faces

@app.on_event("startup")
async def startup_event():
    """서비스 시작 시 모델을 로드합니다."""
    try:
        load_model()
        print("얼굴 검출 모델이 성공적으로 로드되었습니다.")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        raise

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "healthy", "service": "face-detection-service"}

@app.post("/detect")
async def detect_faces(file: UploadFile = File(...)):
    """얼굴 검출 API"""
    try:
        # 이미지 파일 읽기
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="유효하지 않은 이미지 파일입니다.")
        
        # BGR을 RGB로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 얼굴 검출 수행
        faces = process_face_detection(image_rgb)
        
        return {
            "success": True,
            "faces_count": len(faces),
            "faces": faces,
            "image_info": {
                "width": image.shape[1],
                "height": image.shape[0]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"얼굴 검출 중 오류가 발생했습니다: {str(e)}")

@app.post("/detect_base64")
async def detect_faces_base64(image_data: dict):
    """Base64 인코딩된 이미지로 얼굴 검출"""
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
        
        # 얼굴 검출 수행
        faces = process_face_detection(image_rgb)
        
        return {
            "success": True,
            "faces_count": len(faces),
            "faces": faces,
            "image_info": {
                "width": image.shape[1],
                "height": image.shape[0]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"얼굴 검출 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
