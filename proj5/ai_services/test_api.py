#!/usr/bin/env python3
"""
AI MSA 서비스 API 테스트 스크립트
"""

import requests
import base64
import json
from pathlib import Path
import time

# API Gateway URL
API_GATEWAY_URL = "http://localhost:8000"

def test_health_check():
    """헬스 체크 테스트"""
    print("🔍 헬스 체크 테스트...")
    
    try:
        response = requests.get(f"{API_GATEWAY_URL}/health")
        if response.status_code == 200:
            print("✅ 헬스 체크 성공")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        else:
            print(f"❌ 헬스 체크 실패: {response.status_code}")
    except Exception as e:
        print(f"❌ 헬스 체크 오류: {e}")

def create_test_image():
    """테스트용 이미지 생성 (간단한 색상 이미지)"""
    import numpy as np
    import cv2
    
    # 간단한 테스트 이미지 생성
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:] = [100, 150, 200]  # 파란색 배경
    
    # 원 그리기 (얼굴 모양)
    cv2.circle(img, (150, 150), 80, (255, 255, 255), -1)
    cv2.circle(img, (130, 130), 10, (0, 0, 0), -1)  # 왼쪽 눈
    cv2.circle(img, (170, 130), 10, (0, 0, 0), -1)  # 오른쪽 눈
    cv2.ellipse(img, (150, 170), (20, 10), 0, 0, 180, (0, 0, 0), 2)  # 입
    
    # 이미지를 Base64로 인코딩
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return img_base64

def test_face_detection():
    """얼굴 검출 테스트"""
    print("\n🔍 얼굴 검출 테스트...")
    
    try:
        # 테스트 이미지 생성
        img_base64 = create_test_image()
        
        # API 호출
        response = requests.post(
            f"{API_GATEWAY_URL}/api/v1/face-detection/detect_base64",
            json={"image": img_base64}
        )
        
        if response.status_code == 200:
            print("✅ 얼굴 검출 성공")
            result = response.json()
            print(f"  - 검출된 얼굴 수: {result.get('faces_count', 0)}")
        else:
            print(f"❌ 얼굴 검출 실패: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ 얼굴 검출 오류: {e}")

def test_object_detection():
    """객체 검출 테스트"""
    print("\n🔍 객체 검출 테스트...")
    
    try:
        # 테스트 이미지 생성
        img_base64 = create_test_image()
        
        # API 호출
        response = requests.post(
            f"{API_GATEWAY_URL}/api/v1/object-detection/detect_base64",
            json={"image": img_base64}
        )
        
        if response.status_code == 200:
            print("✅ 객체 검출 성공")
            result = response.json()
            print(f"  - 검출된 객체 수: {result.get('objects_count', 0)}")
        else:
            print(f"❌ 객체 검출 실패: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ 객체 검출 오류: {e}")

def test_image_similarity():
    """이미지 유사도 테스트"""
    print("\n🔍 이미지 유사도 테스트...")
    
    try:
        # 두 개의 유사한 테스트 이미지 생성
        img1_base64 = create_test_image()
        img2_base64 = create_test_image()  # 동일한 이미지
        
        # API 호출
        response = requests.post(
            f"{API_GATEWAY_URL}/api/v1/image-similarity/compare_base64",
            json={
                "image1": img1_base64,
                "image2": img2_base64
            }
        )
        
        if response.status_code == 200:
            print("✅ 이미지 유사도 계산 성공")
            result = response.json()
            similarity = result.get('similarity', 0)
            print(f"  - 유사도: {similarity:.4f} ({similarity*100:.2f}%)")
        else:
            print(f"❌ 이미지 유사도 계산 실패: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ 이미지 유사도 계산 오류: {e}")

def test_integrated_analysis():
    """통합 분석 테스트"""
    print("\n🔍 통합 분석 테스트...")
    
    try:
        # 테스트 이미지 생성
        img_base64 = create_test_image()
        
        # 파일로 변환하여 multipart/form-data로 전송
        import io
        img_data = base64.b64decode(img_base64)
        
        files = {
            'file': ('test_image.jpg', io.BytesIO(img_data), 'image/jpeg')
        }
        
        # API 호출
        response = requests.post(
            f"{API_GATEWAY_URL}/api/v1/analyze",
            files=files
        )
        
        if response.status_code == 200:
            print("✅ 통합 분석 성공")
            result = response.json()
            print("  - 분석 결과:")
            print(f"    * 얼굴 검출: {result['analysis']['face_detection'].get('faces_count', 0)}개")
            print(f"    * 객체 검출: {result['analysis']['object_detection'].get('objects_count', 0)}개")
        else:
            print(f"❌ 통합 분석 실패: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ 통합 분석 오류: {e}")

def main():
    """메인 함수"""
    print("🧪 AI MSA 서비스 API 테스트")
    print("=" * 50)
    
    # 서비스가 시작될 때까지 대기
    print("⏳ 서비스 시작을 기다리는 중...")
    time.sleep(5)
    
    # 테스트 실행
    test_health_check()
    test_face_detection()
    test_object_detection()
    test_image_similarity()
    test_integrated_analysis()
    
    print("\n🎉 모든 테스트가 완료되었습니다!")

if __name__ == "__main__":
    main()
