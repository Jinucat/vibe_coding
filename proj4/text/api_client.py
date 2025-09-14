#!/usr/bin/env python3
"""
FastAPI Sequence Classification API 클라이언트 테스트 스크립트
"""

import requests
import json
import time
from typing import List, Dict, Any

class SequenceClassificationClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """서버 상태 확인"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def classify_text(self, text: str, model_name: str = None, method: str = "pipeline") -> Dict[str, Any]:
        """단일 텍스트 분류"""
        data = {
            "text": text,
            "method": method
        }
        if model_name:
            data["model_name"] = model_name
        
        try:
            response = self.session.post(f"{self.base_url}/classify", json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def classify_batch(self, texts: List[str], model_name: str = None, method: str = "pipeline") -> Dict[str, Any]:
        """배치 텍스트 분류"""
        data = {
            "texts": texts,
            "method": method
        }
        if model_name:
            data["model_name"] = model_name
        
        try:
            response = self.session.post(f"{self.base_url}/classify/batch", json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_models_info(self) -> Dict[str, Any]:
        """사용 가능한 모델 정보"""
        try:
            response = self.session.get(f"{self.base_url}/models/info")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def main():
    print("🚀 Sequence Classification API 클라이언트 테스트")
    print("=" * 60)
    
    # 클라이언트 생성
    client = SequenceClassificationClient()
    
    # 1. 헬스 체크
    print("1️⃣ 서버 상태 확인...")
    health = client.health_check()
    if "error" in health:
        print(f"❌ 서버 연결 실패: {health['error']}")
        print("💡 서버가 실행 중인지 확인하세요: python api_server.py")
        return
    else:
        print("✅ 서버 연결 성공")
        print(f"   상태: {health['status']}")
        print(f"   디바이스: {health['device']}")
        print(f"   PyTorch: {health['torch_available']}")
        print(f"   Transformers: {health['transformers_available']}")
    
    print("\n" + "=" * 60)
    
    # 2. 모델 정보 확인
    print("2️⃣ 모델 정보 확인...")
    models_info = client.get_models_info()
    if "error" not in models_info:
        print("✅ 모델 정보 조회 성공")
        print(f"   기본 모델: {models_info['default_model']}")
        print(f"   지원 모델: {', '.join(models_info['supported_models'])}")
    else:
        print(f"❌ 모델 정보 조회 실패: {models_info['error']}")
    
    print("\n" + "=" * 60)
    
    # 3. 단일 텍스트 분류 테스트
    print("3️⃣ 단일 텍스트 분류 테스트...")
    test_texts = [
        "This movie is absolutely fantastic!",
        "I really hate this product, it's terrible.",
        "The weather is okay today.",
        "Amazing work! Great job!",
        "This is the worst experience ever."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 테스트 {i}: {text}")
        result = client.classify_text(text)
        if "error" not in result:
            print(f"   ✅ 결과: {result['label']} (신뢰도: {result['score']:.4f})")
        else:
            print(f"   ❌ 오류: {result['error']}")
    
    print("\n" + "=" * 60)
    
    # 4. 배치 분류 테스트
    print("4️⃣ 배치 텍스트 분류 테스트...")
    batch_result = client.classify_batch(test_texts)
    if "error" not in batch_result:
        print(f"✅ 배치 분류 완료")
        print(f"   총 텍스트: {batch_result['total_count']}")
        print(f"   성공: {batch_result['success_count']}")
        print(f"   실패: {batch_result['total_count'] - batch_result['success_count']}")
        
        print("\n📊 배치 결과:")
        for i, result in enumerate(batch_result['results'], 1):
            print(f"   {i}. {result['label']} ({result['score']:.4f})")
    else:
        print(f"❌ 배치 분류 실패: {batch_result['error']}")
    
    print("\n" + "=" * 60)
    
    # 5. 성능 테스트
    print("5️⃣ 성능 테스트...")
    start_time = time.time()
    for _ in range(10):
        client.classify_text("This is a performance test.")
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 10
    print(f"✅ 평균 응답 시간: {avg_time:.3f}초")
    
    print("\n🎉 모든 테스트 완료!")

if __name__ == "__main__":
    main()
