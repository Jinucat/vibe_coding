#!/usr/bin/env python3
"""
AI MSA 서비스들을 시작하는 스크립트
개발 환경에서 개별 서비스를 실행할 때 사용
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def start_service(service_name, port, working_dir):
    """개별 서비스를 시작합니다."""
    print(f"🚀 {service_name} 서비스를 시작합니다... (포트: {port})")
    
    try:
        # 서비스 디렉토리로 이동
        os.chdir(working_dir)
        
        # Python 서비스 실행
        process = subprocess.Popen([
            sys.executable, "app.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print(f"✅ {service_name} 서비스가 시작되었습니다. (PID: {process.pid})")
        return process
        
    except Exception as e:
        print(f"❌ {service_name} 서비스 시작 실패: {e}")
        return None

def main():
    """메인 함수"""
    print("🎯 AI MSA 서비스 시작 스크립트")
    print("=" * 50)
    
    # 현재 디렉토리
    base_dir = Path(__file__).parent
    
    # 서비스 설정
    services = [
        {
            "name": "Face Detection Service",
            "port": 8001,
            "dir": base_dir / "services" / "face-detection-service"
        },
        {
            "name": "Object Detection Service", 
            "port": 8002,
            "dir": base_dir / "services" / "object-detection-service"
        },
        {
            "name": "Image Similarity Service",
            "port": 8003,
            "dir": base_dir / "services" / "image-similarity-service"
        },
        {
            "name": "API Gateway",
            "port": 8000,
            "dir": base_dir / "api-gateway"
        }
    ]
    
    processes = []
    
    try:
        # 각 서비스 시작
        for service in services:
            if not service["dir"].exists():
                print(f"❌ 서비스 디렉토리를 찾을 수 없습니다: {service['dir']}")
                continue
                
            process = start_service(service["name"], service["port"], service["dir"])
            if process:
                processes.append(process)
            
            # 서비스 간 간격
            time.sleep(2)
        
        print("\n🎉 모든 서비스가 시작되었습니다!")
        print("\n📋 서비스 정보:")
        print("  - API Gateway: http://localhost:8000")
        print("  - Face Detection: http://localhost:8001")
        print("  - Object Detection: http://localhost:8002")
        print("  - Image Similarity: http://localhost:8003")
        print("\n📖 API 문서: http://localhost:8000/docs")
        print("\n⏹️  서비스를 중지하려면 Ctrl+C를 누르세요.")
        
        # 서비스들이 실행되는 동안 대기
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 서비스를 중지합니다...")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        
    finally:
        # 모든 프로세스 종료
        for process in processes:
            if process and process.poll() is None:
                process.terminate()
                process.wait()
        
        print("✅ 모든 서비스가 중지되었습니다.")

if __name__ == "__main__":
    main()
