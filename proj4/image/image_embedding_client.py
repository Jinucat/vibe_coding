#!/usr/bin/env python3
"""
이미지 임베딩 유사도 API 클라이언트 (자동 플로우)
- /health 확인 → (필요시) /load-model → /upload-images(keep_temp=true) → saved_paths로 유사도/배치 호출
"""

import os
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional

DEFAULT_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8001")
DEFAULT_MODEL_PATH = os.getenv("IMAGE_EMBEDDER_PATH", r"models\mobilenet_v3_small.tflite")
DEFAULT_IMAGES = [
    os.getenv("IMG1", r"C:\Users\201\dev\proj1\images\burger.jpg"),
    os.getenv("IMG2", r"C:\Users\201\dev\proj1\images\burger_crop.jpg"),
]

class ImageEmbeddingClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def _get(self, path: str, **kw):
        r = self.session.get(f"{self.base_url}{path}", **kw); r.raise_for_status(); return r.json()
    def _post_json(self, path: str, data: dict, **kw):
        r = self.session.post(f"{self.base_url}{path}", json=data, **kw); r.raise_for_status(); return r.json()
    def _post_form(self, path: str, data: dict = None, files=None, **kw):
        r = self.session.post(f"{self.base_url}{path}", data=data, files=files, **kw); r.raise_for_status(); return r.json()

    def health(self) -> Dict[str, Any]:
        return self._get("/health")

    def load_model(self, model_path: str, l2_normalize: bool = True, quantize: bool = True) -> Dict[str, Any]:
        data = {"model_path_str": model_path, "l2_normalize": str(l2_normalize).lower(), "quantize": str(quantize).lower()}
        return self._post_form("/load-model", data=data)

    def upload_images(self, image_paths: List[str], keep_temp: bool = True) -> Dict[str, Any]:
        files, opened = [], []
        try:
            for p in image_paths:
                fp = Path(p)
                if not fp.exists():
                    raise FileNotFoundError(f"파일을 찾을 수 없습니다: {p}")
                f = open(fp, "rb")
                opened.append(f)
                files.append(("files", (fp.name, f, "application/octet-stream")))
            data = {"keep_temp": str(keep_temp).lower()}
            return self._post_form("/upload-images", data=data, files=files)
        finally:
            for f in opened:
                try: f.close()
                except: pass

    def similarity_single(self, p1: str, p2: str, l2_normalize=True, quantize=True) -> Dict[str, Any]:
        payload = {"image1_path": p1, "image2_path": p2, "l2_normalize": l2_normalize, "quantize": quantize}
        return self._post_json("/similarity/single", payload)

    def similarity_batch(self, paths: List[str], l2_normalize=True, quantize=True) -> Dict[str, Any]:
        payload = {"image_paths": paths, "l2_normalize": l2_normalize, "quantize": quantize}
        return self._post_json("/similarity/batch", payload)

    def upload_and_compare_single(self, local1: str, local2: str) -> Dict[str, Any]:
        files, opened = [], []
        try:
            for p in (local1, local2):
                fp = Path(p)
                if not fp.exists():
                    raise FileNotFoundError(f"파일을 찾을 수 없습니다: {p}")
                f = open(fp, "rb")
                opened.append(f)
                files.append(("file1" if len(files) == 0 else "file2", (fp.name, f, "application/octet-stream")))
            return self._post_form("/similarity/upload-single", files=files)
        finally:
            for f in opened:
                try: f.close()
                except: pass

def print_matrix(mat: List[List[float]], names: List[str]):
    n = len(names); w = max(len(nm) for nm in names) + 2
    print("\n[유사도 행렬]")
    print(" " * w + " ".join(f"{i:>8}" for i in range(n)))
    for i, row in enumerate(mat):
        print(f"{str(i).rjust(w-2)}  " + " ".join(f"{v:>8.4f}" for v in row))

def main():
    print("🚀 이미지 임베딩 유사도 API 클라이언트")
    c = ImageEmbeddingClient(DEFAULT_BASE_URL)

    # 1) 서버/모델 상태
    try:
        h = c.health()
        print(f"✅ 서버 OK | 모델 로드됨: {h.get('model_loaded')} | MediaPipe: {h.get('mediapipe_available')}")
    except Exception as e:
        print(f"❌ 서버 연결 실패: {e}\n   → 서버 실행: python image_embedding_api.py")
        return

    # 2) 자동 모델 로드
    try:
        if not h.get("model_loaded"):
            print(f"ℹ️  모델 자동 로드: {DEFAULT_MODEL_PATH}")
            res = c.load_model(DEFAULT_MODEL_PATH)
            print(f"   → {res.get('message')}: {res.get('model_path')}")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    # 3) 기본 이미지 확인
    imgs = []
    for p in DEFAULT_IMAGES:
        if Path(p).exists():
            imgs.append(p)
        else:
            print(f"⚠️  파일 없음: {p}")
    if len(imgs) < 2:
        print("❌ 기본 이미지 2개가 필요합니다. DEFAULT_IMAGES를 수정하세요.")
        return

    # 4) 가장 간단한 방법: 원스텝 업로드+유사도
    try:
        print("🔁 원스텝: /similarity/upload-single")
        sim = c.upload_and_compare_single(imgs[0], imgs[1])
        print(f"   → {sim['image1_name']} ↔ {sim['image2_name']} : {sim['similarity']:.4f} ({sim['percentage']})")
    except Exception as e:
        print(f"❌ 원스텝 유사도 실패: {e}")

    # 5) 업로드(keep_temp) → saved_paths로 배치 유사도
    try:
        print("⬆️ 업로드(keep_temp=true) → saved_paths 사용")
        up = c.upload_images(imgs, keep_temp=True)
        saved = up.get("saved_paths", [])
        if len(saved) >= 2:
            batch = c.similarity_batch(saved)
            print(f"✅ 배치 유사도 | 총 {batch['total_images']}개")
            print_matrix(batch["similarity_matrix"], batch["image_names"])
        else:
            print(f"❌ 업로드 결과 이상: {up}")
    except Exception as e:
        print(f"❌ 배치 유사도 실패: {e}")

    print("\n🎉 완료!")

if __name__ == "__main__":
    main()
