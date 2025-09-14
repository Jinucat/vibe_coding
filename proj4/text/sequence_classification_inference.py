#!/usr/bin/env python3
"""
Sequence Classification 추론 스크립트 (깔끔 출력 모드)
- 기본: 쓸모없는 로그/프로그레스바 숨김
- --verbose 일 때만 최소한의 디버그 출력
"""

import os
# ---- 로그/프로그레스바/비전 의존 비활성화 (import 전에 설정) ----
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")      # torchvision 강제 비활성화
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")     # 다운로드 바 숨김
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")         # 허깅페이스 텔레메트리 비활성
os.environ.setdefault("PYTHONIOENCODING", "utf-8")             # 출력 인코딩 고정

import argparse
import warnings
import sys, traceback, logging

# ---- 파이썬 로거 레벨 전체 낮춤(기본 ERROR) ----
logging.getLogger().setLevel(logging.ERROR)
for name in ["transformers", "huggingface_hub", "urllib3", "filelock", "tqdm"]:
    logging.getLogger(name).setLevel(logging.ERROR)

# ---- 외부 라이브러리 임포트 ----
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# Transformers 로깅/프로그레스바 끄기
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    try:
        hf_logging.disable_progress_bar()
    except Exception:
        pass
except Exception:
    pass

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception as e:
    print("❌ Transformers 임포트 실패:", repr(e))
    traceback.print_exc()
    sys.exit(2)

warnings.filterwarnings("ignore")  # 파이썬 경고 숨김


def normalize_pipeline_device(dev):
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


def to_torch_device(dev):
    if not TORCH_AVAILABLE:
        return "cpu"
    if dev == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return dev


def load_model_and_tokenizer(model_name, device):
    """모델/토크나이저 로드 (조용히)"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        dev = to_torch_device(device)
        if TORCH_AVAILABLE:
            model = model.to(dev)
        return tokenizer, model, dev
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        traceback.print_exc()
        return None, None, None


def predict_with_pipeline(text, model_name, device):
    """Pipeline 추론 (조용히)"""
    try:
        dev_idx = normalize_pipeline_device(device)
        clf = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=dev_idx
        )
        return clf(text)
    except Exception as e:
        print(f"❌ Pipeline 추론 실패: {e}")
        traceback.print_exc()
        return None


def predict_manual(text, tokenizer, model, device):
    """수동 추론 (PyTorch 필요)"""
    try:
        if not TORCH_AVAILABLE:
            raise RuntimeError("manual 모드는 PyTorch가 필요합니다.")
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
        print(f"❌ 수동 추론 실패: {e}")
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Sequence Classification 추론 스크립트 (깔끔 출력)"
    )
    parser.add_argument("--text", type=str, required=True, help="분류할 텍스트")
    parser.add_argument(
        "--model", type=str,
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="모델명(기본: distilbert-base-uncased-finetuned-sst-2-english)"
    )
    parser.add_argument(
        "--method", type=str, choices=["pipeline", "manual"], default="pipeline",
        help="추론 방법 (pipeline/manual)"
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
        help="디바이스 (auto/cpu/cuda)"
    )
    parser.add_argument("--verbose", action="store_true", help="디버그 정보 출력")
    args = parser.parse_args()

    # verbose가 아니면 우리가 출력하는 최소 메시지만
    if args.verbose:
        print("🚀 Sequence Classification 추론 시작")
        print(f"📝 텍스트: {args.text}")
        print(f"🤖 모델: {args.model}")
        print(f"⚙️  방법: {args.method}")
        print(f"💻 디바이스: {args.device}")
        print("-" * 50)
        dev = to_torch_device(args.device)
        print(f"🔧 실제 사용 디바이스: {dev}")
        if TORCH_AVAILABLE:
            print(f"🔥 CUDA 사용 가능: {torch.cuda.is_available()}")

    if args.method == "pipeline":
        result = predict_with_pipeline(args.text, args.model, args.device)
        if result:
            item = result[0]
            print(f"{item['label']} {item['score']:.4f}")
        else:
            print("❌ 추론 실패")
    else:
        tokenizer, model, dev = load_model_and_tokenizer(args.model, args.device)
        if tokenizer and model:
            result = predict_manual(args.text, tokenizer, model, dev)
            if result:
                print(f"{result['label']} {result['score']:.4f}")
            else:
                print("❌ 추론 실패")
        else:
            print("❌ 모델 로딩 실패")


if __name__ == "__main__":
    main()
