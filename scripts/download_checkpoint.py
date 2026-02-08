#!/usr/bin/env python3
"""
COMET 사전학습 모델 체크포인트 다운로드 스크립트
================================================

HuggingFace Hub 또는 레거시 S3에서 COMET 체크포인트를 다운로드합니다.

사용법:
    # HuggingFace 모델 다운로드 (권장)
    python scripts/download_checkpoint.py --model Unbabel/wmt22-cometkiwi-da

    # 레거시 모델 다운로드
    python scripts/download_checkpoint.py --model wmt21-comet-qe-da --legacy

    # 체크포인트 경로만 확인
    python scripts/download_checkpoint.py --model Unbabel/wmt22-cometkiwi-da --print_path
"""

import argparse
import os
import sys


def download_from_huggingface(model_name: str, output_dir: str) -> str:
    """HuggingFace Hub에서 모델 다운로드"""
    try:
        from comet import download_model
    except ImportError:
        print("[ERROR] comet 패키지를 먼저 설치하세요: pip install unbabel-comet")
        sys.exit(1)

    print(f"[INFO] Downloading {model_name} from HuggingFace Hub...")
    model_path = download_model(model_name)
    print(f"[INFO] Model downloaded to: {model_path}")
    return model_path


def download_legacy(model_name: str) -> str:
    """레거시 COMET 모델 다운로드 (S3)"""
    try:
        from comet import download_model
    except ImportError:
        print("[ERROR] comet 패키지를 먼저 설치하세요: pip install unbabel-comet")
        sys.exit(1)

    print(f"[INFO] Downloading legacy model: {model_name}...")
    model_path = download_model(model_name)
    print(f"[INFO] Model downloaded to: {model_path}")
    return model_path


def find_checkpoint(model_path: str) -> str:
    """다운로드된 모델 디렉토리에서 .ckpt 파일 찾기"""
    for root, dirs, files in os.walk(model_path):
        for f in files:
            if f.endswith(".ckpt"):
                return os.path.join(root, f)
    return None


def main():
    parser = argparse.ArgumentParser(description="COMET 체크포인트 다운로드")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="모델 이름 (예: Unbabel/wmt22-cometkiwi-da, wmt21-comet-qe-da)",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="레거시 모델 다운로드 (S3)",
    )
    parser.add_argument(
        "--print_path",
        action="store_true",
        help="체크포인트 경로만 출력",
    )
    args = parser.parse_args()

    if args.legacy:
        model_path = download_legacy(args.model)
    else:
        model_path = download_from_huggingface(args.model, "checkpoints")

    ckpt = find_checkpoint(model_path)
    if ckpt:
        print(f"\n[SUCCESS] Checkpoint file: {ckpt}")
        print(f"\n사용 예시:")
        print(f"  comet-train --cfg YOUR_CONFIG.yaml \\")
        print(f"      --load_from_checkpoint {ckpt}")
    else:
        print(f"\n[INFO] Model path: {model_path}")
        print(f"  (.ckpt 파일을 직접 확인하세요)")


if __name__ == "__main__":
    main()
