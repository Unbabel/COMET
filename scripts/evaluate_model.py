#!/usr/bin/env python3
"""
학습된 COMET 모델 평가 스크립트
================================

학습된 체크포인트를 사용하여 테스트 데이터에서 번역 품질 점수를 예측하고
성능 지표를 계산합니다.

사용법:
    # 기본 평가 (CSV 파일)
    python scripts/evaluate_model.py \
        --checkpoint lightning_logs/version_X/checkpoints/best.ckpt \
        --test_data data/en-ko-qe/referenceless_val.csv \
        --model_type referenceless

    # UnifiedMetric 평가
    python scripts/evaluate_model.py \
        --checkpoint lightning_logs/version_X/checkpoints/best.ckpt \
        --test_data data/en-ko-qe/unified_qe_val.csv \
        --model_type unified

    # 개별 문장 평가
    python scripts/evaluate_model.py \
        --checkpoint lightning_logs/version_X/checkpoints/best.ckpt \
        --src "activate a scanning directed acyclic graph" \
        --mt "스캐닝 지시 비순환 그래프를 활성화하는 것" \
        --model_type referenceless
"""

import argparse
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats


def load_model(checkpoint_path: str, model_type: str):
    """체크포인트에서 모델 로드"""
    if model_type == "referenceless":
        from comet.models import ReferencelessRegression
        model = ReferencelessRegression.load_from_checkpoint(checkpoint_path)
    elif model_type == "unified":
        from comet.models import UnifiedMetric
        model = UnifiedMetric.load_from_checkpoint(checkpoint_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.eval()
    return model


def evaluate_csv(model, test_path: str):
    """CSV 테스트 데이터 평가"""
    df = pd.read_csv(test_path)
    print(f"[INFO] Test data: {test_path} ({len(df)} samples)")

    # 모델에 필요한 형식으로 데이터 구성
    samples = []
    for _, row in df.iterrows():
        sample = {"src": str(row["src"]), "mt": str(row["mt"])}
        samples.append(sample)

    # 예측
    print("[INFO] Running predictions...")
    output = model.predict(samples, batch_size=32, gpus=1)
    predicted_scores = output.scores

    # 실제 점수가 있으면 상관계수 계산
    if "score" in df.columns:
        actual_scores = df["score"].values

        # Pearson 상관계수
        pearson_r, pearson_p = stats.pearsonr(actual_scores, predicted_scores)
        # Spearman 상관계수
        spearman_r, spearman_p = stats.spearmanr(actual_scores, predicted_scores)
        # Kendall tau
        kendall_tau, kendall_p = stats.kendalltau(actual_scores, predicted_scores)

        # MSE & MAE
        mse = np.mean((np.array(actual_scores) - np.array(predicted_scores)) ** 2)
        mae = np.mean(np.abs(np.array(actual_scores) - np.array(predicted_scores)))

        print(f"\n{'='*60}")
        print(f"[RESULTS] Evaluation Metrics")
        print(f"{'='*60}")
        print(f"  Pearson r:     {pearson_r:.4f} (p={pearson_p:.2e})")
        print(f"  Spearman rho:  {spearman_r:.4f} (p={spearman_p:.2e})")
        print(f"  Kendall tau:   {kendall_tau:.4f} (p={kendall_p:.2e})")
        print(f"  MSE:           {mse:.6f}")
        print(f"  MAE:           {mae:.6f}")
        print(f"{'='*60}")

        # 점수 분포 비교
        print(f"\n  Score Distribution:")
        print(f"  {'':>15}  {'Actual':>10}  {'Predicted':>10}")
        print(f"  {'Mean':>15}  {np.mean(actual_scores):>10.4f}  {np.mean(predicted_scores):>10.4f}")
        print(f"  {'Std':>15}  {np.std(actual_scores):>10.4f}  {np.std(predicted_scores):>10.4f}")
        print(f"  {'Min':>15}  {np.min(actual_scores):>10.4f}  {np.min(predicted_scores):>10.4f}")
        print(f"  {'Max':>15}  {np.max(actual_scores):>10.4f}  {np.max(predicted_scores):>10.4f}")

    else:
        print(f"\n[INFO] Predicted scores (no ground truth available):")
        print(f"  Mean:  {np.mean(predicted_scores):.4f}")
        print(f"  Std:   {np.std(predicted_scores):.4f}")
        print(f"  Min:   {np.min(predicted_scores):.4f}")
        print(f"  Max:   {np.max(predicted_scores):.4f}")

    # 결과를 CSV로 저장
    df["predicted_score"] = predicted_scores
    output_path = test_path.replace(".csv", "_predictions.csv")
    df.to_csv(output_path, index=False)
    print(f"\n[SAVED] Predictions saved to: {output_path}")

    return predicted_scores


def evaluate_single(model, src: str, mt: str):
    """단일 문장 쌍 평가"""
    samples = [{"src": src, "mt": mt}]
    output = model.predict(samples, batch_size=1, gpus=1)

    print(f"\n{'='*60}")
    print(f"[RESULT] Single Sentence Evaluation")
    print(f"{'='*60}")
    print(f"  Source: {src}")
    print(f"  MT:     {mt}")
    print(f"  Score:  {output.scores[0]:.4f}")
    print(f"{'='*60}")

    return output.scores[0]


def main():
    parser = argparse.ArgumentParser(description="COMET 모델 평가")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="모델 체크포인트 경로"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["referenceless", "unified"],
        default="referenceless",
        help="모델 타입",
    )
    parser.add_argument("--test_data", type=str, help="테스트 CSV 경로")
    parser.add_argument("--src", type=str, help="소스 문장 (단일 평가)")
    parser.add_argument("--mt", type=str, help="번역 문장 (단일 평가)")
    parser.add_argument(
        "--gpus", type=int, default=1, help="사용할 GPU 수"
    )

    args = parser.parse_args()

    # 모델 로드
    print(f"[INFO] Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, args.model_type)

    if args.test_data:
        evaluate_csv(model, args.test_data)
    elif args.src and args.mt:
        evaluate_single(model, args.src, args.mt)
    else:
        print("[ERROR] --test_data 또는 --src/--mt를 지정해주세요.")
        sys.exit(1)


if __name__ == "__main__":
    main()
