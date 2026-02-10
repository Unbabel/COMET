#!/usr/bin/env python3
"""
COMET 학습 데이터 점수 분포 분석 및 리밸런싱 스크립트
====================================================

점수 분포의 편향이 COMET 학습에 미치는 영향:
  - 고점수 구간(0.7-1.0)에 데이터가 집중되면 모델이 해당 범위 주변으로만 예측
  - 저점수 구간(0.0-0.3)이 부족하면 나쁜 번역을 식별하지 못함
  - "score distribution collapse" 발생: 모델 출력이 평균 근처로 수렴

해결 전략:
  1. Stratified Sampling: 점수 구간별 균등 샘플링
  2. Inverse-Frequency Weighting: 희소 구간 가중치 증가
  3. Equal-Mass Binning: 동일 샘플 수 구간 분할

참고 논문:
  - "Pitfalls and Outlooks in Using COMET" (WMT 2024, arXiv:2408.15366)
    → COMET은 학습 데이터 점수 분포에 매우 민감
  - "Delving into Deep Imbalanced Regression" (ICML 2021)
    → Label Distribution Smoothing (LDS) 기법 제안
  - "COMET for Low-Resource MT Evaluation" (LREC 2024)
    → "COMET is highly susceptible to the distribution of scores"
  - Kim et al. (ACM TALLIP 2020)
    → 불균형 QE 학습 데이터가 고품질 편향 점수 유발

사용법:
    # 분포 분석만
    python scripts/analyze_and_rebalance.py \
        --input data/en-ko-qe/referenceless_train.csv \
        --analyze_only

    # 균등 리밸런싱 (기본: 구간별 최소 샘플 수에 맞춤)
    python scripts/analyze_and_rebalance.py \
        --input data/en-ko-qe/referenceless_train.csv \
        --output data/en-ko-qe/referenceless_train_balanced.csv \
        --strategy equal

    # 역빈도 가중치 CSV 생성 (학습 시 가중치 적용)
    python scripts/analyze_and_rebalance.py \
        --input data/en-ko-qe/referenceless_train.csv \
        --output data/en-ko-qe/referenceless_train_weighted.csv \
        --strategy weighted

    # 소프트 리밸런싱 (과소 구간 오버샘플 + 과다 구간 언더샘플)
    python scripts/analyze_and_rebalance.py \
        --input data/en-ko-qe/referenceless_train.csv \
        --output data/en-ko-qe/referenceless_train_soft.csv \
        --strategy soft \
        --target_total 3000000
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd


# ============================================================
# 분석 함수
# ============================================================

def analyze_distribution(df: pd.DataFrame, name: str = "Dataset", n_bins: int = 10):
    """점수 분포 상세 분석"""
    scores = df["score"].values

    print(f"\n{'='*70}")
    print(f"  점수 분포 분석: {name}")
    print(f"  총 샘플 수: {len(df):,}")
    print(f"{'='*70}")

    # 기본 통계
    print(f"\n  [기본 통계]")
    print(f"  Mean:   {np.mean(scores):.4f}")
    print(f"  Std:    {np.std(scores):.4f}")
    print(f"  Median: {np.median(scores):.4f}")
    print(f"  Min:    {np.min(scores):.4f}")
    print(f"  Max:    {np.max(scores):.4f}")
    print(f"  Skew:   {pd.Series(scores).skew():.4f}")

    # 구간별 분포
    bin_edges = np.linspace(0, 1, n_bins + 1)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (scores >= bin_edges[i]) & (scores <= bin_edges[i + 1])
        else:
            mask = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
        counts[i] = mask.sum()

    max_count = max(counts)
    print(f"\n  [구간별 분포]")
    print(f"  {'구간':>10}  {'샘플 수':>12}  {'비율':>7}  히스토그램")
    print(f"  {'-'*60}")

    for i in range(n_bins):
        pct = counts[i] / len(df) * 100
        bar_len = int(counts[i] / max_count * 40)
        bar = "█" * bar_len
        label = f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}"
        print(f"  {label:>10}  {counts[i]:>10,}  ({pct:>5.1f}%)  {bar}")

    # 불균형 정도 분석
    imbalance_ratio = max(counts) / max(min(counts), 1)
    print(f"\n  [불균형 지표]")
    print(f"  최대/최소 구간 비율: {imbalance_ratio:.1f}x")
    print(f"  최대 구간: {bin_edges[np.argmax(counts)]:.1f}-{bin_edges[np.argmax(counts)+1]:.1f} ({max(counts):,})")
    print(f"  최소 구간: {bin_edges[np.argmin(counts)]:.1f}-{bin_edges[np.argmin(counts)+1]:.1f} ({min(counts):,})")

    # 학습 영향 진단
    print(f"\n  [학습 영향 진단]")

    low_pct = counts[:3].sum() / len(df) * 100  # 0.0-0.3
    mid_pct = counts[3:7].sum() / len(df) * 100  # 0.3-0.7
    high_pct = counts[7:].sum() / len(df) * 100  # 0.7-1.0

    print(f"  저품질 (0.0-0.3): {counts[:3].sum():>10,} ({low_pct:>5.1f}%)")
    print(f"  중품질 (0.3-0.7): {counts[3:7].sum():>10,} ({mid_pct:>5.1f}%)")
    print(f"  고품질 (0.7-1.0): {counts[7:].sum():>10,} ({high_pct:>5.1f}%)")

    if high_pct > 50:
        print(f"\n  ⚠️  경고: 고품질 구간이 {high_pct:.0f}%로 과다 → 모델이 높은 점수로 편향될 위험")
    if low_pct < 10:
        print(f"  ⚠️  경고: 저품질 구간이 {low_pct:.1f}%로 부족 → 나쁜 번역 식별 능력 저하")
    if imbalance_ratio > 10:
        print(f"  ⚠️  경고: 구간 간 불균형 비율 {imbalance_ratio:.0f}x → 리밸런싱 강력 권장")

    return counts, bin_edges


# ============================================================
# 리밸런싱 전략
# ============================================================

def rebalance_equal(df: pd.DataFrame, n_bins: int = 10, seed: int = 42) -> pd.DataFrame:
    """
    전략 1: 완전 균등 리밸런싱
    각 구간에서 최소 구간의 샘플 수만큼만 샘플링합니다.

    장점: 완벽히 균등한 분포
    단점: 데이터 손실이 클 수 있음 (최소 구간에 맞춤)
    """
    np.random.seed(seed)
    bin_edges = np.linspace(0, 1, n_bins + 1)

    bin_dfs = []
    min_count = float("inf")

    # 각 구간의 데이터 분리 및 최소 카운트 찾기
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (df["score"] >= bin_edges[i]) & (df["score"] <= bin_edges[i + 1])
        else:
            mask = (df["score"] >= bin_edges[i]) & (df["score"] < bin_edges[i + 1])
        bin_df = df[mask]
        bin_dfs.append(bin_df)
        if len(bin_df) > 0:
            min_count = min(min_count, len(bin_df))

    print(f"\n  [균등 리밸런싱]")
    print(f"  각 구간 목표 샘플 수: {min_count:,}")
    print(f"  총 목표: {min_count * n_bins:,} (원본: {len(df):,})")

    # 각 구간에서 min_count만큼 샘플링
    result_dfs = []
    for i, bin_df in enumerate(bin_dfs):
        if len(bin_df) >= min_count:
            sampled = bin_df.sample(n=min_count, random_state=seed)
        else:
            # 오버샘플링 (복원 추출)
            sampled = bin_df.sample(n=min_count, replace=True, random_state=seed)
        result_dfs.append(sampled)

    return pd.concat(result_dfs, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def rebalance_soft(df: pd.DataFrame, target_total: int = 3000000,
                   n_bins: int = 10, seed: int = 42,
                   smoothing: float = 0.5) -> pd.DataFrame:
    """
    전략 2: 소프트 리밸런싱 (권장)
    과다 구간은 언더샘플링, 과소 구간은 오버샘플링하되
    완전 균등은 아닌 '부드러운' 균형을 만듭니다.

    smoothing=1.0: 완전 균등 (equal과 동일)
    smoothing=0.5: 제곱근 역빈도 (sqrt inverse frequency)
    smoothing=0.0: 원본 분포 유지

    장점: 데이터 손실 최소화하면서 분포 개선
    단점: 완벽히 균등하지는 않음
    """
    np.random.seed(seed)
    bin_edges = np.linspace(0, 1, n_bins + 1)

    bin_dfs = []
    bin_counts = []

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (df["score"] >= bin_edges[i]) & (df["score"] <= bin_edges[i + 1])
        else:
            mask = (df["score"] >= bin_edges[i]) & (df["score"] < bin_edges[i + 1])
        bin_df = df[mask]
        bin_dfs.append(bin_df)
        bin_counts.append(len(bin_df))

    bin_counts = np.array(bin_counts, dtype=float)
    bin_counts = np.maximum(bin_counts, 1)  # 0 방지

    # 역빈도 가중치 계산 (smoothing 적용)
    if smoothing > 0:
        # smoothing=0.5: sqrt(1/freq), smoothing=1.0: 1/freq
        weights = (1.0 / bin_counts) ** smoothing
    else:
        weights = bin_counts / bin_counts.sum()  # 원본 분포 유지

    # 가중치 정규화하여 목표 총 샘플 수에 맞춤
    weights = weights / weights.sum()
    target_per_bin = (weights * target_total).astype(int)

    print(f"\n  [소프트 리밸런싱] smoothing={smoothing}")
    print(f"  목표 총 샘플 수: {target_total:,}")
    print(f"  {'구간':>10}  {'원본':>10}  {'목표':>10}  {'비율변화':>10}")
    print(f"  {'-'*45}")
    for i in range(n_bins):
        label = f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}"
        ratio = target_per_bin[i] / max(bin_counts[i], 1)
        change = "▲ 오버샘플" if ratio > 1 else "▼ 언더샘플" if ratio < 1 else "= 유지"
        print(f"  {label:>10}  {int(bin_counts[i]):>10,}  {target_per_bin[i]:>10,}  {change}")

    # 샘플링 실행
    result_dfs = []
    for i, bin_df in enumerate(bin_dfs):
        target = target_per_bin[i]
        if target == 0:
            continue
        if len(bin_df) >= target:
            sampled = bin_df.sample(n=target, random_state=seed)
        else:
            # 오버샘플링 (복원 추출)
            sampled = bin_df.sample(n=target, replace=True, random_state=seed)
        result_dfs.append(sampled)

    return pd.concat(result_dfs, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def add_sample_weights(df: pd.DataFrame, n_bins: int = 10,
                       smoothing: float = 0.5) -> pd.DataFrame:
    """
    전략 3: 가중치 컬럼 추가
    데이터 자체는 변경하지 않고, sample_weight 컬럼을 추가합니다.
    학습 시 커스텀 loss에서 이 가중치를 사용합니다.

    장점: 원본 데이터 보존, 유연한 적용
    단점: COMET 코드 수정 필요 (커스텀 loss)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    scores = df["score"].values

    # 각 샘플의 구간 찾기
    bin_indices = np.digitize(scores, bin_edges[1:])  # 0-indexed
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # 구간별 카운트
    bin_counts = np.bincount(bin_indices, minlength=n_bins).astype(float)
    bin_counts = np.maximum(bin_counts, 1)

    # 역빈도 가중치
    bin_weights = (1.0 / bin_counts) ** smoothing
    bin_weights = bin_weights / bin_weights.mean()  # 평균=1로 정규화

    # 가중치 할당
    df = df.copy()
    df["sample_weight"] = bin_weights[bin_indices]

    print(f"\n  [가중치 부여] smoothing={smoothing}")
    print(f"  {'구간':>10}  {'샘플 수':>10}  {'가중치':>8}")
    print(f"  {'-'*35}")
    for i in range(n_bins):
        label = f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}"
        print(f"  {label:>10}  {int(bin_counts[i]):>10,}  {bin_weights[i]:>8.3f}")

    return df


# ============================================================
# 메인
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="COMET 학습 데이터 점수 분포 분석 및 리밸런싱"
    )
    parser.add_argument("--input", type=str, required=True, help="입력 CSV 경로")
    parser.add_argument("--output", type=str, help="출력 CSV 경로")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["equal", "soft", "weighted"],
        default="soft",
        help="리밸런싱 전략: equal(완전균등), soft(소프트, 권장), weighted(가중치 부여)",
    )
    parser.add_argument(
        "--target_total",
        type=int,
        default=3000000,
        help="소프트 리밸런싱 시 목표 총 샘플 수",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.5,
        help="리밸런싱 강도 (0=원본유지, 0.5=제곱근역빈도, 1.0=완전균등)",
    )
    parser.add_argument("--n_bins", type=int, default=10, help="점수 구간 수")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument(
        "--analyze_only", action="store_true", help="분석만 수행 (리밸런싱 없이)"
    )
    args = parser.parse_args()

    # 데이터 로드
    print(f"[INFO] Loading: {args.input}")
    df = pd.read_csv(args.input)
    if "score" not in df.columns:
        print("[ERROR] 'score' 컬럼이 없습니다.")
        sys.exit(1)

    # 분석
    counts, bin_edges = analyze_distribution(df, "원본 데이터", args.n_bins)

    if args.analyze_only:
        print("\n[DONE] 분석 완료 (--analyze_only)")
        return

    if not args.output:
        print("[ERROR] --output 경로를 지정해주세요.")
        sys.exit(1)

    # 리밸런싱 실행
    if args.strategy == "equal":
        result = rebalance_equal(df, n_bins=args.n_bins, seed=args.seed)
    elif args.strategy == "soft":
        result = rebalance_soft(
            df,
            target_total=args.target_total,
            n_bins=args.n_bins,
            seed=args.seed,
            smoothing=args.smoothing,
        )
    elif args.strategy == "weighted":
        result = add_sample_weights(df, n_bins=args.n_bins, smoothing=args.smoothing)
    else:
        print(f"[ERROR] Unknown strategy: {args.strategy}")
        sys.exit(1)

    # 결과 분석
    analyze_distribution(result, f"리밸런싱 후 ({args.strategy})", args.n_bins)

    # 저장
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"\n[SAVED] {args.output} ({len(result):,} rows)")

    print(f"\n{'='*70}")
    print(f"  리밸런싱 전략 비교 가이드")
    print(f"{'='*70}")
    print(f"  equal:    각 구간 동일 샘플 수 → 가장 공격적, 데이터 손실 큼")
    print(f"  soft:     소프트 리밸런싱 → 권장, 균형과 데이터 보존 타협")
    print(f"  weighted: 가중치 추가 → 데이터 손실 없음, 코드 수정 필요")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
