#!/usr/bin/env python3
"""
COMET Reference-Free 학습을 위한 데이터 전처리 스크립트
=====================================================

이 스크립트는 기존 학습 데이터(en-ko-qe-patent-balanced_*.csv)를
COMET의 ReferencelessRegression 및 UnifiedMetric 모델이 요구하는
형식으로 변환합니다.

사용법:
    python scripts/prepare_data.py \
        --input_dir /path/to/train_data \
        --output_dir data/en-ko-qe

필수 입력 파일:
    - en-ko-qe-patent-balanced_train.csv       (pointwise 학습 데이터)
    - en-ko-qe-patent-balanced_val.csv         (pointwise 검증 데이터)
    - en-ko-qe-patent-balanced_pairwise_train.csv  (pairwise 학습 데이터)
    - en-ko-qe-patent-balanced_pairwise_val.csv    (pairwise 검증 데이터)

출력 파일:
    - referenceless_train.csv   (src, mt, score)
    - referenceless_val.csv     (src, mt, score)
    - unified_qe_train.csv      (src, mt, score) - UnifiedMetric용
    - unified_qe_val.csv        (src, mt, score) - UnifiedMetric용
    - pairwise_expanded_train.csv (pairwise -> pointwise 변환)
    - combined_train.csv        (pointwise + pairwise 합쳐진 데이터)
"""

import argparse
import os
import sys

import pandas as pd
import numpy as np


def load_pointwise_data(filepath: str) -> pd.DataFrame:
    """Pointwise CSV 로드 (lp, src, mt, ref, score, ...)"""
    print(f"[INFO] Loading pointwise data: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  -> {len(df)} rows, columns: {list(df.columns)}")
    return df


def load_pairwise_data(filepath: str) -> pd.DataFrame:
    """Pairwise CSV 로드 (src, mt_good, mt_bad, ref, score_good, score_bad, ...)"""
    print(f"[INFO] Loading pairwise data: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  -> {len(df)} rows, columns: {list(df.columns)}")
    return df


def prepare_referenceless(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pointwise 데이터에서 ReferencelessRegression 학습에 필요한
    (src, mt, score) 컬럼만 추출합니다.

    ref 컬럼은 제거됩니다 (reference-free).
    """
    required_cols = ["src", "mt", "score"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    result = df[required_cols].copy()
    result["src"] = result["src"].astype(str)
    result["mt"] = result["mt"].astype(str)
    result["score"] = result["score"].astype(float)

    # NaN 또는 빈 문자열 제거
    result = result.dropna(subset=["src", "mt", "score"])
    result = result[result["src"].str.strip() != ""]
    result = result[result["mt"].str.strip() != ""]

    print(f"  -> Reference-free format: {len(result)} rows")
    return result


def expand_pairwise_to_pointwise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise 데이터 (src, mt_good, mt_bad, score_good, score_bad)를
    Pointwise 데이터 (src, mt, score)로 변환합니다.

    각 pairwise row에서 2개의 pointwise sample을 생성:
      1) (src, mt_good, score_good)
      2) (src, mt_bad, score_bad)
    """
    good_rows = df[["src", "mt_good", "score_good"]].rename(
        columns={"mt_good": "mt", "score_good": "score"}
    )
    bad_rows = df[["src", "mt_bad", "score_bad"]].rename(
        columns={"mt_bad": "mt", "score_bad": "score"}
    )

    result = pd.concat([good_rows, bad_rows], ignore_index=True)
    result["src"] = result["src"].astype(str)
    result["mt"] = result["mt"].astype(str)
    result["score"] = result["score"].astype(float)

    # NaN 또는 빈 문자열 제거
    result = result.dropna(subset=["src", "mt", "score"])
    result = result[result["src"].str.strip() != ""]
    result = result[result["mt"].str.strip() != ""]

    # 중복 제거 (같은 src-mt 쌍)
    before_dedup = len(result)
    result = result.drop_duplicates(subset=["src", "mt"], keep="first")
    print(f"  -> Expanded pairwise: {before_dedup} -> {len(result)} rows (after dedup)")
    return result


def analyze_data(df: pd.DataFrame, name: str) -> None:
    """데이터 기본 통계 출력"""
    print(f"\n{'='*60}")
    print(f"[DATA STATS] {name}")
    print(f"{'='*60}")
    print(f"  Total rows: {len(df)}")
    print(f"  Score distribution:")
    print(f"    mean:   {df['score'].mean():.4f}")
    print(f"    std:    {df['score'].std():.4f}")
    print(f"    min:    {df['score'].min():.4f}")
    print(f"    max:    {df['score'].max():.4f}")
    print(f"    median: {df['score'].median():.4f}")

    # Score 분포 구간별 집계
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    df["score_bin"] = pd.cut(df["score"], bins=bins, labels=labels, include_lowest=True)
    print(f"  Score bins:")
    for label in labels:
        count = len(df[df["score_bin"] == label])
        pct = count / len(df) * 100
        print(f"    {label}: {count:>8} ({pct:.1f}%)")
    df.drop(columns=["score_bin"], inplace=True)

    # Source/MT 문장 길이 통계
    src_lens = df["src"].str.split().str.len()
    mt_lens = df["mt"].str.split().str.len()
    print(f"  Source sentence length (words): mean={src_lens.mean():.1f}, max={src_lens.max()}")
    print(f"  MT sentence length (words):     mean={mt_lens.mean():.1f}, max={mt_lens.max()}")


def create_sampled_subset(df: pd.DataFrame, max_rows: int, seed: int = 42) -> pd.DataFrame:
    """대용량 데이터에서 균등 샘플링으로 부분 데이터셋 생성"""
    if len(df) <= max_rows:
        return df

    print(f"  -> Sampling {max_rows} from {len(df)} rows (stratified by score bins)")
    np.random.seed(seed)

    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    df["_bin"] = pd.cut(df["score"], bins=bins, labels=labels, include_lowest=True)

    sampled = df.groupby("_bin", group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), max_rows // len(labels)), random_state=seed)
    )

    # 부족하면 남은 만큼 추가 샘플링
    remaining = max_rows - len(sampled)
    if remaining > 0:
        not_sampled = df.drop(sampled.index)
        extra = not_sampled.sample(n=min(remaining, len(not_sampled)), random_state=seed)
        sampled = pd.concat([sampled, extra])

    sampled = sampled.drop(columns=["_bin"])
    return sampled.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="COMET Reference-Free 학습 데이터 전처리"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="원본 학습 데이터 디렉토리 (train_data/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/en-ko-qe",
        help="변환된 데이터 출력 디렉토리",
    )
    parser.add_argument(
        "--max_train_rows",
        type=int,
        default=0,
        help="학습 데이터 최대 행 수 (0=전체 사용). "
             "대용량 데이터에서 빠른 실험을 위해 제한할 수 있습니다.",
    )
    parser.add_argument(
        "--include_pairwise",
        action="store_true",
        help="Pairwise 데이터를 pointwise로 변환하여 포함",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ========================================
    # 1. Pointwise 데이터 로드 및 변환
    # ========================================
    pointwise_train_path = os.path.join(
        args.input_dir, "en-ko-qe-patent-balanced_train.csv"
    )
    pointwise_val_path = os.path.join(
        args.input_dir, "en-ko-qe-patent-balanced_val.csv"
    )

    if not os.path.exists(pointwise_train_path):
        print(f"[ERROR] File not found: {pointwise_train_path}")
        sys.exit(1)
    if not os.path.exists(pointwise_val_path):
        print(f"[ERROR] File not found: {pointwise_val_path}")
        sys.exit(1)

    train_df = load_pointwise_data(pointwise_train_path)
    val_df = load_pointwise_data(pointwise_val_path)

    # Reference-free 형식으로 변환 (src, mt, score)
    train_referenceless = prepare_referenceless(train_df)
    val_referenceless = prepare_referenceless(val_df)

    # ========================================
    # 2. Pairwise 데이터 변환 (선택사항)
    # ========================================
    if args.include_pairwise:
        pairwise_train_path = os.path.join(
            args.input_dir, "en-ko-qe-patent-balanced_pairwise_train.csv"
        )
        pairwise_val_path = os.path.join(
            args.input_dir, "en-ko-qe-patent-balanced_pairwise_val.csv"
        )

        if os.path.exists(pairwise_train_path):
            pairwise_train = load_pairwise_data(pairwise_train_path)
            pairwise_train_expanded = expand_pairwise_to_pointwise(pairwise_train)

            # Pairwise 변환 데이터 단독 저장
            pairwise_out = os.path.join(args.output_dir, "pairwise_expanded_train.csv")
            pairwise_train_expanded.to_csv(pairwise_out, index=False)
            print(f"  -> Saved: {pairwise_out}")

            # Pointwise + Pairwise 합치기
            combined = pd.concat(
                [train_referenceless, pairwise_train_expanded], ignore_index=True
            )
            combined = combined.drop_duplicates(subset=["src", "mt"], keep="first")
            print(f"  -> Combined train: {len(combined)} rows")
        else:
            print(f"[WARN] Pairwise train file not found: {pairwise_train_path}")
            combined = train_referenceless

        if os.path.exists(pairwise_val_path):
            pairwise_val = load_pairwise_data(pairwise_val_path)
            pairwise_val_expanded = expand_pairwise_to_pointwise(pairwise_val)

            combined_val = pd.concat(
                [val_referenceless, pairwise_val_expanded], ignore_index=True
            )
            combined_val = combined_val.drop_duplicates(
                subset=["src", "mt"], keep="first"
            )
        else:
            combined_val = val_referenceless
    else:
        combined = train_referenceless
        combined_val = val_referenceless

    # ========================================
    # 3. 샘플링 (선택사항)
    # ========================================
    if args.max_train_rows > 0:
        train_referenceless = create_sampled_subset(
            train_referenceless, args.max_train_rows, args.seed
        )
        combined = create_sampled_subset(combined, args.max_train_rows, args.seed)

    # ========================================
    # 4. 데이터 저장
    # ========================================
    # ReferencelessRegression용 (pointwise only)
    ref_train_out = os.path.join(args.output_dir, "referenceless_train.csv")
    ref_val_out = os.path.join(args.output_dir, "referenceless_val.csv")
    train_referenceless.to_csv(ref_train_out, index=False)
    val_referenceless.to_csv(ref_val_out, index=False)
    print(f"\n[SAVED] ReferencelessRegression data:")
    print(f"  Train: {ref_train_out} ({len(train_referenceless)} rows)")
    print(f"  Val:   {ref_val_out} ({len(val_referenceless)} rows)")

    # UnifiedMetric QE용 (동일 형식이지만 별도 파일로 관리)
    uni_train_out = os.path.join(args.output_dir, "unified_qe_train.csv")
    uni_val_out = os.path.join(args.output_dir, "unified_qe_val.csv")
    combined.to_csv(uni_train_out, index=False)
    combined_val.to_csv(uni_val_out, index=False)
    print(f"\n[SAVED] UnifiedMetric QE data:")
    print(f"  Train: {uni_train_out} ({len(combined)} rows)")
    print(f"  Val:   {uni_val_out} ({len(combined_val)} rows)")

    # ========================================
    # 5. 데이터 통계 출력
    # ========================================
    analyze_data(train_referenceless, "Referenceless Train")
    analyze_data(val_referenceless, "Referenceless Val")
    if args.include_pairwise:
        analyze_data(combined, "Combined Train (pointwise + pairwise)")

    # ========================================
    # 6. 빠른 테스트용 소규모 데이터셋 생성
    # ========================================
    mini_train = create_sampled_subset(train_referenceless, 1000, args.seed)
    mini_val = create_sampled_subset(val_referenceless, 200, args.seed)

    mini_train_out = os.path.join(args.output_dir, "mini_train.csv")
    mini_val_out = os.path.join(args.output_dir, "mini_val.csv")
    mini_train.to_csv(mini_train_out, index=False)
    mini_val.to_csv(mini_val_out, index=False)
    print(f"\n[SAVED] Mini dataset (for quick testing):")
    print(f"  Train: {mini_train_out} ({len(mini_train)} rows)")
    print(f"  Val:   {mini_val_out} ({len(mini_val)} rows)")

    print("\n[DONE] Data preparation complete!")


if __name__ == "__main__":
    main()
