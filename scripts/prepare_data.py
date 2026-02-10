#!/usr/bin/env python3
"""
COMET Reference-Free 학습을 위한 데이터 전처리 스크립트
=====================================================

원본 학습 데이터를 COMET 학습용 (src, mt, score) 형식으로 변환합니다.
모든 접근법(ReferencelessRegression, UnifiedMetric, COMETKiwi 등)에서
동일한 출력 파일을 사용할 수 있습니다.

사용법:
    # 기본 변환 (pointwise 데이터만)
    python scripts/prepare_data.py \
        --input_dir /path/to/train_data \
        --output_dir data/en-ko-qe

    # Pairwise 데이터도 합쳐서 포함
    python scripts/prepare_data.py \
        --input_dir /path/to/train_data \
        --output_dir data/en-ko-qe \
        --include_pairwise

    # 빠른 실험용 (100만 행으로 제한)
    python scripts/prepare_data.py \
        --input_dir /path/to/train_data \
        --output_dir data/en-ko-qe \
        --max_train_rows 1000000

필수 입력 파일:
    - en-ko-qe-patent-balanced_train.csv       (pointwise 학습 데이터)
    - en-ko-qe-patent-balanced_val.csv         (pointwise 검증 데이터)
    - en-ko-qe-patent-balanced_pairwise_train.csv  (pairwise, --include_pairwise 시)
    - en-ko-qe-patent-balanced_pairwise_val.csv    (pairwise, --include_pairwise 시)

출력 파일:
    - train.csv           # 학습 데이터 (src, mt, score)
    - val.csv             # 검증 데이터 (src, mt, score)
    - mini_train.csv      # 파이프라인 테스트용 (1000 rows)
    - mini_val.csv        # 파이프라인 테스트용 (200 rows)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


def load_csv(filepath: str) -> pd.DataFrame:
    """CSV 파일 로드"""
    print(f"[INFO] Loading: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  -> {len(df):,} rows, columns: {list(df.columns)}")
    return df


def extract_src_mt_score(df: pd.DataFrame) -> pd.DataFrame:
    """(src, mt, score) 컬럼만 추출하고 정제"""
    required_cols = ["src", "mt", "score"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    result = df[required_cols].copy()
    result["src"] = result["src"].astype(str)
    result["mt"] = result["mt"].astype(str)
    result["score"] = result["score"].astype(float)

    result = result.dropna(subset=required_cols)
    result = result[result["src"].str.strip() != ""]
    result = result[result["mt"].str.strip() != ""]

    print(f"  -> Cleaned: {len(result):,} rows")
    return result


def expand_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise → Pointwise 변환
    (src, mt_good, mt_bad, score_good, score_bad) → (src, mt, score) x 2
    """
    good = df[["src", "mt_good", "score_good"]].rename(
        columns={"mt_good": "mt", "score_good": "score"}
    )
    bad = df[["src", "mt_bad", "score_bad"]].rename(
        columns={"mt_bad": "mt", "score_bad": "score"}
    )

    result = pd.concat([good, bad], ignore_index=True)
    result["src"] = result["src"].astype(str)
    result["mt"] = result["mt"].astype(str)
    result["score"] = result["score"].astype(float)

    result = result.dropna(subset=["src", "mt", "score"])
    result = result[result["src"].str.strip() != ""]
    result = result[result["mt"].str.strip() != ""]

    before = len(result)
    result = result.drop_duplicates(subset=["src", "mt"], keep="first")
    print(f"  -> Pairwise expanded: {before:,} -> {len(result):,} rows (after dedup)")
    return result


def stratified_sample(df: pd.DataFrame, max_rows: int, seed: int = 42) -> pd.DataFrame:
    """점수 구간별 균등 샘플링"""
    if len(df) <= max_rows:
        return df

    print(f"  -> Sampling {max_rows:,} from {len(df):,} rows (stratified)")
    np.random.seed(seed)

    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    df = df.copy()
    df["_bin"] = pd.cut(df["score"], bins=bins, labels=labels, include_lowest=True)

    per_bin = max_rows // len(labels)
    sampled = df.groupby("_bin", group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), per_bin), random_state=seed)
    )

    remaining = max_rows - len(sampled)
    if remaining > 0:
        not_sampled = df.drop(sampled.index)
        extra = not_sampled.sample(n=min(remaining, len(not_sampled)), random_state=seed)
        sampled = pd.concat([sampled, extra])

    return sampled.drop(columns=["_bin"]).reset_index(drop=True)


def print_stats(df: pd.DataFrame, name: str) -> None:
    """데이터 통계 출력"""
    scores = df["score"]
    print(f"\n{'='*60}")
    print(f"  {name} ({len(df):,} rows)")
    print(f"{'='*60}")
    print(f"  mean={scores.mean():.4f}  std={scores.std():.4f}  "
          f"median={scores.median():.4f}  min={scores.min():.4f}  max={scores.max():.4f}")

    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    binned = pd.cut(scores, bins=bins, labels=labels, include_lowest=True)
    for label in labels:
        count = (binned == label).sum()
        pct = count / len(df) * 100
        print(f"    {label}: {count:>10,} ({pct:>5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="COMET 학습 데이터 전처리")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="원본 데이터 디렉토리")
    parser.add_argument("--output_dir", type=str, default="data/en-ko-qe",
                        help="출력 디렉토리")
    parser.add_argument("--max_train_rows", type=int, default=0,
                        help="학습 데이터 최대 행 수 (0=전체)")
    parser.add_argument("--include_pairwise", action="store_true",
                        help="Pairwise 데이터를 변환하여 합침")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ========================================
    # 1. Pointwise 데이터 로드
    # ========================================
    train_path = os.path.join(args.input_dir, "en-ko-qe-patent-balanced_train.csv")
    val_path = os.path.join(args.input_dir, "en-ko-qe-patent-balanced_val.csv")

    for path in [train_path, val_path]:
        if not os.path.exists(path):
            print(f"[ERROR] File not found: {path}")
            sys.exit(1)

    train_df = extract_src_mt_score(load_csv(train_path))
    val_df = extract_src_mt_score(load_csv(val_path))

    # ========================================
    # 2. Pairwise 합치기 (선택)
    # ========================================
    if args.include_pairwise:
        pw_train_path = os.path.join(
            args.input_dir, "en-ko-qe-patent-balanced_pairwise_train.csv"
        )
        pw_val_path = os.path.join(
            args.input_dir, "en-ko-qe-patent-balanced_pairwise_val.csv"
        )

        if os.path.exists(pw_train_path):
            pw_train = expand_pairwise(load_csv(pw_train_path))
            train_df = pd.concat([train_df, pw_train], ignore_index=True)
            train_df = train_df.drop_duplicates(subset=["src", "mt"], keep="first")
            print(f"  -> Combined train: {len(train_df):,} rows")
        else:
            print(f"[WARN] Not found: {pw_train_path}")

        if os.path.exists(pw_val_path):
            pw_val = expand_pairwise(load_csv(pw_val_path))
            val_df = pd.concat([val_df, pw_val], ignore_index=True)
            val_df = val_df.drop_duplicates(subset=["src", "mt"], keep="first")
            print(f"  -> Combined val: {len(val_df):,} rows")
        else:
            print(f"[WARN] Not found: {pw_val_path}")

    # ========================================
    # 3. 샘플링 (선택)
    # ========================================
    if args.max_train_rows > 0:
        train_df = stratified_sample(train_df, args.max_train_rows, args.seed)

    # ========================================
    # 4. 저장
    # ========================================
    train_out = os.path.join(args.output_dir, "train.csv")
    val_out = os.path.join(args.output_dir, "val.csv")
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    print(f"\n[SAVED] train: {train_out} ({len(train_df):,} rows)")
    print(f"[SAVED] val:   {val_out} ({len(val_df):,} rows)")

    # Mini dataset
    mini_train = stratified_sample(train_df, 1000, args.seed)
    mini_val = stratified_sample(val_df, 200, args.seed)
    mini_train.to_csv(os.path.join(args.output_dir, "mini_train.csv"), index=False)
    mini_val.to_csv(os.path.join(args.output_dir, "mini_val.csv"), index=False)
    print(f"[SAVED] mini_train: {len(mini_train):,} rows")
    print(f"[SAVED] mini_val:   {len(mini_val):,} rows")

    # ========================================
    # 5. 통계 출력
    # ========================================
    print_stats(train_df, "Train")
    print_stats(val_df, "Val")

    print("\n[DONE] Data preparation complete!")


if __name__ == "__main__":
    main()
