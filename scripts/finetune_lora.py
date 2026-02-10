#!/usr/bin/env python3
"""
COMET COMETKiwi LoRA Fine-tuning 스크립트
==========================================

HuggingFace PEFT (LoRA)를 사용하여 COMETKiwi의 인코더에
저랭크 어댑터만 학습합니다.

Full fine-tuning 대비 장점:
  - 학습 파라미터 수 ~1% 이하 → 과적합 방지
  - VRAM 절약 (V100 32GB에서도 실행 가능)
  - COMETKiwi의 사전학습 지식 보존 (catastrophic forgetting 방지)
  - 소규모 데이터에서도 안정적인 학습

참고 논문:
  - MTQE.en-he (arXiv:2602.06546): Full FT는 COMETKiwi를 오히려 악화시키고,
    LoRA/BitFit이 +2~3pp 개선 달성
  - ComeTH (wasanx/ComeTH): COMETKiwi fine-tune으로 EN-TH에서 +4.9% 향상

사용법:
    # LoRA Fine-tuning
    python scripts/finetune_lora.py \
        --base_model Unbabel/wmt22-cometkiwi-da \
        --train_data data/en-ko-qe/train.csv \
        --val_data data/en-ko-qe/val.csv \
        --output_dir outputs/cometkiwi-lora-en-ko \
        --lora_rank 16 \
        --epochs 3

    # Head-only Fine-tuning (FTHead)
    python scripts/finetune_lora.py \
        --base_model Unbabel/wmt22-cometkiwi-da \
        --train_data data/en-ko-qe/train.csv \
        --val_data data/en-ko-qe/val.csv \
        --output_dir outputs/cometkiwi-fthead-en-ko \
        --mode fthead \
        --epochs 5

    # BitFit (bias terms only)
    python scripts/finetune_lora.py \
        --base_model Unbabel/wmt22-cometkiwi-da \
        --train_data data/en-ko-qe/train.csv \
        --val_data data/en-ko-qe/val.csv \
        --output_dir outputs/cometkiwi-bitfit-en-ko \
        --mode bitfit \
        --epochs 5

설치 필요:
    pip install peft>=0.6.0
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class QEDataset(Dataset):
    """Reference-free QE 데이터셋"""

    def __init__(self, csv_path: str, max_rows: int = 0):
        df = pd.read_csv(csv_path)
        df = df[["src", "mt", "score"]].dropna()
        df["src"] = df["src"].astype(str)
        df["mt"] = df["mt"].astype(str)
        df["score"] = df["score"].astype(float)

        if max_rows > 0 and len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

        self.data = df.to_dict("records")
        logger.info(f"Loaded {len(self.data)} samples from {csv_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def freeze_encoder_keep_bias(model):
    """BitFit: 인코더의 bias 파라미터만 학습 가능하게 설정"""
    trainable_count = 0
    frozen_count = 0

    for name, param in model.named_parameters():
        if "estimator" in name:
            # Head는 항상 학습
            param.requires_grad = True
            trainable_count += param.numel()
        elif "bias" in name:
            # Bias만 학습
            param.requires_grad = True
            trainable_count += param.numel()
        elif "layerwise_attention" in name:
            # Layer attention도 학습
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    total = trainable_count + frozen_count
    logger.info(
        f"BitFit: {trainable_count:,} trainable ({trainable_count/total*100:.2f}%) / "
        f"{frozen_count:,} frozen ({frozen_count/total*100:.2f}%)"
    )


def freeze_encoder_head_only(model):
    """FTHead: Head(estimator)와 layerwise_attention만 학습"""
    trainable_count = 0
    frozen_count = 0

    for name, param in model.named_parameters():
        if "estimator" in name or "layerwise_attention" in name:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    total = trainable_count + frozen_count
    logger.info(
        f"FTHead: {trainable_count:,} trainable ({trainable_count/total*100:.2f}%) / "
        f"{frozen_count:,} frozen ({frozen_count/total*100:.2f}%)"
    )


def apply_lora(model, rank: int = 16, alpha: int = 32, target_modules: list = None):
    """LoRA 어댑터를 인코더에 적용"""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        logger.error("peft 패키지가 필요합니다: pip install peft>=0.6.0")
        sys.exit(1)

    if target_modules is None:
        # XLM-RoBERTa / InfoXLM의 attention + FFN 레이어
        target_modules = [
            "query", "key", "value",  # self-attention
            "dense",  # output projection + FFN
        ]

    # LoRA는 encoder.model 에 적용
    encoder_model = model.encoder.model

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type=None,  # 커스텀 task
    )

    # PEFT 적용
    model.encoder.model = get_peft_model(encoder_model, lora_config)

    # Head와 layerwise_attention은 학습 가능으로 유지
    for name, param in model.named_parameters():
        if "estimator" in name or "layerwise_attention" in name:
            param.requires_grad = True

    # 통계 출력
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    logger.info(
        f"LoRA (rank={rank}, alpha={alpha}): "
        f"{trainable_count:,} trainable ({trainable_count/total_count*100:.2f}%) / "
        f"{total_count:,} total"
    )

    return model


def _move_inputs_to_device(batch_input, device):
    """batch_input을 device로 이동. tuple of dicts 또는 단일 dict 모두 처리."""
    if isinstance(batch_input, (tuple, list)):
        # UnifiedMetric: tuple of dicts (각 forward pass 입력)
        return tuple(
            {k: v.to(device) for k, v in inp.items()} for inp in batch_input
        )
    else:
        # ReferencelessRegression: 단일 dict
        return {k: v.to(device) for k, v in batch_input.items()}


def train_epoch(model, dataloader, optimizer, device, epoch, writer=None, global_step=0, log_interval=100):
    """1 에폭 학습"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        model_inputs, targets = batch
        model_inputs = _move_inputs_to_device(model_inputs, device)
        targets_score = targets["score"].to(device)

        optimizer.zero_grad()

        # UnifiedMetric은 tuple of dicts (여러 forward pass),
        # ReferencelessRegression은 단일 dict
        if isinstance(model_inputs, (tuple, list)):
            # UnifiedMetric 방식: 각 input sequence에 대해 forward pass
            loss = torch.tensor(0.0, device=device)
            for input_seq in model_inputs:
                prediction = model(**input_seq)
                loss = loss + torch.nn.MSELoss()(prediction.score, targets_score)
        else:
            prediction = model(**model_inputs)
            loss = torch.nn.MSELoss()(prediction.score, targets_score)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        # TensorBoard: step별 loss
        if writer is not None:
            writer.add_scalar("train/step_loss", loss.item(), global_step)

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches
            logger.info(f"  Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] loss={avg_loss:.6f}")

    return total_loss / max(num_batches, 1), global_step


def evaluate(model, dataloader, device):
    """검증 데이터 평가"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            model_inputs, targets = batch
            model_inputs = _move_inputs_to_device(model_inputs, device)

            # UnifiedMetric: 여러 forward pass의 마지막 prediction 사용
            if isinstance(model_inputs, (tuple, list)):
                prediction = None
                for input_seq in model_inputs:
                    prediction = model(**input_seq)
            else:
                prediction = model(**model_inputs)

            all_preds.extend(prediction.score.cpu().tolist())
            all_targets.extend(targets["score"].tolist())

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    pearson_r, _ = stats.pearsonr(targets, preds)
    spearman_r, _ = stats.spearmanr(targets, preds)
    kendall_tau, _ = stats.kendalltau(targets, preds)
    mse = np.mean((targets - preds) ** 2)

    return {
        "pearson": pearson_r,
        "spearman": spearman_r,
        "kendall": kendall_tau,
        "mse": mse,
    }


def main():
    parser = argparse.ArgumentParser(description="COMETKiwi LoRA/BitFit/FTHead Fine-tuning")
    parser.add_argument("--base_model", type=str, default="Unbabel/wmt22-cometkiwi-da",
                        help="Base COMET model name or checkpoint path")
    parser.add_argument("--train_data", type=str, required=True, help="Training CSV path")
    parser.add_argument("--val_data", type=str, required=True, help="Validation CSV path")
    parser.add_argument("--output_dir", type=str, default="outputs/cometkiwi-lora",
                        help="Output directory for checkpoints")
    parser.add_argument("--mode", type=str, default="lora",
                        choices=["lora", "bitfit", "fthead"],
                        help="Fine-tuning mode")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate (PEFT 방법은 더 높은 LR 사용 가능)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max_train_rows", type=int, default=0,
                        help="Max training rows (0=all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ========================================
    # 1. 모델 로드
    # ========================================
    logger.info(f"Loading base model: {args.base_model}")

    if os.path.exists(args.base_model):
        # 로컬 체크포인트
        from comet import load_from_checkpoint
        model = load_from_checkpoint(args.base_model)
    else:
        # HuggingFace 모델
        from comet import download_model, load_from_checkpoint
        model_path = download_model(args.base_model)
        model = load_from_checkpoint(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ========================================
    # 2. Fine-tuning 방법 적용
    # ========================================
    if args.mode == "lora":
        logger.info(f"Applying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})")
        model = apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
    elif args.mode == "bitfit":
        logger.info("Applying BitFit (bias-only fine-tuning)")
        freeze_encoder_keep_bias(model)
    elif args.mode == "fthead":
        logger.info("Applying FTHead (head-only fine-tuning)")
        freeze_encoder_head_only(model)

    model = model.to(device)

    # ========================================
    # 3. 데이터 로드
    # ========================================
    train_dataset = QEDataset(args.train_data, max_rows=args.max_train_rows)
    val_dataset = QEDataset(args.val_data, max_rows=10000)  # 검증은 최대 10k

    def collate_fn(batch):
        return model.prepare_sample(batch, stage="fit")

    def collate_fn_val(batch):
        return model.prepare_sample(batch, stage="validate")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset),
        collate_fn=collate_fn, num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        collate_fn=collate_fn_val, num_workers=2,
    )

    # ========================================
    # 4. 옵티마이저
    # ========================================
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=0.01)

    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Epochs: {args.epochs}, LR: {args.learning_rate}")

    # ========================================
    # 5. TensorBoard 초기화
    # ========================================
    tb_log_dir = os.path.join(args.output_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tb_log_dir)
    logger.info(f"TensorBoard log dir: {tb_log_dir}")
    logger.info(f"  -> tensorboard --logdir {tb_log_dir}")

    # 하이퍼파라미터 기록
    writer.add_text("hparams", (
        f"mode={args.mode}, base_model={args.base_model}, "
        f"lr={args.learning_rate}, batch_size={args.batch_size}, "
        f"epochs={args.epochs}, lora_rank={args.lora_rank}, lora_alpha={args.lora_alpha}, "
        f"train_samples={len(train_dataset)}, val_samples={len(val_dataset)}, "
        f"trainable_params={sum(p.numel() for p in trainable_params):,}"
    ))

    # ========================================
    # 6. 학습 루프
    # ========================================
    best_kendall = -1
    best_epoch = -1
    global_step = 0

    for epoch in range(args.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        logger.info(f"{'='*60}")

        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, device, epoch + 1,
            writer=writer, global_step=global_step,
        )
        logger.info(f"  Train loss: {train_loss:.6f}")

        metrics = evaluate(model, val_loader, device)
        logger.info(f"  Val Pearson:  {metrics['pearson']:.4f}")
        logger.info(f"  Val Spearman: {metrics['spearman']:.4f}")
        logger.info(f"  Val Kendall:  {metrics['kendall']:.4f}")
        logger.info(f"  Val MSE:      {metrics['mse']:.6f}")

        # TensorBoard: 에폭별 metrics 기록
        writer.add_scalar("train/epoch_loss", train_loss, epoch + 1)
        writer.add_scalar("val/pearson", metrics["pearson"], epoch + 1)
        writer.add_scalar("val/spearman", metrics["spearman"], epoch + 1)
        writer.add_scalar("val/kendall", metrics["kendall"], epoch + 1)
        writer.add_scalar("val/mse", metrics["mse"], epoch + 1)

        # 체크포인트 저장
        if metrics["kendall"] > best_kendall:
            best_kendall = metrics["kendall"]
            best_epoch = epoch + 1
            save_path = os.path.join(args.output_dir, "best_model.ckpt")

            # COMET 형식으로 저장 (load_from_checkpoint으로 로드 가능)
            torch.save({
                "state_dict": model.state_dict(),
                "hyper_parameters": dict(model.hparams),
            }, save_path)
            logger.info(f"  -> Best model saved: {save_path} (kendall={best_kendall:.4f})")

        # 에폭별 체크포인트
        epoch_path = os.path.join(args.output_dir, f"epoch{epoch+1}_kendall{metrics['kendall']:.4f}.ckpt")
        torch.save({
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
        }, epoch_path)

    # TensorBoard 종료
    writer.flush()
    writer.close()
    logger.info(f"\nTensorBoard logs saved to: {tb_log_dir}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete!")
    logger.info(f"Best epoch: {best_epoch}, Best Kendall: {best_kendall:.4f}")
    logger.info(f"Best model: {os.path.join(args.output_dir, 'best_model.ckpt')}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
