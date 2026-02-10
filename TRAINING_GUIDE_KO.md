# COMET Reference-Free 모델 학습 가이드 (A-Z)

> EN-KO 특허 번역 품질 평가를 위한 Reference-Free COMET 모델 학습 완벽 가이드

## 목차

1. [개요](#1-개요)
2. [배경 지식](#2-배경-지식)
3. [환경 설정](#3-환경-설정)
4. [데이터 준비](#4-데이터-준비)
5. [학습 접근법 비교](#5-학습-접근법-비교)
6. [Step-by-Step 학습 실행](#6-step-by-step-학습-실행)
7. [모델 평가](#7-모델-평가)
8. [하이퍼파라미터 튜닝](#8-하이퍼파라미터-튜닝)
9. [문제 해결 (Troubleshooting)](#9-문제-해결-troubleshooting)
10. [참고 자료](#10-참고-자료)

---

## 1. 개요

### 1.1 목표

현재 보유한 EN-KO 특허 번역 품질 평가 데이터를 사용하여 **reference 없이** source와 MT만으로 번역 품질 점수를 예측하는 COMET 모델을 학습합니다.

### 1.2 Reference-Free란?

일반적인 번역 평가 지표(BLEU, COMET 기본 모델)는 정답 번역(reference)이 필요합니다. 하지만 실제 서비스 환경에서는 reference가 없는 경우가 많습니다. **Reference-Free (Quality Estimation, QE)** 모델은 원문(source)과 기계번역(MT)만으로 품질을 예측합니다.

```
# Reference-based (기존)
Score = Model(source, MT, reference)

# Reference-free (목표)
Score = Model(source, MT)     ← reference 불필요!
```

### 1.3 보유 데이터 현황

| 파일 | 행 수 | 용도 |
|------|-------|------|
| `en-ko-qe-patent-balanced_train.csv` | ~9,663,341 | Pointwise 학습 |
| `en-ko-qe-patent-balanced_val.csv` | ~508,621 | Pointwise 검증 |
| `en-ko-qe-patent-balanced_pairwise_train.csv` | ~1,437,196 | Pairwise 학습 |
| `en-ko-qe-patent-balanced_pairwise_val.csv` | ~1,314 | Pairwise 검증 |

**Pointwise 형식**: `src, mt, ref, score` (개별 점수)
**Pairwise 형식**: `src, mt_good, mt_bad, score_good, score_bad` (쌍 비교)

---

## 2. 배경 지식

### 2.1 COMET 아키텍처

COMET은 사전학습된 다국어 언어 모델(XLM-RoBERTa, InfoXLM 등)을 인코더로 사용하고, 그 위에 회귀 Head를 붙여 번역 품질 점수를 예측합니다.

```
┌─────────────────────────────────────────────────┐
│                  Quality Score                    │
│                    (0~1)                          │
├─────────────────────────────────────────────────┤
│            Feed-Forward Head                      │
│         (2048 → 1024 → 1)                        │
├─────────────────────────────────────────────────┤
│           Feature Construction                    │
│   [mt_emb, src_emb, mt*src, |mt-src|]            │
├──────────────────┬──────────────────────────────┤
│  MT Embedding    │    Source Embedding            │
├──────────────────┴──────────────────────────────┤
│       Pretrained Encoder (XLM-R / InfoXLM)       │
│               (frozen → unfreeze)                 │
└─────────────────────────────────────────────────┘
```

### 2.2 Reference-Free 모델 종류

COMET에는 2가지 Reference-Free 아키텍처가 있습니다:

#### (A) ReferencelessRegression (단순 구조)

```python
# comet/models/regression/referenceless.py
src_emb = encoder(source)      # 소스 인코딩
mt_emb = encoder(MT)           # MT 인코딩  (별도 인코딩!)

features = [mt_emb, src_emb, mt_emb * src_emb, |mt_emb - src_emb|]
score = feedforward(features)  # 4 * 1024 = 4096 dim 입력
```

- Source와 MT를 **별도로** 인코딩
- 4가지 특징 벡터 조합 (연결, 곱, 절대 차이)
- 구조가 단순하고 이해하기 쉬움
- 과거 모델: `wmt20-comet-qe-da`, `wmt21-comet-qe-da`

#### (B) UnifiedMetric QE 모드 (COMETKiwi 구조, 추천)

```python
# comet/models/multitask/unified_metric.py
combined_input = "[CLS] MT [SEP] Source [SEP]"
encoder_out = encoder(combined_input)  # 하나의 시퀀스로 인코딩!

cls_embedding = encoder_out[:, 0, :]   # CLS 토큰
score = feedforward(cls_embedding)     # 1024 dim 입력
```

- Source와 MT를 **하나의 시퀀스로 연결**하여 인코딩
- Cross-attention 효과 (토큰 간 상호작용)
- CLS 토큰으로 문장 표현
- 현재 SOTA: `wmt22-cometkiwi-da`, `wmt23-cometkiwi-da-xl`

### 2.3 어떤 구조를 선택해야 할까?

| 기준 | ReferencelessRegression | UnifiedMetric QE |
|------|------------------------|-------------------|
| 성능 | 보통 | **더 높음** (SOTA) |
| 학습 난이도 | 쉬움 | 약간 복잡 |
| 메모리 사용 | 더 높음 (2회 인코딩) | 더 효율적 (1회 인코딩) |
| Fine-tuning 호환 | wmt20/21 QE 체크포인트 | **COMETKiwi 체크포인트** |
| 추천도 | 실험/비교용 | **프로덕션 추천** |

**결론: UnifiedMetric QE 모드 + COMETKiwi fine-tuning을 가장 추천합니다.**

### 2.4 학습 전략: From Scratch vs Fine-tuning

#### From Scratch
- 사전학습 인코더(XLM-R)에 새로운 Head를 학습
- 장점: 도메인에 처음부터 맞출 수 있음
- 단점: 많은 데이터와 긴 학습 시간 필요

#### Full Fine-tuning
- COMETKiwi 전체 파라미터를 학습
- 장점: 도메인에 깊이 적응 가능, 대용량 데이터에 적합
- 단점: **소규모 데이터에서 과적합/분포 붕괴 위험** (MTQE 논문에서 확인됨)

#### Parameter-Efficient Fine-tuning (PEFT)
- LoRA, BitFit, FTHead 등으로 일부 파라미터만 학습
- 장점: 과적합 방지, VRAM 절약, 안정적 학습
- 단점: 표현력이 제한될 수 있음

### 2.5 실제 사례 분석: COMETKiwi Fine-tuning 성공/실패 사례

#### (A) ComeTH (wasanx/ComeTH) - EN-TH Fine-tuning 성공 사례

[ComeTH](https://huggingface.co/wasanx/ComeTH)는 COMETKiwi(`wmt22-cometkiwi-da`)를
영어-태국어 번역 품질 평가에 맞게 fine-tuning한 모델입니다.

| 항목 | 내용 |
|------|------|
| **베이스 모델** | `Unbabel/wmt22-cometkiwi-da` (UnifiedMetric) |
| **인코더** | `microsoft/infoxlm-large` |
| **학습 데이터** | Human MQM 어노테이션 + Claude 3.5 Sonnet 증강 데이터 |
| **에폭** | 5 |
| **Gradient Accumulation** | 8 |
| **결과** | Baseline Spearman 0.4570 → **+4.9% 향상** |

**핵심 인사이트:**
- COMETKiwi를 특정 언어쌍에 fine-tuning하면 유의미한 성능 향상이 가능함을 입증
- **LLM 증강 데이터** (Claude 3.5 Sonnet이 생성한 품질 판단)가 성능을 추가로 향상
- LLM 증강 모델이 직접 LLM 평가보다도 높은 상관계수 달성 (비용 효율적)
- 우리의 접근법 3과 **동일한 전략** (COMETKiwi + UnifiedMetric + fine-tuning)

#### (B) MTQE.en-he (arXiv:2602.06546) - Full Fine-tuning 위험성 발견

2026년 2월 발표된 이 논문은 **소규모 데이터에서 COMETKiwi full fine-tuning의 위험성**을
실증적으로 보여줍니다. 300개 EN-HE 샘플에서 실험:

| Fine-tuning 방법 | 학습 파라미터 비율 | 결과 |
|------------------|-------------------|------|
| **Full Fine-tuning** | 100% | **성능 악화** (2~3pp 하락, 분포 붕괴) |
| **LoRA** | ~1% | +2~3pp 향상 (Pearson/Spearman) |
| **BitFit** (bias만) | ~0.2% | +2~3pp 향상 |
| **FTHead** (head만) | ~2% | +2~3pp 향상 (LoRA보다 약간 낮음) |

> **경고**: 소규모 데이터에서 Full fine-tuning은 COMETKiwi에서 **overfitting과
> score distribution collapse**를 유발합니다. 모델이 특정 점수 범위로만 예측하게 됩니다.

**단, 이 논문의 조건과 우리 상황은 매우 다릅니다:**

| 조건 | MTQE.en-he | **우리 프로젝트** |
|------|-----------|-----------------|
| 학습 데이터 크기 | **300 샘플** | **~960만 샘플** (32,000배) |
| 검증 데이터 | 100 샘플 | ~50만 샘플 |
| 도메인 | 일반 WMT | 특허 (특정 도메인) |
| 언어쌍 | EN-HE | EN-KO |

**우리의 대용량 데이터(960만 행)에서는 full fine-tuning이 성공할 가능성이 높습니다.**
소규모 데이터에서의 과적합/분포 붕괴 문제는 대규모 데이터에서는 자연스럽게 완화됩니다.
하지만 **안전한 대안**으로 LoRA/BitFit/FTHead도 함께 제공합니다.

#### (C) 기타 참고 사례

| 프로젝트 | 전략 | 핵심 교훈 |
|----------|------|-----------|
| **AfriCOMET** (아프리카 13개 언어) | 언어 특화 인코더(AfroXLM-R) 사용 | 언어 계열 전용 인코더가 효과적 |
| **IndicCOMET** (인도 5개 언어, 7000 MQM) | COMET-MQM 체크포인트에서 시작 | 더 강한 체크포인트 = 더 좋은 결과 |
| **Cometoid** (WMT 2023, 지식 증류) | 교사 모델로 합성 점수 생성 | Reference-free가 reference-based 능가 가능 |
| **도메인 적응 연구** (ACL 2024) | 도메인 간 전이 분석 | fine-tuned 모델은 학습 도메인 밖에서 성능 저하 가능 |

### 2.6 우리 데이터에 대한 전략적 판단

**결론: 접근법 3 (COMETKiwi Full Fine-tuning)은 성공 가능성이 높습니다.**

근거:
1. **ComeTH가 동일 전략으로 성공** → COMETKiwi fine-tuning 자체는 검증된 방법
2. **960만 행의 대규모 데이터** → MTQE 논문의 과적합 문제(300샘플)와는 상황이 완전히 다름
3. **특허 도메인 특화** → 도메인 적응이 성능 향상에 기여할 것

**추천 실행 순서 (수정됨):**
1. 먼저 **접근법 5 (FTHead)** → 가장 안전하고 빠르게 baseline 확보
2. 그 다음 **접근법 3 (Full Fine-tuning)** → 대용량 데이터의 장점 활용
3. 접근법 3에서 과적합 징후 시 **접근법 6 (LoRA)** → 중간 지점
4. 성능 비교 후 최적 방법 선택

---

## 3. 환경 설정

### 3.1 하드웨어 요구사항

| 구성 | 최소 | 권장 |
|------|------|------|
| GPU | 1x V100 (32GB) | 1-2x A100 (80GB) |
| RAM | 32GB | 64GB+ |
| Disk | 50GB | 100GB+ |
| CUDA | 11.7+ | 12.0+ |

> **참고**: 데이터가 ~960만 행으로 매우 크기 때문에 RAM이 충분해야 합니다.
> 메모리가 부족하면 `--max_train_rows`로 데이터를 줄여 실험할 수 있습니다.

### 3.2 소프트웨어 설치

```bash
# 1. 프로젝트 디렉토리 이동
cd /path/to/COMET

# 2. Python 가상환경 생성 (권장)
python -m venv .venv
source .venv/bin/activate

# 3. Poetry를 사용한 의존성 설치 (pyproject.toml 기반)
pip install poetry
poetry install

# 또는 pip으로 직접 설치
pip install unbabel-comet

# 4. 추가 패키지 (평가 스크립트용)
pip install scipy scikit-learn

# 5. LoRA Fine-tuning용 (접근법 5, 6에서 사용)
pip install peft>=0.6.0

# 5. 설치 확인
comet-score --help
comet-train --help
```

### 3.3 설치 확인

```bash
# Python에서 COMET 임포트 확인
python -c "
import comet
print(f'COMET version: {comet.__version__}')
from comet.models import ReferencelessRegression, UnifiedMetric
print('Models imported successfully')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
```

---

## 4. 데이터 준비

### 4.1 COMET이 요구하는 데이터 형식

**ReferencelessRegression** (`referenceless.py:202-214`):
```csv
src,mt,score
"source sentence in English","번역된 한국어 문장",0.75
```

**UnifiedMetric QE** (`unified_metric.py:288-306`):
```csv
src,mt,score
"source sentence in English","번역된 한국어 문장",0.75
```

> 두 모델 모두 `src, mt, score` 3개 컬럼만 필요합니다.
> `ref` 컬럼은 무시됩니다 (reference-free).

### 4.2 데이터 변환 실행

```bash
# 기본 변환 (pointwise 데이터만)
python scripts/prepare_data.py \
    --input_dir /path/to/train_data \
    --output_dir data/en-ko-qe

# Pairwise 데이터도 포함
python scripts/prepare_data.py \
    --input_dir /path/to/train_data \
    --output_dir data/en-ko-qe \
    --include_pairwise

# 빠른 실험용 (100만 행으로 제한)
python scripts/prepare_data.py \
    --input_dir /path/to/train_data \
    --output_dir data/en-ko-qe \
    --max_train_rows 1000000 \
    --include_pairwise
```

### 4.3 변환 결과 확인

```
data/en-ko-qe/
├── referenceless_train.csv       # ReferencelessRegression용 학습 (src, mt, score)
├── referenceless_val.csv         # ReferencelessRegression용 검증
├── unified_qe_train.csv          # UnifiedMetric QE용 학습
├── unified_qe_val.csv            # UnifiedMetric QE용 검증
├── pairwise_expanded_train.csv   # Pairwise→Pointwise 변환 데이터
├── mini_train.csv                # 파이프라인 테스트용 (1000 rows)
└── mini_val.csv                  # 파이프라인 테스트용 (200 rows)
```

### 4.4 데이터 포맷 상세 설명

현재 보유 데이터의 주요 컬럼:

```
# Pointwise (en-ko-qe-patent-balanced_train.csv)
lp          : 언어쌍 (en-ko)
src         : 영어 원문                    ← COMET 사용
mt          : 기계 번역 (한국어)            ← COMET 사용
ref         : 정답 번역 (한국어)            ← Reference-free에서는 미사용
score       : 품질 점수 (0~1)              ← COMET 사용
domain      : 도메인 (us_cl 등)
model_type  : MT 모델 종류 (gemma 등)
src_scores  : 소스 기반 점수
mqm_scores  : MQM 점수
```

`score` 컬럼이 0~1 범위인 것은 COMET 학습에 이상적입니다 (정규화 불필요).

### 4.5 점수 분포 분석과 리밸런싱 (매우 중요!)

> **핵심**: COMET은 학습 데이터의 점수 분포에 **매우 민감**합니다.
> "COMET is highly susceptible to the distribution of scores in the training data"
> — Falcao et al. (LREC 2024), Zouhar et al. (WMT 2024)

#### 현재 데이터의 점수 분포 문제

```
0.0-0.3:   344,930 (  3.5%)  ██▌             ← 심각하게 부족!
0.3-0.5: 1,363,461 ( 14.2%) ██████████
0.5-0.7: 2,197,215 ( 22.8%) ████████████████
0.7-1.0: 5,717,209 ( 59.4%) ████████████████████████████████████  ← 과다!
```

**문제점:**
1. **고품질 편향**: 0.7~1.0 구간이 60%를 차지 → 모델이 항상 높은 점수를 예측하는 경향
2. **저품질 식별 불가**: 0.0~0.3 구간이 3.5%뿐 → 나쁜 번역을 낮은 점수로 평가하지 못함
3. **평균 회귀**: MSE 손실은 본질적으로 학습 데이터 평균으로 수렴하는 경향
4. **예측 범위 압축**: 출력이 0.5~0.8 좁은 범위에 몰리는 "score distribution collapse"

#### 이 문제가 발생하는 이유

이것은 WMT DA 학습 데이터에서도 동일하게 발생하는 알려진 문제입니다.
Zouhar et al. (WMT 2024) "Pitfalls in Using COMET" 논문에서 실험으로 증명:

| 실험 | 결과 |
|------|------|
| 전체 데이터로 학습 | 기준선 |
| 상위 75% 점수만 사용 | 점수가 체계적으로 **높아짐** |
| 하위 75% 점수만 사용 | 점수가 체계적으로 **낮아짐** |

→ 학습 데이터의 점수 분포가 모델 출력을 직접적으로 편향시킵니다.

#### 해결 방법: 리밸런싱

```bash
# 1. 먼저 현재 분포 분석
python scripts/analyze_and_rebalance.py \
    --input data/en-ko-qe/referenceless_train.csv \
    --analyze_only

# 2-A. 소프트 리밸런싱 (권장, 균형과 데이터 보존 타협)
python scripts/analyze_and_rebalance.py \
    --input data/en-ko-qe/referenceless_train.csv \
    --output data/en-ko-qe/referenceless_train_balanced.csv \
    --strategy soft \
    --target_total 3000000 \
    --smoothing 0.5

# 2-B. 완전 균등 리밸런싱 (가장 공격적)
python scripts/analyze_and_rebalance.py \
    --input data/en-ko-qe/referenceless_train.csv \
    --output data/en-ko-qe/referenceless_train_equal.csv \
    --strategy equal
```

#### 리밸런싱 전략 비교

| 전략 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **soft (권장)** | 과소 구간 오버샘플 + 과다 구간 언더샘플 | 균형과 데이터 보존 타협 | 완벽히 균등하지 않음 |
| equal | 모든 구간 동일 샘플 수 | 완벽히 균등 | 데이터 손실 큼 (최소 구간에 맞춤) |
| weighted | sample_weight 컬럼 추가 | 원본 보존 | 커스텀 loss 코드 필요 |

#### 소프트 리밸런싱 적용 시 예상 변화

`--smoothing 0.5` (제곱근 역빈도) 적용 시:

```
            원본                →        리밸런싱 후
0.0-0.3:   3.5% (34만)         →        ~12% (36만, 오버샘플)
0.3-0.5:  14.2% (136만)        →        ~18% (54만, 유사)
0.5-0.7:  22.8% (220만)        →        ~25% (75만, 언더샘플)
0.7-1.0:  59.4% (572만)        →        ~45% (135만, 언더샘플)
```

→ 저품질 구간이 보강되어 모델이 좋고 나쁜 번역을 더 잘 구별할 수 있게 됩니다.

#### 학습 시 적용

리밸런싱된 데이터를 사용하려면 config에서 train_data 경로를 변경:

```yaml
train_data:
  - data/en-ko-qe/referenceless_train_balanced.csv   # 리밸런싱 데이터
```

#### 주의사항

1. **검증 데이터는 리밸런싱하지 마세요** — 실제 분포를 반영해야 정확한 평가 가능
2. **오버샘플링된 데이터에서 과적합 주의** — 같은 샘플이 반복되므로 에폭 수를 줄이기
3. **단계적 실험** — 먼저 원본으로 학습 → 리밸런싱으로 학습 → 결과 비교

---

## 5. 학습 접근법 비교

총 **6가지 접근법**을 준비했습니다. 상황에 맞게 선택하세요.

### 접근법 비교표

| # | 접근법 | 아키텍처 | 시작점 | 학습 파라미터 | 난이도 | 위험도 |
|---|--------|---------|--------|-------------|--------|--------|
| 1 | ReferencelessRegression Scratch | ReferencelessRegression | XLM-R | 100% | ★☆☆ | 낮음 |
| 2 | UnifiedMetric QE Scratch | UnifiedMetric | InfoXLM | 100% | ★★☆ | 낮음 |
| **3** | **COMETKiwi Full Fine-tuning** | **UnifiedMetric** | **COMETKiwi** | **100%** | **★★☆** | **중간** |
| 4 | ReferencelessRegression Fine-tuning | ReferencelessRegression | wmt21-qe | 100% | ★★☆ | 중간 |
| **5** | **COMETKiwi FTHead (Head만)** | **UnifiedMetric** | **COMETKiwi** | **~2%** | **★☆☆** | **낮음** |
| **6** | **COMETKiwi LoRA** | **UnifiedMetric** | **COMETKiwi** | **~1%** | **★★☆** | **낮음** |

### ComeTH와 우리 접근법 비교

| 항목 | ComeTH (EN-TH) | **우리 프로젝트 (EN-KO)** |
|------|----------------|--------------------------|
| 베이스 모델 | wmt22-cometkiwi-da | wmt22-cometkiwi-da |
| 방법 | Full Fine-tuning | Full FT + LoRA + FTHead (비교) |
| 데이터 규모 | 소~중규모 (MQM) | **~960만 행** (초대규모) |
| 데이터 증강 | Claude 3.5 Sonnet | MQM 점수 기반 |
| 에폭 | 5 | 3~5 |
| Grad Accum | 8 | 4~8 |
| 성과 | Spearman +4.9% | 측정 예정 |

### 추천 실행 순서

```
STEP 0: 미니 테스트 (파이프라인 확인)
  │
  ▼
STEP 5: 접근법 5 - FTHead (가장 안전, 빠른 baseline)
  │
  ├── 성능 양호 → STEP 3으로
  └── 성능 부족 → STEP 6으로
  │
STEP 3: 접근법 3 - Full Fine-tuning (대용량 데이터 활용)
  │
  ├── 과적합 없음 → 최적 모델 후보
  └── 과적합 발생 → STEP 6으로
  │
STEP 6: 접근법 6 - LoRA (과적합 방지 + 성능 균형)
  │
  ▼
최적 모델 선택 (val_kendall 기준)
```

---

## 6. Step-by-Step 학습 실행

### STEP 0: 파이프라인 테스트 (필수!)

실제 학습 전에 작은 데이터로 모든 것이 정상 동작하는지 확인합니다.

```bash
# 미니 데이터가 준비되었는지 확인
ls data/en-ko-qe/mini_train.csv data/en-ko-qe/mini_val.csv

# 미니 테스트 실행
bash scripts/run_training.sh mini

# 또는 직접 실행
comet-train --cfg configs/models/en-ko-qe/approach_mini_test.yaml --seed_everything 12
```

정상 동작 시 아래와 유사한 출력이 나옵니다:
```
TRAINER ARGUMENTS:
{...}
MODEL ARGUMENTS:
{...}
Epoch 0: 100%|██████████| 20/20 [00:XX<00:00, X.XX it/s, train_loss=X.XXX]
```

### STEP 1: 접근법 1 - ReferencelessRegression From Scratch

```bash
# 학습 실행
bash scripts/run_training.sh scratch1 --seed 12

# 또는 직접 실행
comet-train \
    --cfg configs/models/en-ko-qe/approach1_referenceless_scratch.yaml \
    --seed_everything 12
```

### STEP 2: 접근법 2 - UnifiedMetric QE From Scratch

```bash
bash scripts/run_training.sh scratch2 --seed 12

# 또는 직접 실행
comet-train \
    --cfg configs/models/en-ko-qe/approach2_unified_qe_scratch.yaml \
    --seed_everything 12
```

### STEP 3: 접근법 3 - COMETKiwi Fine-tuning (추천)

이 접근법이 가장 높은 성능을 낼 가능성이 높습니다.

```bash
# 3-1. COMETKiwi 체크포인트 다운로드
python scripts/download_checkpoint.py --model Unbabel/wmt22-cometkiwi-da

# 다운로드된 체크포인트 경로가 출력됩니다. 예:
#   Checkpoint file: /root/.cache/huggingface/hub/.../checkpoints/model.ckpt

# 3-2. 체크포인트 경로 확인
# (출력된 경로를 CHECKPOINT_PATH에 저장)
CHECKPOINT_PATH="위에서_출력된_경로"

# 3-3. Fine-tuning 실행
bash scripts/run_training.sh finetune --checkpoint $CHECKPOINT_PATH

# 또는 직접 실행
comet-train \
    --cfg configs/models/en-ko-qe/approach3_finetune_cometkiwi.yaml \
    --load_from_checkpoint $CHECKPOINT_PATH \
    --seed_everything 12
```

### STEP 4: 접근법 4 - ReferencelessRegression Fine-tuning

```bash
# 4-1. 기존 QE 체크포인트 다운로드
python scripts/download_checkpoint.py --model wmt21-comet-qe-da --legacy

# 4-2. Fine-tuning 실행
CHECKPOINT_PATH="다운로드된_체크포인트_경로"
bash scripts/run_training.sh ft-qe --checkpoint $CHECKPOINT_PATH
```

### STEP 5: 접근법 5 - COMETKiwi FTHead (Head만 학습, 가장 안전)

인코더를 완전히 동결하고 estimator head와 layerwise_attention만 학습합니다.
MTQE 논문에서 검증된 안전한 방법으로, 가장 먼저 시도하기를 추천합니다.

```bash
# 5-1. COMETKiwi 체크포인트 다운로드 (STEP 3에서 이미 했다면 생략)
python scripts/download_checkpoint.py --model Unbabel/wmt22-cometkiwi-da
CHECKPOINT_PATH="다운로드된_경로"

# 5-2. FTHead Fine-tuning 실행
python scripts/finetune_lora.py \
    --base_model $CHECKPOINT_PATH \
    --train_data data/en-ko-qe/unified_qe_train.csv \
    --val_data data/en-ko-qe/unified_qe_val.csv \
    --output_dir outputs/cometkiwi-fthead-en-ko \
    --mode fthead \
    --learning_rate 1e-4 \
    --batch_size 16 \
    --epochs 5

# 대용량 데이터 시 샘플링하여 빠른 실험
python scripts/finetune_lora.py \
    --base_model $CHECKPOINT_PATH \
    --train_data data/en-ko-qe/unified_qe_train.csv \
    --val_data data/en-ko-qe/unified_qe_val.csv \
    --output_dir outputs/cometkiwi-fthead-en-ko-1M \
    --mode fthead \
    --max_train_rows 1000000 \
    --epochs 3
```

### STEP 6: 접근법 6 - COMETKiwi LoRA (Parameter-Efficient)

LoRA 어댑터를 인코더에 적용하여 ~1%의 파라미터만 학습합니다.
Full fine-tuning의 과적합 위험 없이 인코더도 일부 적응시킬 수 있습니다.

```bash
# 추가 설치 필요
pip install peft>=0.6.0

# 6-1. LoRA Fine-tuning
python scripts/finetune_lora.py \
    --base_model $CHECKPOINT_PATH \
    --train_data data/en-ko-qe/unified_qe_train.csv \
    --val_data data/en-ko-qe/unified_qe_val.csv \
    --output_dir outputs/cometkiwi-lora-en-ko \
    --mode lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --learning_rate 1e-4 \
    --batch_size 16 \
    --epochs 3

# 6-2. BitFit (bias만 학습, 최소한의 변경)
python scripts/finetune_lora.py \
    --base_model $CHECKPOINT_PATH \
    --train_data data/en-ko-qe/unified_qe_train.csv \
    --val_data data/en-ko-qe/unified_qe_val.csv \
    --output_dir outputs/cometkiwi-bitfit-en-ko \
    --mode bitfit \
    --learning_rate 1e-4 \
    --epochs 5
```

**LoRA 하이퍼파라미터 가이드:**
| 파라미터 | 설명 | 권장값 |
|---------|------|--------|
| `lora_rank` | 저랭크 차원 (클수록 표현력 증가) | 8, 16, 32 |
| `lora_alpha` | 스케일링 팩터 (보통 rank의 2배) | 16, 32, 64 |
| `learning_rate` | PEFT는 더 높은 LR 사용 가능 | 1e-4 ~ 5e-4 |

### 학습 중 모니터링

학습이 시작되면 TensorBoard로 실시간 모니터링이 가능합니다:

```bash
# TensorBoard 실행 (별도 터미널)
tensorboard --logdir lightning_logs/

# 웹 브라우저에서 http://localhost:6006 접속
```

주요 모니터링 지표:
- `train_loss`: 학습 손실 (감소해야 함)
- `val_kendall`: Kendall τ 상관계수 (증가해야 함, 핵심 지표)
- `val_pearson`: Pearson 상관계수 (증가해야 함)
- `val_spearman`: Spearman 상관계수 (증가해야 함)

### 체크포인트 위치

학습이 완료되면 체크포인트는 다음 경로에 저장됩니다:
```
lightning_logs/
└── version_X/
    ├── checkpoints/
    │   ├── epoch=0-step=XXXX-val_kendall=0.XXXX.ckpt
    │   ├── epoch=1-step=XXXX-val_kendall=0.XXXX.ckpt
    │   └── epoch=2-step=XXXX-val_kendall=0.XXXX.ckpt
    ├── hparams.yaml
    └── events.out.tfevents.*
```

`val_kendall` 값이 가장 높은 체크포인트가 최적 모델입니다.

---

## 7. 모델 평가

### 7.1 검증 데이터에서 평가

```bash
# ReferencelessRegression 모델 평가
python scripts/evaluate_model.py \
    --checkpoint lightning_logs/version_X/checkpoints/best.ckpt \
    --test_data data/en-ko-qe/referenceless_val.csv \
    --model_type referenceless

# UnifiedMetric 모델 평가
python scripts/evaluate_model.py \
    --checkpoint lightning_logs/version_X/checkpoints/best.ckpt \
    --test_data data/en-ko-qe/unified_qe_val.csv \
    --model_type unified
```

출력 예시:
```
============================================================
[RESULTS] Evaluation Metrics
============================================================
  Pearson r:     0.8234 (p=1.23e-45)
  Spearman rho:  0.7891 (p=2.34e-40)
  Kendall tau:   0.6123 (p=3.45e-35)
  MSE:           0.012345
  MAE:           0.089012
============================================================
```

### 7.2 개별 문장 평가

```bash
python scripts/evaluate_model.py \
    --checkpoint lightning_logs/version_X/checkpoints/best.ckpt \
    --model_type referenceless \
    --src "activate a scanning directed acyclic graph to inspect the load-ready data" \
    --mt "로드 준비 데이터를 검사하기 위해 스캐닝 지시 비순환 그래프를 활성화하는 것"
```

### 7.3 comet-score CLI로 평가

```bash
# 텍스트 파일로 평가 (줄 단위)
comet-score \
    -s source_sentences.txt \
    -t mt_sentences.txt \
    --model lightning_logs/version_X/checkpoints/best.ckpt
```

### 7.4 Python API로 사용

```python
from comet import load_from_checkpoint

# 모델 로드
model = load_from_checkpoint("lightning_logs/version_X/checkpoints/best.ckpt")

# 예측
data = [
    {
        "src": "The method according to claim 13",
        "mt": "청구항 13에 따른 방법은"
    },
    {
        "src": "A device for controlling fluid flow",
        "mt": "유체 흐름을 제어하기 위한 장치"
    }
]

output = model.predict(data, batch_size=8, gpus=1)
print(output.scores)       # [0.82, 0.91]
print(output.system_score) # 0.865 (평균)
```

---

## 8. 하이퍼파라미터 튜닝

### 8.1 주요 하이퍼파라미터

| 파라미터 | 설명 | 기본값 | 튜닝 범위 |
|---------|------|--------|-----------|
| `encoder_learning_rate` | 인코더 학습률 | 1e-6 | 1e-7 ~ 5e-6 |
| `learning_rate` | Head 학습률 | 1.5e-5 | 1e-5 ~ 5e-5 |
| `batch_size` | 배치 크기 | 16 | 8, 16, 32 |
| `accumulate_grad_batches` | 그래디언트 누적 | 4 | 2, 4, 8, 16 |
| `nr_frozen_epochs` | 인코더 동결 기간 | 0.3 | 0.1 ~ 0.9 |
| `layerwise_decay` | 레이어별 학습률 감쇠 | 0.95 | 0.9 ~ 1.0 |
| `dropout` | 드롭아웃 비율 | 0.1 | 0.05 ~ 0.3 |
| `hidden_sizes` | Head 은닉층 | [2048, 1024] | 다양 |
| `max_epochs` | 최대 에폭 수 | 5 | 3 ~ 10 |
| `warmup_steps` | 워밍업 스텝 | 0 | 0 ~ 500 |

### 8.2 튜닝 우선순위

1. **유효 배치 크기** (`batch_size * accumulate_grad_batches * devices`)
   - 64~256이 일반적으로 좋음
   - 너무 크면 일반화 성능 저하

2. **학습률 비율** (`learning_rate / encoder_learning_rate`)
   - 보통 10~15x 차이
   - Fine-tuning 시 둘 다 절반으로 줄이기

3. **인코더 동결 기간** (`nr_frozen_epochs`)
   - From scratch: 0.3 (30% 후 unfreeze)
   - Fine-tuning: 0.5~0.9 (더 오래 동결)

4. **드롭아웃** - 과적합 징후가 보이면 0.15~0.2로 증가

### 8.3 대용량 데이터 학습 팁

데이터가 ~960만 행으로 매우 크므로:

```yaml
# 1. 그래디언트 누적으로 유효 배치 크기 증가
accumulate_grad_batches: 8    # 16 * 8 = 128 유효 배치

# 2. 에폭을 줄이고 데이터를 많이 봄
max_epochs: 2                 # 960만 * 2 = ~1920만 스텝

# 3. 멀티 GPU 활용
devices: 2
strategy: ddp                 # 분산 학습

# 4. Mixed Precision (VRAM 절약 + 속도 향상)
# trainer.yaml에서:
precision: 16                 # FP16 학습
```

### 8.4 YAML 설정 수정 방법

설정 파일을 직접 수정하거나, 커맨드라인에서 오버라이드할 수 있습니다:

```bash
# YAML 파일 수정 없이 파라미터 오버라이드
comet-train \
    --cfg configs/models/en-ko-qe/approach1_referenceless_scratch.yaml \
    --seed_everything 42 \
    --referenceless_regression_metric.init_args.batch_size 32 \
    --referenceless_regression_metric.init_args.learning_rate 2e-5
```

---

## 9. 문제 해결 (Troubleshooting)

### 9.1 CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**해결 방법:**
1. `batch_size`를 줄이기 (16 → 8 → 4)
2. `accumulate_grad_batches`를 늘려서 유효 배치 크기 유지
3. `keep_embeddings_frozen: True` 확인
4. `precision: 16` 추가 (Mixed Precision)
5. `max_length`가 큰 문장이 있으면 데이터 전처리에서 잘라내기

```yaml
# 메모리 절약 설정 예시
batch_size: 4
accumulate_grad_batches: 16   # 유효 배치 = 64
keep_embeddings_frozen: True
# trainer에 추가:
precision: 16
```

### 9.2 학습이 수렴하지 않음 (Loss가 감소하지 않음)

**원인과 해결:**
1. **학습률이 너무 크거나 작음** → `learning_rate`를 조정
2. **인코더가 너무 오래 동결** → `nr_frozen_epochs` 줄이기
3. **데이터 문제** → `score` 분포 확인, 이상치 제거
4. **배치 크기 문제** → 유효 배치 크기를 64~256으로 조정

### 9.3 val_kendall이 개선되지 않음

1. **과적합**: `dropout` 증가, `max_epochs` 줄이기
2. **학습 데이터 부족**: pairwise 데이터 포함 (`--include_pairwise`)
3. **인코더 문제**: `layerwise_decay`를 0.9로 낮추기

### 9.4 데이터 로딩 시 메모리 부족

데이터가 ~960만 행이면 RAM에서 로딩 시 문제가 될 수 있습니다.

```bash
# 데이터를 줄여서 학습
python scripts/prepare_data.py \
    --input_dir /path/to/train_data \
    --output_dir data/en-ko-qe \
    --max_train_rows 2000000    # 200만 행으로 제한
```

### 9.5 체크포인트 로드 실패

```
RuntimeError: Error(s) in loading state_dict
```

**해결:**
- `--strict_load` 옵션을 빼고 실행 (strict=False가 기본)
- 아키텍처가 체크포인트와 일치하는지 확인
  - COMETKiwi → UnifiedMetric (O)
  - COMETKiwi → ReferencelessRegression (X, 호환 불가)

### 9.6 Multi-GPU 학습 문제

```yaml
# DDP 설정
trainer:
  init_args:
    accelerator: gpu
    devices: 2                  # GPU 수
    strategy: ddp               # Distributed Data Parallel
    use_distributed_sampler: true
```

```bash
# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0,1
comet-train --cfg your_config.yaml
```

---

## 10. 참고 자료

### 10.1 공식 문서

- **COMET 공식 문서**: https://unbabel.github.io/COMET/html/index.html
- **COMET 학습 가이드**: https://unbabel.github.io/COMET/html/training.html
- **COMET 모델 카탈로그**: https://unbabel.github.io/COMET/html/models.html
- **COMET GitHub**: https://github.com/Unbabel/COMET
- **COMET 모델 목록 (MODELS.md)**: https://github.com/Unbabel/COMET/blob/master/MODELS.md

### 10.2 논문

- **COMET (원본)**: Rei et al., "COMET: A Neural Framework for MT Evaluation" (EMNLP 2020)
  - https://aclanthology.org/2020.emnlp-main.213/
- **COMETKiwi**: Rei et al., "COMETKiwi: IST-Unbabel 2022 Submission for the Quality Estimation Shared Task" (WMT 2022)
  - https://aclanthology.org/2022.wmt-1.60/
- **COMET-22**: Rei et al., "COMET-22: Unbabel-IST 2022 Submission for the Metrics Shared Task" (WMT 2022)
  - https://aclanthology.org/2022.wmt-1.52/
- **xCOMET**: Guerreiro et al., "xCOMET: Transparent Machine Translation Evaluation through Fine-grained Error Detection" (2023)
  - https://arxiv.org/abs/2310.10482
- **UniTE**: Wan et al., "UniTE: Unified Translation Evaluation" (ACL 2022)
  - https://arxiv.org/abs/2204.13346
- **MTQE.en-he** (COMETKiwi LoRA/BitFit Fine-tuning): "Machine Translation Quality Estimation for English-Hebrew" (2026)
  - https://arxiv.org/abs/2602.06546
  - Full fine-tuning의 위험성과 PEFT 방법의 우수성을 실증
- **AfriCOMET** (아프리카 언어 COMET): "AfriCOMET: COMET for African Languages" (2023)
  - https://arxiv.org/abs/2311.09828
- **Pitfalls in Using COMET**: Zouhar et al., "Pitfalls and Outlooks in Using COMET" (WMT 2024)
  - https://arxiv.org/abs/2408.15366
  - Fine-tuned 메트릭의 도메인 편향과 score 분포 문제 분석
- **COMET for Low-Resource MT Evaluation** (LREC 2024)
  - https://aclanthology.org/2024.lrec-main.315/
- **Cometoid** (지식 증류 기반 QE): "Cometoid: Knowledge Distillation from COMET" (WMT 2023)
  - https://aclanthology.org/2023.wmt-1.62/
- **BitFit**: Ben Zaken et al., "BitFit: Simple Parameter-efficient Fine-tuning" (ACL 2022)
  - https://aclanthology.org/2022.acl-short.1/
- **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022)
  - https://arxiv.org/abs/2106.09685

### 10.3 블로그 및 한국어 자료

- **COMET 신경망 기반 번역 품질 평가 지표 (한국어)**: https://velog.io/@judy_choi/NMT-COMET-%EC%8B%A0%EA%B2%BD%EB%A7%9D-%EA%B8%B0%EB%B0%98-%EB%B2%88%EC%97%AD-%ED%92%88%EC%A7%88-%ED%8F%89%EA%B0%80-%EC%A7%80%ED%91%9C

### 10.4 HuggingFace 모델 및 Fine-tuning 사례

- **wmt22-cometkiwi-da**: https://huggingface.co/Unbabel/wmt22-cometkiwi-da
- **wmt22-comet-da**: https://huggingface.co/Unbabel/wmt22-comet-da
- **XCOMET-XL**: https://huggingface.co/Unbabel/XCOMET-XL
- **wmt23-cometkiwi-da-xl**: https://huggingface.co/Unbabel/wmt23-cometkiwi-da-xl
- **ComeTH (EN-TH fine-tuned)**: https://huggingface.co/wasanx/ComeTH
  - COMETKiwi fine-tuning 성공 사례 (Spearman +4.9%)
- **ComeTH 데이터셋**: https://huggingface.co/datasets/wasanx/cometh_finetune
- **ComeTH 모델 컬렉션**: https://huggingface.co/collections/wasanx/cometh-model-682c410d3ba4cfbca8a07c9d

### 10.5 주요 소스 코드 경로

| 파일 | 설명 |
|------|------|
| `comet/cli/train.py` | 학습 CLI 진입점 |
| `comet/models/base.py` | 모든 모델의 기반 클래스 |
| `comet/models/regression/referenceless.py` | ReferencelessRegression 구현 |
| `comet/models/multitask/unified_metric.py` | UnifiedMetric (COMETKiwi) 구현 |
| `comet/encoders/xlmr.py` | XLM-RoBERTa 인코더 |
| `comet/modules/feedforward.py` | Feed-Forward Head |
| `comet/modules/layerwise_attention.py` | 레이어별 어텐션 |
| `configs/models/en-ko-qe/` | 본 가이드의 학습 설정 파일 |
| `scripts/prepare_data.py` | 데이터 전처리 스크립트 |
| `scripts/evaluate_model.py` | 모델 평가 스크립트 |
| `scripts/run_training.sh` | 학습 실행 스크립트 |
| `scripts/finetune_lora.py` | LoRA/BitFit/FTHead Fine-tuning 스크립트 |
| `scripts/download_checkpoint.py` | 체크포인트 다운로드 |

---

## 빠른 시작 요약 (Quick Start)

전체 과정을 한눈에 보려면:

```bash
# 1. 환경 설정
cd /path/to/COMET
pip install unbabel-comet scipy peft

# 2. 데이터 준비
python scripts/prepare_data.py \
    --input_dir /path/to/train_data \
    --output_dir data/en-ko-qe \
    --include_pairwise

# 3. 파이프라인 테스트
comet-train --cfg configs/models/en-ko-qe/approach_mini_test.yaml

# 4. COMETKiwi 체크포인트 다운로드
python scripts/download_checkpoint.py --model Unbabel/wmt22-cometkiwi-da
CKPT="체크포인트_경로"

# 5. [안전한 방법] FTHead Fine-tuning (Head만, 가장 먼저 시도!)
python scripts/finetune_lora.py \
    --base_model $CKPT \
    --train_data data/en-ko-qe/unified_qe_train.csv \
    --val_data data/en-ko-qe/unified_qe_val.csv \
    --output_dir outputs/cometkiwi-fthead \
    --mode fthead --epochs 5

# 6. [대용량 데이터 활용] Full Fine-tuning (ComeTH와 동일 전략)
comet-train \
    --cfg configs/models/en-ko-qe/approach3_finetune_cometkiwi.yaml \
    --load_from_checkpoint $CKPT \
    --seed_everything 12

# 7. [과적합 방지] LoRA Fine-tuning (과적합 시 대안)
python scripts/finetune_lora.py \
    --base_model $CKPT \
    --train_data data/en-ko-qe/unified_qe_train.csv \
    --val_data data/en-ko-qe/unified_qe_val.csv \
    --output_dir outputs/cometkiwi-lora \
    --mode lora --lora_rank 16 --epochs 3

# 8. 평가 (각 모델 비교)
python scripts/evaluate_model.py \
    --checkpoint outputs/cometkiwi-fthead/best_model.ckpt \
    --test_data data/en-ko-qe/unified_qe_val.csv \
    --model_type unified

# 9. 실제 사용
python -c "
from comet import load_from_checkpoint
model = load_from_checkpoint('outputs/cometkiwi-fthead/best_model.ckpt')
data = [{'src': 'The method of claim 1', 'mt': '청구항 1의 방법'}]
print(model.predict(data, batch_size=1, gpus=1).scores)
"
```
