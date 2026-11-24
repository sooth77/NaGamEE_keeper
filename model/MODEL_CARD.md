# 모델 카드 · 나감이 4B AWQ

## 개요

| 항목 | 내용 |
|---|---|
| 이름 | `final_student_4b_awq_asym` |
| 용도 | 공동생활 규약 근거 판정과 설명 (한국어) |
| 교사 모델 | Qwen3-32B |
| 학생 모델 | Qwen3-4B |
| 학습 | 지식 증류 후 대화셋 964개로 보강 학습 |
| 양자화 | AWQ W4A16 비대칭, 그룹 크기 128, `lm_head` 제외 ([`recipe.yaml`](recipe.yaml)) |
| 저장 형식 | `compressed-tensors` (pack-quantized), safetensors 1개 |
| 파일 크기 | 2,666,063,024 bytes (약 2.67GB) |
| 기반 모델 라이선스 | Apache 2.0 (Qwen3) |

가중치는 GitHub 파일 크기 제한(일반 100MB, LFS 무료 2GB)을 넘어서 이 저장소에 넣지 않았어요.
이 폴더에는 설정 파일과 양자화 레시피만 있어요.

## 구조

| 항목 | 값 |
|---|---|
| 아키텍처 | `Qwen3ForCausalLM` |
| 디코더 층 | 36 |
| 히든 크기 | 2,560 |
| FFN 중간 크기 | 9,728 |
| 어텐션 헤드 | 쿼리 32 / KV 8 (GQA) |
| 어휘 수 | 151,936 |

## 양자화 레시피

llm-compressor의 `AWQModifier`로 활성값 분포를 보고 스케일을 맞춘 뒤, `QuantizationModifier`로 선형층 가중치를 4bit로 바꿨어요.

| 스무딩 대상 | 함께 맞추는 층 |
|---|---|
| `input_layernorm` | `q_proj`, `k_proj`, `v_proj` |
| `v_proj` | `o_proj` |
| `post_attention_layernorm` | `gate_proj`, `up_proj` |
| `up_proj` | `down_proj` |

## 실행 설정

| 항목 | 값 |
|---|---|
| 온도 | 0 (샘플링 안 함) |
| 최대 생성 토큰 | 1,536 |
| 생각 모드 | 끔 (`enable_thinking=False`) |
| 평가 GPU | NVIDIA RTX PRO 6000 Blackwell |
| 최대 GPU 메모리 | 8.45GiB (공개 평가 실행 기록) |

## 성능 요약

공개 평가셋 200개 답변 기준, LangGraph와 함께 쓴 나감이 결과예요. 자세한 내용은 [`../docs/EVALUATION.md`](../docs/EVALUATION.md)에 있어요.

| 지표 | 나감이 | 32B 비교 기준 |
|---|---:|---:|
| 판정 정확도 (엄격) | 69.5% | 85.5% |
| 개정 대응 | 7/7 | 5/7 |
| 첫 토큰 (V3 중앙값) | 0.22초 | 0.84초 |

## 한계

- 여러 조항을 함께 따지는 서술형 답이 약해요 (여러 항목 서술 15%).
- 과거 사건에 어느 판본을 적용할지 고르는 문제를 자주 틀려요.
- 판단에 필요한 날짜나 금액이 빠졌을 때 되묻지 않고 단정하는 경우가 있어요.
- 이 과제의 가상 규약집에 맞춰 만든 모델이라, 실제 법률·재무 판단에 쓰면 안 돼요.
