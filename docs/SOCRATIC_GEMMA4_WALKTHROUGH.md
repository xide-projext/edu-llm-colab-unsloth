# 🧪 Gemma 4 (E2B) 소크라테스 문답 — Before→After 워크스루 (기술 선택 내비게이션)

> 노트북 [`notebooks/socratic_gemma4_before_after.ipynb`](../notebooks/socratic_gemma4_before_after.ipynb) 의 동반 문서.
> "왜 이 기술을 선택했는가"를 **각 단계의 내비게이션**으로 설명한다.
> 이론 배경: [`METHOD_SELECTION.md`](METHOD_SELECTION.md) · 논문형 사례: [`CASE_STUDY_scenario1_socratic.md`](CASE_STUDY_scenario1_socratic.md)

---

## 0. 한눈에 (목표)

| | 내용 |
|---|---|
| **모델** | `unsloth/gemma-4-E2B-it` (초경량 instruct, 무료 T4) |
| **결함(Before)** | "정답을 바로 줘" → 소크라테스식 역질문을 안 함 |
| **가설** | *지식*이 아니라 *행동 정책* 문제 → SFT로 교정 가능 |
| **방법(After)** | korean-socratic-qa로 LoRA SFT(응답 토큰만 loss) |
| **확인** | 같은 프롬프트로 Before/After 비교 + 역질문률 근사 측정 |

---

## ✅ 핵심 4문답 (시작 전 반드시 답하기)

파인튜닝을 시작하기 전, 이 네 질문에 답이 있어야 한다. (노트북 셀 번호와 1:1 대응)

### Q1. 모델(학습 전)은 무엇으로 할 것인가?
**`unsloth/gemma-4-E2B-it`** (Gemma 4, effective 2B, **instruct**).
- 왜 이것: ① E2B는 초경량이라 **무료 Colab T4**에서 학습 가능, ② **instruct** 버전이라 대화·지시는 이미 알지만 *소크라테스 정책*은 미학습 → 결함을 깨끗하게 시연·교정하기 좋음.
- 왜 base(`-it` 없는)가 아닌가: base는 대화 정렬이 안 돼 있어 결함 비교의 기준선으로 부적합.
- 코드: 노트북 **1장** `FastModel.from_pretrained("unsloth/gemma-4-E2B-it", ...)`.

### Q2. 기존 모델에서 기대 결과가 안 나오는 것을 확인·기록했는가?
**예 — 노트북 2장(Before)에서 확인하고, 7장에서 수치로 기록한다.**
- **확인**: 동일한 소크라테스 시스템 지시(`SOCRATIC_SYS`)를 주고 3개 고정 질문(`PROBES`)을 던져, 모델이 **역질문 없이 정답/증명을 바로 설명**하는 출력을 그대로 출력·관찰(노트북 2장).
- **기록**: 같은 `PROBES`로 **역질문률(SQR 근사)·정답누설율(ALR 근사)** 을 계산해 Before 값으로 남김(노트북 7장). After와 **동일 프롬프트·동일 지표**로 대조하므로 변화가 정량 기록된다.
- 정밀 기록(IAR/SQR/ALR/RPS, McNemar/bootstrap)은 [`CASE_STUDY_scenario1_socratic.md`](CASE_STUDY_scenario1_socratic.md) §5–6.

### Q3. 학습 데이터셋에서 입력·출력은 어떻게 정의되는가?
원천: **`JosephLee/korean-socratic-qa`** (맥락 → 역질문). 통합 포맷으로 매핑(노트북 4장 `to_convo`):

| 학습 필드 | 정의 | 채워지는 값 |
|-----------|------|-------------|
| **입력(user)** | 시나리오 지시 + 학생 질문/맥락 | `instruction`(소크라테스 지시) + `input`(맥락) |
| **출력(assistant)** | 모델이 학습할 모범 답변 | `output` = **역질문/힌트** (정답 비노출) |

```python
user = ex['instruction'] + ('\n\n' + ex['input'] if ex['input'] else '')
convo = [{'role':'user','content':user}, {'role':'assistant','content':ex['output']}]
```
- 핵심: 출력이 *정답이 아니라 역질문*이어야 한다. 그리고 **loss는 출력(assistant) 토큰에만** 준다(아래 Q4).
- 베이스 능력 보존을 위해 `general`(KoAlpaca)을 소량 섞는다.

### Q4. 학습 방법은 무엇을 할 것인가?
**SFT(지도 미세조정) + LoRA**, 응답 토큰에만 손실.
- **방법 종류**: SFT (←결함이 *정책*이므로). CPT/DPO/GRPO 아님(이유는 §2.2).
- **적용 범위**: LoRA(본체 동결 + 어댑터). FFT 아님, 메모리·망각 이점.
- **설정**: `FastModel.get_peft_model(r=8, finetune_language_layers=True, finetune_vision_layers=False)` → `SFTTrainer` → **`train_on_responses_only`** 로 user 부분 마스킹(=출력에만 loss). 코드: 노트북 3·5장.

> 위 4문답이 채워지면 학습을 시작한다. 각 결정의 *대안/언제 바꾸나*는 아래 §2 내비게이션 참고.

---

## 1. 결함 재현 (Before) — 무엇을 보는가

같은 시스템 지시("정답 바로 주지 마")를 줘도, 소형 instruct 모델은 **즉답**한다.
이는 정렬(helpfulness) 편향의 결과로, 소형 모델은 프롬프트만으로 이 정책을 안정적으로 거스르지 못한다.
→ 노트북 2장에서 3개 질문으로 *역질문 없는 즉답* 을 눈으로 확인한다.

---

## 2. 기술 선택 내비게이션 (각 결정의 왜/대안/언제)

> 형식: **결정 → 왜 → 대안 → 언제 바꾸나**. (노트북의 🧭 박스와 동일)

### 2.1 모델 = Gemma 4 E2B
- **왜**: effective 2B로 무료 T4에서 학습 가능. instruct라 대화는 알지만 소크라테스 정책은 미학습 → 결함 시연에 적합.
- **대안**: E4B/12B(똑똑·느림·OOM) · base 버전(대화 미정렬, 부적합).
- **언제**: 품질 부족→E4B / OOM→`load_in_4bit=True`.

### 2.2 방법 = SFT (CPT/DPO/GRPO 아님)
- **왜**: 결함이 *정책(행동)* → (입력→모범 응답) 모방이 정확한 신호. (METHOD_SELECTION 결함유형표)
- **대안**: 지식부재=CPT · 우열만=DPO · 자동채점가능=GRPO.
- **언제**: 데이터 없으면 방법 이전에 데이터 확보부터.

### 2.3 적용 범위 = LoRA (FFT 아님)
- **왜**: 정책만 주입하면 되므로 본체 동결+어댑터로 충분. 망각·메모리 이점.
- **대안**: QLoRA(4bit, 더 절약) · FFT(전체, 대개 불필요).
- **언제**: 최종 정확도 한계→FFT 검토 / 메모리 부족→4bit.

### 2.4 LoRA r = 8
- **왜**: 행동 교정은 적은 용량으로 충분. 크면 과적합(모든 답을 역질문화).
- **대안**: r=16~32(표현력↑).
- **언제**: After 약함→r↑ / 과교정·회귀→r↓.

### 2.5 데이터 = korean-socratic-qa (+소량 general)
- **왜**: 신호가 (맥락→역질문, 정답 비노출)로 가설과 일치. general 혼합으로 과교정/망각(H3) 방지.
- **대안**: KoAlpaca 단독=즉답편향 강화(역효과) · socratic 단독=과교정 위험.
- **언제**: 다른 과제 회귀 발생→general 비중↑.

### 2.6 손실 = 응답 토큰만 (`train_on_responses_only`)
- **왜**: "역질문 생성" 확률을 직접 높임. 질문은 외우지 않음 → 정책 학습의 핵심.
- **대안**: 전체 시퀀스 loss(프롬프트까지 외움, 비효율).
- **언제**: 대화형 SFT에선 거의 항상 켠다.

---

## 3. 검증 (After) — 가설이 맞았는지

- 노트북 6장: **Before와 동일 프롬프트**로 재질의 → 역질문/힌트로 바뀌는지 확인.
- 노트북 7장: 역질문률(SQR)·정답누설율(ALR) **근사** 측정.
- 정밀 평가(IAR/SQR/ALR/RPS, McNemar/bootstrap, LLM-judge)는 [`CASE_STUDY_scenario1_socratic.md`](CASE_STUDY_scenario1_socratic.md) §5.
- **해석**: 지식을 안 넣었는데 행동이 바뀌면 → "정책 문제였다"(H1) 지지. general만 학습해도 안 바뀌면 → socratic 신호가 원인(H2).

---

## 4. 실패 모드 & 대응

| 증상 | 원인 | 대응 |
|------|------|------|
| 모든 질문에 무조건 역질문 | 과적합(H3) | epochs↓, general↑, r↓ |
| 되묻는 척 + 정답 노출 | 데이터 누설 | socratic 누설 필터 강화 |
| 다른 과제 품질 하락 | 망각/간섭 | general 비중↑, r↓ |
| OOM | E2B 16bit 메모리 | `load_in_4bit=True`, seq↓ |
| 한국어 어색 | (영어 혼입 시) | 영어 소스 비중↓ |

---

## 5. 정리 — 노트북이 가르치는 사고 흐름

```
결함 관찰(Before) → 유형 가설(정책) → 방법 선택(SFT/LoRA, 내비게이션)
   → 데이터 신호(역질문 쌍) → 학습(응답토큰 loss) → 검증(After 비교)
```
각 🧭 박스의 *왜/대안/언제* 가 곧 의사결정 훈련이다. 새 시나리오에도 같은 흐름을 적용하라.
