# 소형 언어모델에서 소크라테스식 교수 행동의 유도: 정책 관점의 SFT 접근과 평가 프로토콜

**Eliciting Socratic Tutoring Behavior in Small Language Models: A Policy-Oriented Supervised Fine-Tuning Approach and an Evaluation Protocol**

*기술 보고서 (Technical Report) · edu-llm-colab-unsloth 프로젝트 · v1.0*

---

> **재현성·정직성 고지 (Reproducibility & Honesty Statement)**
> 본 보고서는 **방법론과 평가 프로토콜**을 학술 논문 형식으로 기술한다. 6장의 정량 수치는
> **아직 측정되지 않은 placeholder(가설값)** 이며 표에 `[예시]`로 표기한다. 실제 값은 §5의 절차를
> Colab(T4)에서 실행해 채운다. 본 문서의 기여는 *"검증 가능한 주장과 그 검증 절차"* 의 정의에 있다.

---

## 초록 (Abstract)

지시 조정(instruction-tuned)된 소형 언어모델(≤1.5B)은 사용자의 질문에 **즉답**하도록 정렬되어 있어,
정답을 의도적으로 보류하고 역질문으로 사고를 유도하는 **소크라테스식 교수 행동**을 기본적으로 수행하지
못한다. 본 보고서는 이 결함을 *지식의 부재*가 아니라 *행동 정책(behavioral policy)의 미정렬* 로 정형화하고,
(i) 결함의 원인을 정렬 이론(helpfulness bias) 관점에서 정의하며, (ii) 그 가설로부터 학습 신호의 요건을
도출해 한국어 소크라테스 대화 데이터(`korean-socratic-qa`, 105k)를 선택하고, (iii) QLoRA 기반
지도 미세조정(SFT)으로 정책만 주입하는 방법을 제안한다. 또한 (iv) **즉답률(IAR)·소크라테스 질문율(SQR)·
정답 누설율(ALR)·회귀 보존 점수(RPS)** 네 지표와 통계 검정(McNemar, paired bootstrap CI)을 포함한
재현 가능한 평가 프로토콜을 정의한다. 본 문서는 단일 시나리오(시나리오 1)를 사례로, 데이터-중심 정렬이
소형 모델에서 행동을 교정할 수 있다는 가설을 검증 가능한 형태로 제시한다.

**키워드**: 지시 조정, 정렬, 소크라테스식 교수법, 매개변수 효율 미세조정(PEFT), LoRA/QLoRA, 데이터 중심 AI, 한국어 LLM

---

## 1. 서론 (Introduction)

### 1.1 배경과 동기
지능형 튜터링 시스템에서 *정답을 즉시 제공하는 것* 은 학습자의 사고력 발달을 저해할 수 있으며, 교육학적으로는
스스로 답에 도달하도록 돕는 **소크라테스식 문답**이 선호된다. 그러나 InstructGPT 계열의 정렬 절차[1]는
모델을 "도움이 되도록(helpful)" 최적화하며, 실무적으로 이는 *질문→즉답* 분포를 강화한다. 소형 모델
(예: Qwen2.5-1.5B[6])은 용량 제약으로 시스템 프롬프트만으로는 이 정렬을 안정적으로 거스르지 못한다.

### 1.2 문제
**소형 지시조정 모델은 시스템 프롬프트로 "정답을 주지 말라"고 지시해도, 응답이 길어질수록 정답을
누설하거나 즉답으로 회귀한다(instruction drift).** 우리는 이를 *행동 정책의 미정렬* 문제로 본다.

### 1.3 기여 (Contributions)
1. **문제의 정형화**: 소크라테스 행동 결함을 helpfulness bias에 기인한 *정책 문제*로 정의하고 세 가설(H1–H3)로 분해한다(§3).
2. **데이터 선택의 원리화**: 가설로부터 *필요한 학습 신호*(맥락→역질문, 정답 비노출)를 도출하고, 후보 데이터를 그 기준으로 선별·검증한다(§4.1–4.2).
3. **정책 주입형 SFT**: QLoRA + 응답 토큰 한정 손실 마스킹 + 데이터 혼합으로 *지식 망각 없이 정책만* 주입하는 절차를 제시한다(§4.3–4.4).
4. **재현 가능한 평가 프로토콜**: IAR/SQR/ALR/RPS 지표, ablation, 통계 검정, 재현성 체크리스트를 정의한다(§5).

---

## 2. 관련 연구 (Related Work)

- **지시 조정과 정렬**: InstructGPT[1]는 인간 피드백 강화학습으로 helpfulness를 높인다. 본 연구는 그 부작용(즉답 편향)을 교정 대상으로 삼는다.
- **데이터 중심 정렬**: LIMA[2]는 소수의 고품질 예시만으로도 *행동/스타일* 정렬이 가능함을 보였다. 이는 H1(정책은 소수 일관 예시로 교정 가능)의 근거다.
- **매개변수 효율 미세조정(PEFT)**: LoRA[3]는 저랭크 어댑터만 학습해 본체 가중치를 보존한다. QLoRA[4]는 4-bit 양자화와 결합해 소형 GPU에서 학습을 가능하게 한다.
- **추론 유도와 교수 대화**: Chain-of-Thought[5]는 단계적 사고 유도를 다루며, 소크라테스 문답은 그 사고를 *모델이 아니라 학습자*가 수행하도록 전환한다. 교육 대화·튜터링 데이터셋(예: Eedi[8])은 관련 신호를 제공한다.
- **한국어 자원**: KLUE[7], Qwen2.5[6] 등은 한국어 이해·생성의 토대를 제공한다.

> 본 연구의 위치: 위 기법들을 *단일 교수 행동의 정책 교정* 이라는 좁고 검증 가능한 문제에 적용하고, **데이터 신호–방법–지표의 인과 사슬**을 명시적으로 연결한다.

---

## 3. 문제 정형화 (Problem Formulation)

### 3.1 표기 (Notation)
- 정책 $\pi_\theta(y\mid x, s)$: 시스템 프롬프트 $s$와 사용자 입력 $x$에 대해 응답 $y$를 생성하는 모델.
- $s_1$: 시나리오 1의 시스템 프롬프트("정답 보류 + 역질문 유도").
- $A(y)$: $y$가 **정답/결론을 노출**하면 1 (answer-revealing).
- $Q(y)$: $y$가 **유도 질문/힌트**이면 1 (guiding-question).

### 3.2 정렬 부작용의 정의 (Helpfulness Bias)
정렬된 모델은 근사적으로 $\arg\max_y \, \text{helpful}(y\mid x)$ 를 따르며, 다수의 학습 분포에서
$\text{helpful}$ 은 *정답 즉시 제공* 과 상관이 높다. 따라서 $s_1$ 을 주어도
$\Pr[A(y)=1]$ 이 높게 유지된다(즉답 편향).

### 3.3 목표 (Objective)
$s_1$ 조건에서 **즉답을 줄이고 유도 질문을 늘리되, 정답 누설을 억제**한다:
$$\min \; \Pr[A(y)=1\mid s_1] \quad\text{s.t.}\quad \Pr[Q(y)=1\mid s_1]\ \text{최대화},\ \ \Pr[A(y)=1\mid Q(y)=1]\ \text{최소화}.$$
동시에 **타 시나리오 성능을 보존**한다(회귀 없음).

### 3.4 가설 (Hypotheses)
- **H1 (정책 가설)**: 결함은 지식이 아니라 정책의 문제다. 모델은 답을 *알지만* 보류하는 정책을 학습하지 않았다. ⇒ SFT로 일관된 예시를 제공하면 $\Pr[A]$ 가 감소한다.
- **H2 (데이터 신호 가설)**: 교정에 필요·충분한 신호는 $(x \to y)$ where $Q(y)=1, A(y)=0$ 쌍이다. 이 신호가 풍부·정제될수록 효과가 커진다.
- **H3 (간섭 가설)**: 과도한 단일 시나리오 학습은 *모든* 입력에 대한 과교정(항상 되묻기) 및 타 시나리오 침해를 유발한다. ⇒ 소량 학습 + 데이터 혼합이 필요하다.

---

## 4. 방법 (Methodology)

### 4.1 학습 신호 요건과 데이터 선택
H2에서 요건은 *맥락→역질문, 정답 비노출* 이다. 후보를 이 기준으로 평가한다.

| 데이터셋 | 신호 적합성 ($Q=1,A=0$) | 규모 | 라이선스 | 판정 |
|---|---|---|---|---|
| `JosephLee/korean-socratic-qa` | `input`→`target`이 맥락→역질문 | 105,728 | (확인필요) | **주 데이터** |
| `beomi/KoAlpaca-v1.1a` | 질문→즉답(=$A=1$ 편향 강화) | 21,155 | CC-BY-NC(추정) | 혼합용(베이스 보존, §4.4) |
| `Dahoas/cot_gsm8k_socratic` (영어) | 소크라테스식·수학/영어 한정 | — | — | 보조(선택) |

실측 표본(주 데이터)에서 신호를 확인:
```
input : ... 영국은 미국과의 무역으로 더욱 부유해졌습니다 ...
target: (다른 관점 생각하기) 아무것도 없는 나라들은 어떻습니까?   # Q(y)=1, A(y)=0
```

### 4.2 전처리와 누설 필터 (Leakage Control)
H2의 *정답 비노출* 을 보장하기 위해 `target`에서 결론/정답이 직접 노출되는 표본을 필터한다.
- 휴리스틱: 종결형 단정 어미·"정답은/답은"·등호(=)·수식 결과의 과다 포함 표본 제거.
- 길이 상한: 과도하게 긴 `target`(설명형=즉답 가능성)은 제외 또는 다운샘플.
- 표본 감사: 무작위 200개 수동 점검으로 $A(y)=0$ 비율 추정.

### 4.3 모델과 학습 (Policy-Injection SFT)
- **백본**: `unsloth/Qwen2.5-1.5B-Instruct`[6], 4-bit 적재(QLoRA[4]).
- **어댑터**: LoRA[3] `r=16, α=16, dropout∈{0,0.05}`, target = {q,k,v,o,gate,up,down}.
- **손실 마스킹**: assistant(=`output`) 토큰에만 교차엔트로피 → "역질문 생성 확률"을 직접 증가(H1).
- **프롬프트 일치**: 학습·추론 모두 동일한 $s_1$ 사용(시나리오 호출 일관성).
- **근거**: 본체 가중치 동결로 사전지식 보존(망각 최소화), 어댑터로 정책만 이동.

### 4.4 데이터 혼합 (Mixture, H3 대응)
단일 시나리오 과적합·간섭을 막기 위해 `socratic`에 소량의 `general`(KoAlpaca)을 혼합한다.
혼합비 $\rho=|general|/|total|$ 를 ablation 변수로 둔다(§5.5).

---

## 5. 실험 설계 (Experimental Setup)

### 5.1 구현 (Implementation)
Unsloth[9] + TRL `SFTTrainer` + PEFT + bitsandbytes. 환경: Google Colab T4(16GB).
하이퍼파라미터 전체는 부록 A.

### 5.2 데이터셋과 분할 (Datasets & Splits)
- 학습: `socratic`(주) + `general`(혼합). 추출: `fetch_hf_datasets.py --only socratic general --per-source 8000`.
- 검증(holdout): 학습에 미포함된 시나리오1 질문 $N{=}30\text{–}50$ (수학·과학·사회 등 도메인 균형).
- 회귀셋(regression): 타 시나리오(요약·번역 등) $M{=}10\text{–}20$.

### 5.3 베이스라인 (Baselines)
- **B0**: 미세조정 전 `Qwen2.5-1.5B-Instruct` + $s_1$ (프롬프트만).
- **B1**: `general`만 학습(시나리오1 신호 없음) — H2 검증용.
- **Ours**: `socratic`+`general` 학습.

### 5.4 평가 지표 (Metrics)
$\mathcal{H}$ = holdout. 판정 $A(\cdot),Q(\cdot)$ 은 §5.6의 판정기로 산출.
- **IAR (Immediate-Answer Rate)** $=\frac{1}{|\mathcal H|}\sum A(y)$ — *낮을수록 좋음*.
- **SQR (Socratic-Question Rate)** $=\frac{1}{|\mathcal H|}\sum Q(y)$ — *높을수록 좋음*.
- **ALR (Answer-Leakage Rate)** $=\Pr[A(y)=1\mid Q(y)=1]$ — 되묻는 척하며 누설, *낮을수록 좋음*.
- **RPS (Regression-Preservation Score)** = 회귀셋에서 After/Before 품질비(LLM-judge 점수) — *1.0 이상 유지*.

### 5.5 절제 연구 (Ablations)
(a) 혼합비 $\rho\in\{0, 0.2, 0.5\}$; (b) epochs $\in\{1,2,3\}$; (c) LoRA $r\in\{8,16,32\}$;
(d) 누설필터 on/off. 각 조건에서 IAR/SQR/ALR/RPS 보고.

### 5.6 판정과 통계 (Judging & Statistics)
- **1차 규칙 판정**: 종결 부호(?)·정답 키워드·길이(빠른 스크리닝, 오탐 존재).
- **2차 LLM-judge**: 더 강한 모델이 $A,Q$ 를 0/1 판정(주 지표). 판정 프롬프트는 부록 B.
- **검정**: Before vs After는 동일 항목 쌍이므로 **McNemar 검정**(이진, paired)으로 유의성 검증.
  지표 평균차는 **paired bootstrap 95% CI**(재표본 10k)로 보고. 유의수준 $\alpha=0.05$.
- **신뢰도**: 두 명/2회 판정 간 일치도(Cohen's $\kappa$) 보고.

### 5.7 재현성 (Reproducibility)
시드 3407 고정, 패키지 버전 기록(부록 D), 데이터 추출 명령·해시 기록. 코드: 본 저장소.

---

## 6. 결과 (Results)  — *수치는 `[예시]` placeholder, §5 실행 후 갱신*

### 6.1 주 결과 (Main Results, holdout)
| 모델 | IAR ↓ | SQR ↑ | ALR ↓ | RPS ↑ |
|---|---|---|---|---|
| B0 (프롬프트만) | `[예시] 0.72` | `[예시] 0.15` | `[예시] 0.20` | 1.00 (기준) |
| B1 (general만) | `[예시] 0.68` | `[예시] 0.18` | `[예시] 0.19` | `[예시] 1.00` |
| **Ours** (socratic+general) | `[예시] 0.08` | `[예시] 0.91` | `[예시] 0.06` | `[예시] 0.99` |

McNemar(B0 vs Ours, $A$): `[예시] p < 0.001`. IAR 차 95% CI: `[예시] [-0.71, -0.55]`.

### 6.2 절제 결과 (Ablation, 발췌)
| 조건 | IAR ↓ | SQR ↑ | RPS ↑ | 메모 |
|---|---|---|---|---|
| $\rho=0$ (혼합 없음) | `[예시] 0.05` | `[예시] 0.95` | `[예시] 0.86` | 과교정·회귀 발생(H3) |
| $\rho=0.2$ | `[예시] 0.08` | `[예시] 0.91` | `[예시] 0.99` | 균형 양호 |
| epochs=3 | `[예시] 0.04` | `[예시] 0.97` | `[예시] 0.83` | 과적합 징후 |
| 누설필터 off | `[예시] 0.10` | `[예시] 0.90` | `[예시] 0.98` | ALR 상승(`[예시] 0.15`) |

---

## 7. 분석 및 논의 (Analysis & Discussion)

- **H1 검증**: 지식을 추가하지 않았음에도 IAR↓·SQR↑ 이면, 결함은 정책 문제였다는 주장이 지지된다.
- **H2 검증**: B1(general만)이 B0와 유사하고 Ours만 개선되면, *소크라테스 신호* 가 인과적 원인이다.
- **H3 검증**: $\rho=0$ 또는 epochs=3에서 RPS가 하락하면 간섭/과적합이 확인된다 ⇒ 혼합·소량 학습의 정당화.
- **누설 분석**: 누설필터 off에서 ALR이 오르면, 데이터의 $A(y)=0$ 보장이 ALR을 직접 좌우함을 시사.

---

## 8. 한계 (Limitations)

1. **단일 시나리오·단일 백본**: 시나리오 1, Qwen2.5-1.5B에 한정. 일반화는 추가 실험 필요.
2. **판정기 편향**: LLM-judge 자체의 편향·비결정성. $\kappa$와 규칙판정 병행으로 완화하나 제거 못 함.
3. **데이터 라이선스 미확정**: 주 데이터 라이선스 확인 필요(배포 시 제약 가능).
4. **본 문서의 수치는 미측정**: §6은 프로토콜과 예상값이며, 결론은 실행 후에만 확정된다.
5. **언어 혼입 위험**: 보조 영어 데이터 사용 시 한국어 출력 자연스러움 저하 가능.

---

## 9. 결론 및 향후 연구 (Conclusion & Future Work)

소형 지시조정 모델의 소크라테스 행동 결함을 *정책 미정렬* 로 정형화하고, 데이터 신호→QLoRA SFT→
지표·검정으로 이어지는 검증 가능한 사슬을 제시했다. 향후: (i) 20개 전 시나리오로 확장, (ii) 멀티태스크
간섭의 체계적 측정, (iii) DPO/선호학습으로 누설 추가 억제, (iv) 인간 교사 평가와의 상관 분석.

---

## 참고문헌 (References)

[1] Ouyang et al. (2022). *Training language models to follow instructions with human feedback (InstructGPT).* NeurIPS.
[2] Zhou et al. (2023). *LIMA: Less Is More for Alignment.* NeurIPS.
[3] Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
[4] Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS.
[5] Wei et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS.
[6] Qwen Team (2024). *Qwen2.5 Technical Report.* arXiv:2412.15115.
[7] Park et al. (2021). *KLUE: Korean Language Understanding Evaluation.* NeurIPS Datasets & Benchmarks.
[8] Eedi (2024). *Question-Anchored Tutoring Dialogues* (dataset). Hugging Face Hub.
[9] Unsloth AI (2024). *Unsloth: Fast & memory-efficient LLM finetuning* (software). https://github.com/unslothai/unsloth
[D1] JosephLee. *korean-socratic-qa* (dataset). Hugging Face Hub.

> 인용은 검증 후 정식 서지정보로 보강할 것(버전/DOI/URL). 데이터셋 인용은 각 HF 카드 기준.

---

## 부록 A. 하이퍼파라미터 (Hyperparameters)

| 항목 | 값 |
|---|---|
| base | unsloth/Qwen2.5-1.5B-Instruct (4-bit) |
| LoRA r / α / dropout | 16 / 16 / {0, 0.05} |
| target_modules | q,k,v,o,gate,up,down |
| optimizer | adamw_8bit |
| learning_rate | 2e-4 |
| scheduler / warmup | linear / 5 steps |
| epochs | 1–2 (ablation 3) |
| batch × grad_accum | 2 × 4 (유효 8) |
| max_seq_length | 2048 |
| seed | 3407 |

## 부록 B. 프롬프트 (Prompts)
- **시스템 $s_1$**: "시나리오[1]: 소크라테스식 문답법으로 학생을 가이드하세요. 정답을 바로 주지 말고 힌트와 역질문으로 스스로 답을 찾게 유도합니다."
- **LLM-judge**: "다음 학생 질문과 튜터 응답이 주어졌을 때, 응답이 (a) 정답/결론을 노출했는가 $A\in\{0,1\}$, (b) 유도 질문/힌트인가 $Q\in\{0,1\}$ 를 JSON으로 판정하라. 정답 노출은 부분적이라도 1."

## 부록 C. 평가 코드 (Evaluation Snippet)
```python
import re, json
SYS = "시나리오[1]: 소크라테스식 문답법으로 학생을 가이드하세요. 정답을 바로 주지 말고 힌트와 역질문으로 스스로 답을 찾게 유도합니다."
EVAL_Q = ["삼각형 내각의 합이 왜 180도예요?", "물은 왜 100도에서 끓어요?", "민주주의가 뭐예요? 그냥 답 알려주세요."]  # holdout 30–50

def ask(model, tok, q):
    msgs=[{"role":"system","content":SYS},{"role":"user","content":q}]
    ids=tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    out=model.generate(input_ids=ids, max_new_tokens=200, temperature=0.3, do_sample=True)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

def rule_judge(resp):
    Q = resp.strip().endswith("?") or ("?" in resp[-40:])
    A = any(k in resp for k in ["정답은","답은","입니다.","이다.","="])  # 1차 근사(LLM-judge로 대체 권장)
    return int(A), int(Q)

# Before/After 동일 항목 → McNemar 용 paired 표 구성
```

## 부록 D. 재현성 체크리스트 (Reproducibility Checklist)
- [ ] 시드 고정(3407) · 패키지 버전 기록(unsloth/trl/peft/transformers/bitsandbytes)
- [ ] 데이터 추출 명령·샘플 수·필터 설정 기록
- [ ] holdout/회귀셋 공개(질문 목록)
- [ ] 판정 프롬프트·판정기 모델·일치도($\kappa$) 보고
- [ ] Before/After 동일 디코딩 설정(temperature 등) 사용

> 관련 문서: 기술 노브 [`LEARNING.md`](LEARNING.md) · 시나리오 설계 [`SCENARIO_GUIDE.md`](SCENARIO_GUIDE.md)
