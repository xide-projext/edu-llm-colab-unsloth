# 🎓 edu-llm-colab-unsloth

Google Colab 무료 GPU에서 **소형 LLM을 교육용으로 파인튜닝**하는 실험 환경입니다.
[Unsloth](https://github.com/unslothai/unsloth)를 사용해 무료 T4 GPU에서도 OOM 없이 빠르게 학습합니다.
20가지 교육 시나리오(소크라테스 문답·단계별 채점·독해·코딩 등)를 실제 HuggingFace 데이터셋으로 학습합니다.

> GPT-2가 아니라 더 똑똑한 최신 초경량 모델(Qwen2.5-1.5B / Llama-3.2-1B)을 기본값으로 사용합니다.

---

## 🛠️ 사용 기술 / 도구 (Tech Stack)

| 분류 | 도구 | 역할 |
|------|------|------|
| **실행 환경** | [Google Colab](https://colab.research.google.com) | 무료 T4 GPU 클라우드 노트북. 설치 없이 브라우저에서 학습 |
| **파인튜닝 엔진** | [Unsloth](https://github.com/unslothai/unsloth) | LLM 파인튜닝 최적화 라이브러리. 메모리 ~80%↓·속도 2~3배↑로 무료 GPU에서도 OOM 방지 |
| **학습 프레임워크** | [TRL](https://github.com/huggingface/trl) (`SFTTrainer`) | 지도학습 파인튜닝(SFT) 트레이너 |
| **경량 학습 기법** | [PEFT](https://github.com/huggingface/peft) (LoRA/QLoRA) | 전체가 아닌 일부 어댑터만 학습 → 메모리·시간 절약 |
| **양자화** | [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | 4bit 양자화로 모델을 작은 VRAM에 적재 |
| **모델 백본** | [Transformers](https://github.com/huggingface/transformers) · [Qwen2.5](https://huggingface.co/Qwen) / [Llama-3.2](https://huggingface.co/meta-llama) | 사전학습 초경량 LLM (한국어 우수) |
| **데이터** | [HuggingFace Datasets](https://github.com/huggingface/datasets) | 한국어 교육 데이터셋을 streaming으로 추출·변환 |
| **데이터 소스 관리** | `scripts/fetch_hf_datasets.py` | 16개 HF 데이터셋 → 20시나리오 통합 포맷 변환 (자체 레지스트리) |
| **버전 관리/배포** | Git · GitHub (`gh`) | 코드/노트북 관리, Colab에서 clone |
| **내보내기(선택)** | GGUF → [Ollama](https://ollama.com) / [LM Studio](https://lmstudio.ai) / llama.cpp | 학습한 모델을 로컬 추론 엔진에서 사용 |

**전체 흐름**: `Colab(T4)` → `Unsloth`로 `Qwen2.5` 4bit 적재(`bitsandbytes`) → `PEFT(LoRA)` 어댑터 추가 →
`HF Datasets`에서 교육 데이터 fetch → `TRL SFTTrainer`로 학습 → 추론 테스트 → LoRA/GGUF 저장.

> 📖 각 기술을 **왜·언제·무엇을 조절하나(vibe learning)** 관점으로 정리한 학습 문서: **[docs/LEARNING.md](docs/LEARNING.md)**
> — 입문자는 이 문서의 *큰 그림*과 *치트시트*부터 보세요.
>
> 🧭 **파인튜닝 방법 선택 사고법** — Unsloth의 여러 방법(SFT·CPT·DPO·GRPO 등) 중 *왜 그것인가*를 가설로 먼저 생각하게 가르치는 문서: **[docs/METHOD_SELECTION.md](docs/METHOD_SELECTION.md)**
>
> 🧩 시나리오를 **새로 만들거나 고를 때** 필요한 최소 정보·데이터셋 검색·등록 절차: **[docs/SCENARIO_GUIDE.md](docs/SCENARIO_GUIDE.md)**
>
> 📑 **실험 보고서(논문 규격)** — 시나리오 1 사례: 문제 정형화·가설(H1–H3)·데이터 선택 원리·QLoRA SFT·평가 프로토콜(IAR/SQR/ALR/RPS, McNemar/bootstrap)·재현성: **[docs/CASE_STUDY_scenario1_socratic.md](docs/CASE_STUDY_scenario1_socratic.md)**
>
> 🧪 **Gemma 4 (E2B) Before→After 워크스루** — 결함 재현→파인튜닝→검증 + 기술 선택 내비게이션: **[docs/SOCRATIC_GEMMA4_WALKTHROUGH.md](docs/SOCRATIC_GEMMA4_WALKTHROUGH.md)** (노트북 `socratic_gemma4_before_after.ipynb`)
>
> 📚 **시나리오별 데이터셋 설명** — 16개 소스의 매핑·필드·라이선스: **[docs/DATASETS.md](docs/DATASETS.md)**

---

## 🚀 빠른 시작 (3단계)

1. **GitHub에 업로드** → `notebooks/unsloth_edu_finetune.ipynb` 를 [Google Colab](https://colab.research.google.com)에서 엽니다.
   (Colab → 파일 → 노트 업로드, 또는 GitHub 탭에서 저장소 열기)
2. 상단 메뉴 **런타임 → 런타임 유형 변경 → T4 GPU** 선택
3. **런타임 → 모두 실행** (Run all)

노트북이 자동으로 이 저장소를 clone하고 HuggingFace에서 학습 데이터를 받습니다:
```bash
git clone https://github.com/xide-projext/edu-llm-colab-unsloth.git
```

---

## 📁 구조

```
edu-llm-colab-unsloth/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── unsloth_edu_finetune.ipynb       # ⭐ 메인 Colab 파인튜닝 (코드 기반, 재현용)
│   ├── socratic_gemma4_before_after.ipynb # 🧪 Gemma4 E2B 소크라테스 Before→After + 선택 내비게이션
│   └── unsloth_studio_ui.ipynb          # 🖥️ Unsloth Studio 웹 UI (노코드 체험용)
├── data/
│   ├── scenarios.json               # 20대 교육 시나리오 정의 (선정이유·기대치·dataset 매핑)
│   ├── seed_train.jsonl             # 시나리오별 목표 톤 예시 (참고용, 학습엔 미사용)
│   └── hf_train.jsonl               # fetch_hf_datasets.py 가 HF에서 생성 (gitignore)
└── scripts/
    ├── fetch_hf_datasets.py         # ⭐ HF 데이터셋 → 20시나리오 포맷 변환 (16개 소스 레지스트리)
    ├── prepare_dataset.py           # 시드/로컬 데이터 검증·통계·분할
    ├── build_notebook.py            # 메인 노트북(.ipynb) 재생성기
    └── build_studio_notebook.py     # Studio UI 노트북 재생성기
```

> **두 가지 사용법**
> - 🖥️ **노코드 체험**: `notebooks/unsloth_studio_ui.ipynb` → Colab에서 Unsloth Studio 웹 UI를 띄워 클릭으로 모델 다운로드·학습
> - ⚙️ **코드 재현 학습**: `notebooks/unsloth_edu_finetune.ipynb` → 16개 HF 데이터셋을 20시나리오로 변환해 재현 가능하게 학습

---

## 🤖 추천 모델 (가장 작은 것부터)

| 모델 | 크기 | 특징 |
|------|------|------|
| `unsloth/Qwen2.5-0.5B-Instruct` | 0.5B | 가장 빠름, 한국어 가능 |
| `unsloth/Qwen2.5-1.5B-Instruct` | 1.5B | **기본값**. 한국어 우수, 가성비 최고 |
| `unsloth/Llama-3.2-1B-Instruct` | 1B | 메타 1B급 최고 성능 |

노트북 3번 셀의 `MODEL_NAME` 한 줄만 바꾸면 됩니다.

---

## 📥 학습 데이터 — HuggingFace Hub에서 (권장)

손으로 만든 예시(`seed_train.jsonl`)는 **시나리오별 목표 톤 참고용**일 뿐, 실제 파인튜닝에는
HuggingFace Hub의 **실제 한국어 교육 데이터셋**을 받아 씁니다.

```bash
pip install datasets
# 소스별 8000개씩 streaming 추출 → data/hf_train.jsonl 생성
python scripts/fetch_hf_datasets.py --per-source 8000 --val-ratio 0.05
python scripts/fetch_hf_datasets.py --list            # 추천 목록만 보기
python scripts/fetch_hf_datasets.py --only socratic math   # 일부 소스만
```

### 추천 데이터셋 (시나리오 매핑)

| 소스 키 | HF 데이터셋 | 행수 | 라이선스 | 매핑 시나리오 |
|---------|-------------|------|----------|---------------|
| `general` | [beomi/KoAlpaca-v1.1a](https://huggingface.co/datasets/beomi/KoAlpaca-v1.1a) | 21k | CC-BY-NC-4.0(추정) | 0 · 일반 instruction 베이스 |
| `socratic` | [JosephLee/korean-socratic-qa](https://huggingface.co/datasets/JosephLee/korean-socratic-qa) | **105k** | 확인필요 | 1 · 소크라테스 문답 |
| `math` | [kuotient/orca-math-word-problems-193k-korean](https://huggingface.co/datasets/kuotient/orca-math-word-problems-193k-korean) | **193k** | CC-BY-SA-4.0 | 6 · 수학 단계 풀이 |
| `empathy` | [jojo0217/korean_safe_conversation](https://huggingface.co/datasets/jojo0217/korean_safe_conversation) | 27k | **Apache-2.0** | 11·12·15 · 정서지원 |
| `edu` | [neuralfoundry-coder/aihub-korean-education-instruct-sample](https://huggingface.co/datasets/neuralfoundry-coder/aihub-korean-education-instruct-sample) | 6k | CC-BY-NC-SA-4.0 | 2·16·17 · 교육 상담·분석 |
| `roleplay` | [huggingface-KREW/korean-role-playing](https://huggingface.co/datasets/huggingface-KREW/korean-role-playing) | 32k | **Apache-2.0** | 9 · 인물 롤플레잉 |
| `translation` | [bawin/korean-english-translation-1k](https://huggingface.co/datasets/bawin/korean-english-translation-1k) | 1k | 확인필요 | 19 · 번역 |
| `reasoning` | [SabaPivot/KMMLU-Summarized-Chain_of_Thought](https://huggingface.co/datasets/SabaPivot/KMMLU-Summarized-Chain_of_Thought) | 7k | 확인필요 | 3 · 단계적 추론 |

### 영어 데이터셋 (cross-lingual 행동 학습 · `lang=en`)
한국어 직결 데이터가 없는 시나리오를 영어 공개 데이터로 보강합니다. 모델(Qwen2.5)은 다국어이므로
**행동/형식**을 학습하는 데 유효합니다. 단, 출력까지 한국어로 만들려면 비중을 낮추거나 번역 전처리를 권장합니다.

| 소스 키 | HF 데이터셋 | 행수 | 라이선스 | 매핑 시나리오 |
|---------|-------------|------|----------|---------------|
| `debug_en` | [taisazero/socratic-debugging-benchmark](https://huggingface.co/datasets/taisazero/socratic-debugging-benchmark) | 1k | **MIT** | 8 · 소크라테스 디버깅 |
| `science_en` | [allenai/sciq](https://huggingface.co/datasets/allenai/sciq) | 14k | CC-BY-NC-3.0 | 10 · 과학 교육 |
| `lesson_en` | [xriminact/brightai_edge_lesson_plan_dataset](https://huggingface.co/datasets/xriminact/brightai_edge_lesson_plan_dataset) | 4k | 확인필요 | 20 · 수업 지도안 |
| `motivation_en` | [to-be/annomi-motivational-interviewing-therapy-conversations](https://huggingface.co/datasets/to-be/annomi-motivational-interviewing-therapy-conversations) | 133 | OpenRAIL | 14 · 동기/목표 |

### 교체된 시나리오 (4·5·13·18 — 데이터 없는 주제를 데이터 풍부한 주제로 변경)
데이터가 없던 4개 시나리오를 **실제 데이터셋이 풍부한 주제로 교체**했습니다. (`scenarios.json` 반영)

| 소스 키 | HF 데이터셋 | 행수 | 라이선스 | 교체 시나리오 |
|---------|-------------|------|----------|---------------|
| `summary` | [daekeun-ml/naver-news-summarization-ko](https://huggingface.co/datasets/daekeun-ml/naver-news-summarization-ko) | 27k | **Apache-2.0** | 4 · 지문 요약·핵심정리 *(← 난이도조절)* |
| `reading` | [klue/klue (mrc)](https://huggingface.co/datasets/klue/klue) | 23k | CC-BY-SA-4.0 | 5 · 지문 독해 질의응답 *(← 개념연결)* |
| `code` | [m-a-p/CodeFeedback-Filtered-Instruction](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction) | 157k | **Apache-2.0** | 13 · 코딩 실습·피드백 *(← 게이미피케이션)* |
| `writing` | [coastral/korean-writing-style-instruct](https://huggingface.co/datasets/coastral/korean-writing-style-instruct) | 29k | **Apache-2.0** | 18 · 글쓰기 스타일 지도 *(← 학부모 문체변환)* |

> ⚠️ **라이선스**: 상업적 사용 시 `empathy`(Apache-2.0)가 가장 자유롭습니다. KoAlpaca/AI Hub 계열은
> 비상업(NC) 조건이 있을 수 있으니 배포 전 각 데이터셋 카드를 확인하세요. 연구·교육 목적 파인튜닝엔 무방합니다.

### 통합 데이터 포맷 (JSONL 한 줄)
```json
{"scenario_id": 1, "instruction": "시나리오[1]: 소크라테스식 문답법으로...", "input": "학생 질문", "output": "튜터 답변", "source": "JosephLee/korean-socratic-qa"}
```
- `instruction` → 시스템 프롬프트(시나리오 유형 명시) → 모델이 상황을 **구별**하는 핵심
- `input` → 학생/사용자 발화 · `output` → 학습할 이상적 답변

### 커버리지 현황 (HF 검색 기반 — 합성 데이터 미사용)
- **✅ 20개 시나리오 전부 실제 데이터셋 확보** (4·5·13·18은 데이터 풍부한 주제로 교체)
- **한국어**: 0 · 1 · 2 · 3 · 4 · 5 · 6 · 9 · 11 · 12 · 15 · 16 · 17 · 18 · 19
- **영어 보강**: 8 · 10 · 13 · 14 · 20
- 교체로 빠진 원래 주제(난이도조절·개념연결·게이미피케이션·학부모 문체변환)는 공개 데이터가 없어, 필요 시 `seed_train.jsonl` few-shot으로만 활용하세요.

#### 참고 데이터셋 (학습 레지스트리 미포함)
- [Eedi/Question-Anchored-Tutoring-Dialogues-2k](https://huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k) (69k, CC-BY-NC-4.0) — 시나리오 4(적응형 튜터링) 후보. 턴 단위 행이라 InterventionId로 학생→교사 턴 페어링 전처리 필요(미연동).
- [HAERAE-HUB/KMMLU](https://huggingface.co/datasets/HAERAE-HUB/KMMLU) (17k⬇) — 한국어 시험 평가 벤치마크. **CC-BY-ND**(파생 금지)라 학습 부적합, **평가 전용**.
- [m-a-p/CodeFeedback-Filtered-Instruction](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction) (15k⬇) — 시나리오 8 대규모 영어 코드 피드백(힌트형 아님, 보강용).
- [ewhk9887/korean_code_review](https://huggingface.co/datasets/ewhk9887/korean_code_review) — 시나리오 8 한국어 후보지만 47행·GPL-3.0으로 소규모.

---

## 📜 20대 교육 시나리오

`data/scenarios.json` 에 전체 정의(시스템 프롬프트, 데이터 설계, 선정이유, 기대치)가 있습니다.

| # | 단계 | 시나리오 |
|---|------|----------|
| 1 | 인지능력 | 소크라테스식 문답 |
| 2 | 인지능력 | 눈높이 비유 설명 |
| 3 | 인지능력 | 메타인지 자극 (오답 피드백) |
| 4 | 인지능력 | 난이도 동적 조절 |
| 5 | 인지능력 | 개념 간 마인드맵 연결 |
| 6 | 교과특화 | 수학 서술형 단계별 채점 |
| 7 | 교과특화 | 영어 에세이 교정 |
| 8 | 교과특화 | 코딩 디버깅 (힌트형) |
| 9 | 교과특화 | 역사 인물 롤플레잉 |
| 10 | 교과특화 | 과학 실험 안전 가이드 |
| 11 | 정서지원 | 학습 슬럼프 상담 |
| 12 | 정서지원 | 과정 중심 칭찬 |
| 13 | 정서지원 | 주의 집중 게이미피케이션 |
| 14 | 정서지원 | 스몰 스텝 목표 설정 |
| 15 | 정서지원 | 느린 학습자 언어 배려 |
| 16 | 교사지원 | 시험 문제 자동 생성 |
| 17 | 교사지원 | 학생 종합의견 초안 |
| 18 | 교사지원 | 학부모 안내문 문체 변환 |
| 19 | 교사지원 | 다국어 가정통신문 |
| 20 | 교사지원 | 수업 지도안 설계 |

---

## ⚠️ 참고

- 일부 온라인 가이드의 `unsloth-studio` pip 패키지, localtunnel 비밀번호 방식 등은 부정확합니다.
  이 저장소는 **Unsloth 정식 노트북 방식**(FastLanguageModel + LoRA + SFTTrainer)을 사용합니다.
- 시드 데이터는 시나리오당 1~2개입니다. 실제 효과를 보려면 **시나리오당 30~100개**로 늘리세요.
