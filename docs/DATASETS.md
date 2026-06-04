# 📚 시나리오별 데이터셋 설명

20개 교육 시나리오 각각에 어떤 HuggingFace 데이터셋을 매핑했는지, **무엇을·왜·어떻게(필드 매핑)·라이선스**를 정리한다.
출처: `scripts/fetch_hf_datasets.py` 의 `SOURCES` 레지스트리 + `data/scenarios.json`.

- 데이터 받기: `python scripts/fetch_hf_datasets.py --per-source 8000 --val-ratio 0.05`
- 목록 보기: `python scripts/fetch_hf_datasets.py --list`
- 일부만: `python scripts/fetch_hf_datasets.py --only socratic math`

> 통합 포맷: `{scenario_id, instruction, input, output, source}` · `instruction`=시나리오 행동 지시(=system).

---

## 1. 등록된 데이터 소스 (16개)

| 키 | HF 데이터셋 | 행수 | 라이선스 | lang | 시나리오 | input → output 매핑 |
|----|-------------|------|----------|------|----------|---------------------|
| `general` | [beomi/KoAlpaca-v1.1a](https://huggingface.co/datasets/beomi/KoAlpaca-v1.1a) | 21k | CC-BY-NC(추정) | ko | 0 (공통 베이스) | instruction → output |
| `socratic` | [JosephLee/korean-socratic-qa](https://huggingface.co/datasets/JosephLee/korean-socratic-qa) | 105k | 확인필요 | ko | 1 소크라테스 문답 | input(맥락) → target(역질문) |
| `reasoning` | [SabaPivot/KMMLU-Summarized-Chain_of_Thought](https://huggingface.co/datasets/SabaPivot/KMMLU-Summarized-Chain_of_Thought) | 7k | 확인필요 | ko | 3 단계적 추론 | 문제+보기 → CoT+정답 (빈 CoT 제외) |
| `summary` | [daekeun-ml/naver-news-summarization-ko](https://huggingface.co/datasets/daekeun-ml/naver-news-summarization-ko) | 27k | Apache-2.0 | ko | 4 요약·핵심정리 | document → summary |
| `reading` | [klue/klue (mrc)](https://huggingface.co/datasets/klue/klue) | 23k | CC-BY-SA-4.0 | ko | 5 독해 질의응답 | 지문+질문 → 정답 |
| `math` | [kuotient/orca-math-word-problems-193k-korean](https://huggingface.co/datasets/kuotient/orca-math-word-problems-193k-korean) | 193k | CC-BY-SA-4.0 | ko | 6 수학 단계 풀이 | question → answer |
| `debug_en` | [taisazero/socratic-debugging-benchmark](https://huggingface.co/datasets/taisazero/socratic-debugging-benchmark) | 1k | MIT | en | 8 코딩 디버깅(힌트) | lm_input → lm_target |
| `roleplay` | [huggingface-KREW/korean-role-playing](https://huggingface.co/datasets/huggingface-KREW/korean-role-playing) | 32k | Apache-2.0 | ko | 9 인물 롤플레잉 | text(대화 JSON) → user/assistant |
| `science_en` | [allenai/sciq](https://huggingface.co/datasets/allenai/sciq) | 14k | CC-BY-NC-3.0 | en | 10 과학 교육 | question → support+answer |
| `empathy` | [jojo0217/korean_safe_conversation](https://huggingface.co/datasets/jojo0217/korean_safe_conversation) | 27k | Apache-2.0 | ko | 11 정서지원 | instruction → output |
| `code` | [m-a-p/CodeFeedback-Filtered-Instruction](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction) | 157k | Apache-2.0 | en | 13 코딩 실습·피드백 | query → answer |
| `motivation_en` | [to-be/annomi-motivational-interviewing-therapy-conversations](https://huggingface.co/datasets/to-be/annomi-motivational-interviewing-therapy-conversations) | 133 | OpenRAIL | en | 14 동기/목표 | conversations → user/assistant |
| `edu` | [neuralfoundry-coder/aihub-korean-education-instruct-sample](https://huggingface.co/datasets/neuralfoundry-coder/aihub-korean-education-instruct-sample) | 6k | CC-BY-NC-SA-4.0 | ko | 17 교육 상담·분석 | conversations → user/assistant |
| `writing` | [coastral/korean-writing-style-instruct](https://huggingface.co/datasets/coastral/korean-writing-style-instruct) | 29k | Apache-2.0 | ko | 18 글쓰기 스타일 | conversations(from/value) |
| `translation` | [bawin/korean-english-translation-1k](https://huggingface.co/datasets/bawin/korean-english-translation-1k) | 1k | 확인필요 | ko | 19 번역 | korean → english |
| `lesson_en` | [xriminact/brightai_edge_lesson_plan_dataset](https://huggingface.co/datasets/xriminact/brightai_edge_lesson_plan_dataset) | 4k | 확인필요 | en | 20 수업 지도안 | user → output(JSON 지도안) |

---

## 2. 시나리오별 매핑 (20개 전체)

| # | 시나리오 | 전용 데이터 | 비고 |
|---|----------|-------------|------|
| 0 | (공통 instruction 베이스) | `general` | 모든 시나리오의 기본 대화 능력 보존용 |
| 1 | 소크라테스 문답 | ✅ `socratic` | 맥락→역질문 직결 |
| 2 | 눈높이 비유 설명 | ⚠️ 전용 없음 | `general`/`edu` 스필오버 + few-shot |
| 3 | 메타인지/단계적 추론 | ✅ `reasoning` | KMMLU CoT |
| 4 | 지문 요약·핵심정리 | ✅ `summary` | (난이도조절에서 교체) |
| 5 | 지문 독해 질의응답 | ✅ `reading` | (개념연결에서 교체) |
| 6 | 수학 단계별 풀이 | ✅ `math` | 193k 대규모 |
| 7 | 영어 에세이 교정 | ⚠️ 전용 없음 | few-shot(seed) 권장 |
| 8 | 코딩 디버깅(힌트형) | ✅ `debug_en` | 영어, 소크라테스 디버깅 |
| 9 | 인물 롤플레잉 | ✅ `roleplay` | 대화 JSON |
| 10 | 과학 교육 | ✅ `science_en` | 영어 sciq |
| 11 | 정서지원 상담 | ✅ `empathy` | 시나리오 12·15도 일부 커버 |
| 12 | 과정 중심 칭찬 | ⚠️ 전용 없음 | `empathy` 스필오버 + few-shot |
| 13 | 코딩 실습·피드백 | ✅ `code` | (게이미피케이션에서 교체) |
| 14 | 동기/목표 설정 | ✅ `motivation_en` | 영어, 소규모(133) |
| 15 | 느린 학습자 배려 | ⚠️ 전용 없음 | `empathy`/`general` + few-shot |
| 16 | 시험 문제 생성 | ⚠️ 전용 없음 | `reasoning`(KMMLU)·`edu` 활용 가능 |
| 17 | 교육 상담·분석 | ✅ `edu` | AI Hub 기반 |
| 18 | 글쓰기 스타일 지도 | ✅ `writing` | (문체변환에서 교체) |
| 19 | 번역 | ✅ `translation` | 소규모(1k) |
| 20 | 수업 지도안 | ✅ `lesson_en` | 영어, JSON 지도안 |

**요약**: 전용 데이터 확보 **15개**(1·3·4·5·6·8·9·10·11·13·14·17·18·19·20) + 베이스 `general`(0).
전용 없음 **5개**(2·7·12·15·16) → 인접 소스 스필오버 + `seed_train.jsonl` few-shot으로 보완.

---

## 3. 라이선스 주의

| 등급 | 데이터 | 사용 |
|------|--------|------|
| **자유(상업 OK)** | Apache-2.0(`summary`,`empathy`,`code`,`writing`,`roleplay`), MIT(`debug_en`) | 제약 적음 |
| **공유조건** | CC-BY-SA(`reading`,`math`) | 동일 라이선스 공유 |
| **비상업(NC)** | `general`,`science_en`,`edu` | 연구·교육만 권장 |
| **확인 필요** | `socratic`,`reasoning`,`translation`,`lesson_en` | 배포 전 각 카드 확인 |
| **학습 제외(참고)** | KMMLU 원본(CC-BY-ND), korean_code_review(GPL·소규모) | 평가/참고용 |

> 상업 배포 시 NC/ND/확인필요 항목은 제외하거나 라이선스 확인 후 사용.

---

## 4. 새 데이터셋 추가/교체

절차와 검증은 [`SCENARIO_GUIDE.md`](SCENARIO_GUIDE.md) 참고. 핵심 2단계:
1. `data/scenarios.json` 에 시나리오 정의(+`dataset` 필드)
2. `scripts/fetch_hf_datasets.py` 의 `SOURCES` 에 매핑(+필요시 매퍼) → `--list`/`--only`로 검증

> 관련: 방법 선택 [`METHOD_SELECTION.md`](METHOD_SELECTION.md) · 노브 [`LEARNING.md`](LEARNING.md)
