# 🧩 시나리오 만들기·고르기 가이드

새 교육 시나리오를 **추가하거나, 어떤 시나리오를 학습할지 고를 때** 필요한 최소 정보와 절차를 정리합니다.
핵심 원칙: **"행동을 한 문장으로 정의 → 그 행동을 가르칠 실제 데이터가 있는가"** 이 두 가지가 전부입니다.

---

## 1. 시나리오 = 무엇인가

하나의 시나리오 = **"이런 상황에서 이렇게 답해라"는 행동 패턴 1개**.
학습 데이터 한 줄은 항상 이 형태입니다:

```json
{"scenario_id": 1, "instruction": "시나리오[1]: ...행동 지시...", "input": "사용자 발화", "output": "이상적 답변", "source": "데이터셋ID"}
```

| 필드 | 역할 | 한 줄 설명 |
|------|------|-----------|
| `instruction` | system | **시나리오 행동 지시** — 모델이 상황을 구별하는 핵심 |
| `input` | user | 학생/사용자가 한 말 |
| `output` | assistant | 모델이 따라 배울 모범 답변 |
| `scenario_id` | 메타 | 시나리오 번호 |
| `source` | 메타 | 출처 데이터셋(추적·라이선스용) |

---

## 2. 새 시나리오를 만들 때 — 최소 정보 7가지 (체크리스트)

새 시나리오 1개를 정의하려면 아래만 채우면 됩니다:

- [ ] **① 제목(title)** — 예: "소크라테스식 문답"
- [ ] **② 행동 한 문장(system_prompt)** — 모델이 할 행동. *가장 중요.* 예: "정답을 바로 주지 말고 역질문으로 유도하라"
- [ ] **③ 왜(rationale)** — 교육적 근거 한 줄. 예: "사고력 저하 방지"
- [ ] **④ 기대치(expectation)** — 평가 기준 한 줄. 예: "질문형 답변 성공률 90%"
- [ ] **⑤ 데이터 출처(dataset)** — ⭐ **실제 HF 데이터셋이 있는가?** (없으면 6장 참고)
- [ ] **⑥ 필드 매핑** — 데이터셋의 어떤 컬럼이 `input`/`output`이 되는가
- [ ] **⑦ 라이선스** — 학습에 써도 되는가 (5장 참고)

> ②번이 막연하면 시나리오가 아닙니다. "친절하게 답한다" ❌ → "틀린 단계를 짚어주되 정답은 말하지 않는다" ✅
> 행동이 **구체적·구별 가능**해야 모델이 시나리오별로 다르게 학습합니다.

---

## 3. 데이터셋이 있는지 찾는 법 (복붙 명령)

⑤·⑥·⑦은 **HuggingFace를 직접 조회**해서 채웁니다. 우리가 쓴 명령 그대로:

**(a) 후보 검색** — 다운로드순:
```bash
curl -s "https://huggingface.co/api/datasets?search=korean+summarization&sort=downloads&direction=-1&limit=8" \
  | python3 -c "import sys,json;[print(x['id'],x.get('downloads',0)) for x in json.load(sys.stdin)]"
```

**(b) 스키마·크기 확인** (컬럼명 = 필드 매핑 결정):
```bash
curl -s "https://datasets-server.huggingface.co/info?dataset=<DATASET_ID>" \
  | python3 -c "import sys,json;d=json.load(sys.stdin)['dataset_info'];[print(k,sum(s['num_examples'] for s in v['splits'].values()),list(v['features'])) for k,v in d.items()]"
```

**(c) 실제 샘플 확인** (한국어 맞는지, 품질):
```bash
curl -s "https://datasets-server.huggingface.co/first-rows?dataset=<DATASET_ID>&config=default&split=train" \
  | python3 -c "import sys,json;r=json.load(sys.stdin)['rows'][0]['row'];[print(k,':',str(v)[:120]) for k,v in r.items()]"
```

**(d) 라이선스 확인**:
```bash
curl -s "https://huggingface.co/api/datasets/<DATASET_ID>" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print([t for t in d.get('tags',[]) if t.startswith('license:')])"
```

> 더 빠르게: `python3 scripts/fetch_hf_datasets.py --list` 로 현재 등록된 16개 소스를 봅니다.

---

## 4. 좋은 데이터셋 고르는 기준

| 기준 | 좋음 | 피하기 |
|------|------|--------|
| **크기** | 수천~수십만 행 | 수십 행(소규모는 보조만) |
| **형식** | input→output 쌍이 명확 (생성형) | 단일 라벨(분류) → 행동 학습엔 약함 |
| **언어** | 한국어(`lang=ko`) | 영어는 행동만 학습, 한국어 출력엔 약함 |
| **라이선스** | **Apache-2.0 / MIT / CC-BY** | **CC-BY-ND**(파생금지) → 학습 불가, 평가만 |
| **품질** | 샘플이 실제 교육적·자연스러움 | GPT 흔적·노이즈 많음 |
| **인기** | ⬇다운로드·❤좋아요 높음 | 검증 안 된 것 |

라이선스 빠른 판단: **상업용이면 Apache/MIT 우선. NC(비상업)는 연구·교육만. ND(파생금지)는 학습 금지.**

---

## 5. 데이터셋이 없을 때 (우선순위)

행동은 좋은데 데이터가 없다면, **이 순서로** 해결합니다 (우리가 실제로 한 방식):

1. **다른 키워드/영어로 재검색** — 한국어 없으면 영어 데이터셋도 OK (모델이 다국어). 단 `lang=en` 표시.
2. **유사 데이터로 매핑** — 예: 소크라테스 디버깅용 영어 데이터를 시나리오 8에 연결.
3. **시나리오 자체를 교체** — 데이터가 풍부한 인접 주제로 변경 (예: "난이도 조절"→"지문 요약").
4. **few-shot 으로만 활용** — 학습엔 안 넣고 `seed_train.jsonl` 예시를 프롬프트에 참고.
5. ~~합성 데이터 생성~~ — 이 프로젝트는 **합성 미사용** 원칙.

---

## 6. 실제로 추가하는 2단계

### 6-1. `data/scenarios.json` 에 정의 추가
```json
{
  "id": 21,
  "stage": "2. 교과목 및 실전 스킬 특화",
  "title": "새 시나리오 제목",
  "system_prompt": "시나리오[21]: 모델이 할 행동을 명확히.",
  "data_design": "데이터셋의 어떤 컬럼 → input/output",
  "rationale": "왜 필요한지 한 줄",
  "expectation": "평가 기준 한 줄",
  "dataset": "owner/dataset-id"
}
```

### 6-2. `scripts/fetch_hf_datasets.py` 의 `SOURCES` 에 매핑 추가
```python
"my_key": {
    "hf_id": "owner/dataset-id",
    "config": "default",          # 멀티 config 면 지정, 아니면 생략
    "split": "train",
    "scenario_id": 21,
    "instruction": "시나리오[21]: ...",   # scenarios.json 의 system_prompt 와 동일하게
    "map": lambda r: (r.get("질문컬럼", ""), r.get("답변컬럼", "")),
    "license": "Apache-2.0",
    "rows": 12345,
    "lang": "ko",                 # 영어면 "en"
    "note": "시나리오 21(...) — 컬럼설명",
},
```
- **검증**: `python3 scripts/fetch_hf_datasets.py --list` 로 보이는지 확인 → `--only my_key --per-source 20` 으로 소량 테스트.
- 컬럼이 대화/중첩이면 헬퍼(`_from_conversations`, `_from_sharegpt`, `_from_klue_mrc` 등) 패턴을 참고해 매퍼 작성.

---

## 7. 한눈에 보는 결정 플로우

```
새 행동을 정의했다 (system_prompt 한 문장)
        │
        ▼
HF에 데이터 있나? ──(검색: 3장 명령)
   │ 있음                              │ 없음
   ▼                                   ▼
스키마/라이선스 OK? ──아니오──┐     영어로도 없나?
   │ 예                      │        │ 있음 → lang=en 으로 추가
   ▼                        │        │ 없음
input/output 매핑 가능?      │        ▼
   │ 예                      │     인접 주제로 시나리오 교체 가능?
   ▼                        │        │ 예 → 주제 변경 후 데이터 연결
scenarios.json + SOURCES 추가 │        │ 아니오 → few-shot 참고용만 (학습 제외)
   ▼                        │
--list / --only 로 검증 → 커밋 ◄──────┘
```

---

## 8. 복붙 템플릿 (최소 정보 메모용)

새 시나리오 제안 시 이 표만 채워서 주시면 제가 바로 등록합니다:

```
제목:
행동(한 문장):
왜:
기대치:
후보 데이터셋(HF id 또는 "찾아줘"):
input 컬럼 / output 컬럼:
언어(ko/en):
```

> 참고 문서: 기술 조절은 [`docs/LEARNING.md`](LEARNING.md), 현재 20시나리오는 [`data/scenarios.json`](../data/scenarios.json).
