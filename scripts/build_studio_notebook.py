#!/usr/bin/env python3
"""notebooks/unsloth_studio_ui.ipynb 생성 — 코랩에서 Unsloth Studio UI 띄우기.

공식 Colab 방식(unslothai/unsloth/studio/Unsloth_Studio_Colab.ipynb)을 그대로 따른다:
  - curl install.sh (로컬용) 가 아니라, 레포 clone + studio/setup.sh --local
  - 포트 노출은 studio backend 의 colab.start() 가 자동 처리
"""
import json
from pathlib import Path

def md(*lines): return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}
def code(*lines): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [l + "\n" for l in lines]}

cells = [
    md(
        "# 🖥️ Unsloth Studio UI on Colab (노코드)",
        "",
        "코랩에서 **Unsloth Studio 웹 UI**를 띄워 클릭만으로 모델을 받고 파인튜닝합니다.",
        "**공식 Colab 방식** 그대로입니다 (unslothai/unsloth/studio).",
        "",
        "> ⚠️ 먼저 **런타임 → 런타임 유형 변경 → T4 GPU** 선택 후 **런타임 → 모두 실행**!",
        "",
        "참고: `pip install unsloth-studio` 나 `curl install.sh` 는 코랩용이 **아닙니다**.",
        "코랩에서는 아래처럼 레포를 clone하고 `setup.sh` → `colab.start()` 로 띄웁니다.",
    ),
    md("## 0. GPU 확인 (T4 인지 확인)"),
    code("!nvidia-smi"),
    md(
        "## 1. Setup — Unsloth 레포 clone & 설치",
        "공식 setup 스크립트를 실행합니다. (수 분 소요, 의존성 설치)",
    ),
    code(
        "!git clone --depth 1 --branch main https://github.com/unslothai/unsloth.git",
        "%cd /content/unsloth",
        "!chmod +x studio/setup.sh && ./studio/setup.sh --local",
    ),
    md(
        "## 2. Studio 시작",
        "`start()` 가 서버 실행 + 포트 노출(Open 버튼)을 자동으로 처리합니다.",
        "실행 후 나오는 **\"Open Unsloth Studio\"** 링크/박스를 클릭하세요.",
    ),
    code(
        "import sys",
        'sys.path.insert(0, "/content/unsloth/studio/backend")',
        "from colab import start",
        "start()",
    ),
    md(
        "## 3. UI에서 모델 다운로드 & 학습",
        "",
        "열린 Studio 화면에서:",
        "1. **모델 선택칸**에 HuggingFace 모델명 입력 → 자동 다운로드",
        "   - 추천(초경량): `unsloth/Qwen2.5-1.5B-Instruct`, `unsloth/Qwen2.5-0.5B-Instruct`, `unsloth/Llama-3.2-1B-Instruct`",
        "2. **데이터** 업로드(CSV/JSONL). 우리 20시나리오 데이터를 쓰려면 아래 4번 셀로 먼저 생성.",
        "3. LoRA/하이퍼파라미터 조정 → **Train** → **Model Arena** 비교 → **Export**(GGUF 등).",
        "",
        "각 설정의 의미·조절법은 레포의 `docs/LEARNING.md` 치트시트 참고.",
    ),
    md(
        "## 4. (선택) 우리 레포 20시나리오 데이터 생성 → UI에 업로드",
        "Studio UI의 데이터 업로드칸에 넣을 `hf_train.jsonl` 을 만듭니다.",
    ),
    code(
        "%cd /content",
        "!git clone -q https://github.com/xide-projext/edu-llm-colab-unsloth.git",
        "%cd /content/edu-llm-colab-unsloth",
        "!pip install -q datasets",
        "!python scripts/fetch_hf_datasets.py --per-source 2000 --val-ratio 0.05",
        'print("생성됨: /content/edu-llm-colab-unsloth/data/hf_train.jsonl → 좌측 파일탭에서 다운로드 후 Studio UI에 업로드")',
    ),
    md(
        "---",
        "### 문제 해결",
        "- **Open 링크가 에러/빈 화면**: 쿠키·애드블록 때문. 2번 셀을 다시 실행하거나, 출력 박스 아래로 스크롤하면 UI가 직접 보입니다. (알려진 버그 unslothai/unsloth#4516)",
        "- **setup.sh 실패**: 1번 셀 로그 확인 후 재실행. 런타임이 **T4 GPU** 인지 다시 확인.",
        "- **OOM**: UI에서 더 작은 모델(0.5B), batch/seq 축소 (LEARNING.md 치트시트).",
        "- **코드로 재현 학습**: UI 대신 `unsloth_edu_finetune.ipynb` 사용 (권장 — 재현·버전관리).",
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = Path("notebooks/unsloth_studio_ui.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"생성 완료: {out}  (셀 {len(cells)}개)")
