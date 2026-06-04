#!/usr/bin/env python3
"""notebooks/unsloth_studio_ui.ipynb 생성 — 코랩에서 Unsloth Studio UI 띄우기 (방법 B)."""
import json
from pathlib import Path

def md(*lines): return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}
def code(*lines): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [l + "\n" for l in lines]}

cells = [
    md(
        "# 🖥️ Unsloth Studio UI on Colab (노코드)",
        "",
        "코랩에서 **Unsloth Studio 웹 UI**를 띄워 클릭만으로 모델을 받고 파인튜닝합니다.",
        "코드로 재현하는 학습은 `unsloth_edu_finetune.ipynb` 를 쓰세요. 이건 **빠른 체험용**입니다.",
        "",
        "> ⚠️ 먼저 **런타임 → 런타임 유형 변경 → T4 GPU** 선택!",
        "",
        "참고: 가짜 정보 주의 — Studio는 `pip install unsloth-studio` 가 **아니라** 공식 설치 스크립트로 깔고,",
        "`unsloth studio` 명령으로 실행합니다. (port 8888)",
    ),
    md("## 0. GPU 확인"),
    code("!nvidia-smi"),
    md(
        "## 1. Unsloth Studio 설치",
        "공식 설치 스크립트 사용 (MacOS/Linux/WSL/Colab 공통). 1~2분 소요.",
    ),
    code(
        "!curl -fsSL https://unsloth.ai/install.sh | sh",
    ),
    md(
        "## 2. Studio 서버 실행 (백그라운드, port 8888)",
        "서버를 백그라운드로 띄우고 기동될 때까지 잠시 기다립니다.",
    ),
    code(
        "import subprocess, time, os",
        "",
        "# 로그를 파일로 남기며 백그라운드 실행",
        "logf = open('studio.log', 'w')",
        "proc = subprocess.Popen(",
        '    ["unsloth", "studio", "-H", "0.0.0.0", "-p", "8888"],',
        "    stdout=logf, stderr=subprocess.STDOUT,",
        ")",
        "print('Studio 서버 기동 중... (PID', proc.pid, ')')",
        "time.sleep(25)   # 서버 준비 대기",
        "print('--- studio.log (마지막 20줄) ---')",
        "!tail -n 20 studio.log",
    ),
    md(
        "## 3. UI 열기 — 포트 8888 노출",
        "코랩 내장 기능으로 포트를 노출합니다. **외부 터널(ngrok/localtunnel) 불필요.**",
        "아래 셀 실행 후 나오는 링크/창에서 Studio UI가 열립니다.",
    ),
    code(
        "from google.colab.output import serve_kernel_port_as_window, serve_kernel_port_as_iframe",
        "",
        "# (A) 새 창으로 열기 (권장)",
        "serve_kernel_port_as_window(8888)",
        "",
        "# (B) 노트북 안에 iframe으로 보고 싶으면 위 줄 대신 아래 사용:",
        "# serve_kernel_port_as_iframe(8888, height=800)",
    ),
    md(
        "## 4. UI에서 모델 다운로드 & 학습",
        "",
        "열린 Studio 화면에서:",
        "1. **모델 선택칸**에 HuggingFace 모델명 입력 → 자동 다운로드",
        "   - 추천(초경량): `unsloth/Qwen2.5-1.5B-Instruct`, `unsloth/Qwen2.5-0.5B-Instruct`, `unsloth/Llama-3.2-1B-Instruct`",
        "2. **데이터** 업로드 (CSV/JSONL) — 우리 레포 데이터를 쓰려면 아래 셀로 먼저 받으세요.",
        "3. LoRA/하이퍼파라미터 조정 후 **Train** → **Model Arena**로 비교 → **Export**(GGUF 등).",
        "",
        "각 설정의 의미·조절법은 레포의 `docs/LEARNING.md` 치트시트를 참고하세요.",
    ),
    code(
        "# (선택) 우리 레포의 20시나리오 학습 데이터 받아서 Studio에 업로드용으로 준비",
        "REPO = 'https://github.com/xide-projext/edu-llm-colab-unsloth.git'",
        "import os",
        "if not os.path.exists('edu-llm-colab-unsloth'):",
        "    !git clone -q $REPO",
        "%cd edu-llm-colab-unsloth",
        "!pip install -q datasets",
        "!python scripts/fetch_hf_datasets.py --per-source 2000 --val-ratio 0.05",
        "print('생성됨: data/hf_train.jsonl → Studio UI에서 이 파일을 업로드하세요')",
    ),
    md(
        "---",
        "### 문제 해결",
        "- **Open 링크가 에러**: 쿠키/애드블록 때문. 3번 셀을 다시 실행하거나 iframe(B) 방식 사용.",
        "- **서버가 안 뜸**: `!tail -n 50 studio.log` 로 로그 확인. 설치(1번) 재실행.",
        "- **OOM**: 더 작은 모델(0.5B) 선택, batch/seq 축소 (LEARNING.md 치트시트).",
        "- 코드 기반 재현 학습이 필요하면 `unsloth_edu_finetune.ipynb` 사용.",
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
