---
title: GAIA Benchmark Solver
emoji: 🏅
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.40.0"
app_file: app.py
pinned: false
---

# 🏅 GAIA Benchmark Solver

**Hugging Face AI Agents Course — Unit 4 Final Assignment**

Multimodal AI agent that solves [GAIA Level 1](https://huggingface.co/datasets/gaia-benchmark/GAIA) benchmark questions. Achieved **70% (14/20)** on the evaluation set.

[![Hugging Face Space](https://img.shields.io/badge/🤗-Live%20Space-yellow)](https://huggingface.co/spaces/anderescaray/final_assignment_ai_agents_course)

---

## Architecture

    app.py                  ← Gradio UI + benchmark runner
    agent/
      model.py              ← LLM backend (Claude Haiku via LiteLLM)
      solver.py             ← GAIASolver: CodeAgent orchestration loop
      prompts.py            ← Exact-match answer formatting rules
    tools/
      download.py           ← GAIA attached file downloader
      audio.py              ← Whisper-based audio transcription
      youtube.py            ← YouTube transcript extraction

**Model:** `claude-haiku-4-5-20251001` (Anthropic) via LiteLLM  
**Agent:** smolagents `CodeAgent` — uses Python as its reasoning substrate  
**Tools:** DuckDuckGo search, webpage visits, YouTube transcripts, Whisper audio, file downloads

---

## Results

| Metric | Value |
|--------|-------|
| Score | **70% (14/20)** |
| Level | GAIA Level 1 |
| Pass threshold | 30% |

---

## Setup

### 1. Clone & install

    git clone https://github.com/anderescaray/gaia-agent
    cd gaia-agent
    pip install -r requirements.txt

### 2. Configure secrets

For HF Spaces: add `ANTHROPIC_API_KEY` and `HF_TOKEN` as **Space secrets**.

### 3. Run

    python app.py

---

## Key design decisions

| Problem | Solution |
|---------|----------|
| GAIA tasks often have attached files (CSV, audio, images) | `DownloadTaskFileTool` fetches from scoring API by `task_id` |
| Agent state leaks between questions | `reset=True` on every `agent.run()` call |
| LLM adds verbose preamble, breaking exact-match scoring | Conservative `_clean()` strips only unambiguous prefixes |
| HF Inference endpoints unreliable for large models | Switched to Anthropic API via LiteLLM |

---