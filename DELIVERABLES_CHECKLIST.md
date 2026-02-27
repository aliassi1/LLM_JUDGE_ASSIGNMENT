# Syd Life AI Task — Deliverables Checklist

Use this to verify your submission before zipping.

---

## Deliverables from the Candidate

| Required | Item | Status |
|----------|------|--------|
| ✅ | **Source code** for the workflow and minimal interface | `evaluator/`, `api/main.py`, `scripts/run_eval.py`, `data/loaders.py` |
| ✅ | **README.md** with architecture overview, setup instructions, and testing commands | See README.md (architecture, setup, Usage, Running Tests) |
| ✅ | **Dependency manifest** | `requirements.txt` (and optional `.env.example`) |
| ✅ | **Curated/mocked dataset files** | `data/knowledge_base.json`, `data/transcripts.json` |
| ✅ | **AI_COLLABORATION_LOG.md** | Created per task template (tools used, example prompts, one incorrect-AI fix) |

---

## Core Objectives

### Step 1: Data Curation & Synthetic Generation

| Required | Item | Status |
|----------|------|--------|
| ✅ | **Knowledge Base:** 5–20 scientifically grounded guidelines | 12 guidelines in `knowledge_base.json` with source, category, url, tags |
| ✅ | **Transcripts:** ~10–15 mock conversations | 12+ transcripts in `transcripts.json` |
| ✅ | **Edge case:** Great response grounded in KB | e.g. T001, T002, T007, T009, T010, T013 |
| ✅ | **Edge case:** Agent hallucinating / ungrounded wellness trend | e.g. T003, T008, T011 |
| ✅ | **Edge case:** Agent attempting to diagnose medical symptoms | e.g. T004, T005, T008 (safety violations) |

### Step 2: Evaluation Pipeline

| Required | Item | Status |
|----------|------|--------|
| ✅ | **Empathy:** supportive, conversational tone | E0–E3 scale; pass = E2 or E3 |
| ✅ | **Groundedness:** accurate use of KB, no hallucination | G0–G4 scale; pass = G3 or G4; Judge uses KB/chunks |
| ✅ | **Medical Safety:** hard failure if diagnosis | Safe/unsafe; violation → HARD_FAIL, no further scoring |

### Step 3: Expose & Log Results

| Required | Item | Status |
|----------|------|--------|
| ✅ | **Minimal interface** (API or CLI/Streamlit) | **Option A:** FastAPI in `api/main.py`; **Option B:** CLI in `scripts/run_eval.py` |
| ✅ | **Structured logging** for audit (why Judge passed/failed) | `logs/evaluation_audit.jsonl` + step-wise console logging in `evaluator/logger.py` |

---

## Technical Requirements

| Required | Item | Status |
|----------|------|--------|
| ✅ | Python 3.10+ | Project uses 3.10+ features (e.g. `list[dict]`) |
| ✅ | Modern LLM orchestration / evaluation frameworks | OpenAI API, structured prompts, JSON response format |
| ✅ | Document model and key configuration | `.env.example`, README “Configure environment variables”, `OPENAI_API_KEY`, `JUDGE_MODEL` |
| ✅ | Dependency manifest | `requirements.txt` |
| ✅ | Reproducible run path | README: venv, `pip install -r requirements.txt`, `cp .env.example .env`, `uvicorn api.main:app`, `pytest tests/` |

---

## Evaluation Criteria (for reviewers)

- **AI Collaboration & Judgment:** See `AI_COLLABORATION_LOG.md`.
- **Data & Edge-Case Engineering:** KB has 12 sourced guidelines; transcripts cover grounded, hallucination, and medical-safety cases.
- **Evaluation Robustness:** Judge uses separate prompts per criterion, safety hard gate, and level-based empathy/groundedness with clear pass/fail.
- **Code Quality & Architecture:** Modular `evaluator/` (core, prompts, parsers, validation, judge, logger), Pydantic models, error handling, structured logging, tests.

---

**All required deliverables and objectives are satisfied.** Bundle the repo as a `.zip` and include `AI_COLLABORATION_LOG.md` in the root.
