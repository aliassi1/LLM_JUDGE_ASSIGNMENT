# Syd Life AI — Component Architecture

This document describes the solution architecture: how the codebase is split into components, what each component does, and how they interact.

---

## High-Level Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ENTRY POINTS                                       │
│  api/main.py (FastAPI)          scripts/run_eval.py (CLI)                   │
└────────────────────────────┬────────────────────────────┬──────────────────┘
                             │                             │
                             ▼                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                           │
│  data/loaders.py  — load_transcripts(), load_knowledge_base()                │
└─────────────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EVALUATOR PACKAGE                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ core        │  │ prompts     │  │ validation  │  │ parsers              │ │
│  │ (domain)    │  │ (KB+prompts)│  │ (transcript)│  │ (judge response)    │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                │                     │             │
│         └────────────────┴────────────────┴─────────────────────┘             │
│                                      │                                        │
│                                      ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ judge.py  — JudgeLLM (orchestrates: _call LLM, validate, parse, verdict)│ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                        │
│  ┌──────────────────────────────────┴──────────────────────────────────────┐│
│  │ logger.py  — log_evaluation_result(), log_pipeline_summary(), log_error() ││
│  └──────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Core (evaluator/core)

**Purpose:** Domain model and business rules. No I/O, no external services.

**Contents:**
- **criteria.py** — Pydantic models and verdict logic:
  - `Verdict` (PASS, FAIL, HARD_FAIL), `Flag` (HALLUCINATION, MEDICAL_SAFETY_VIOLATION, LOW_EMPATHY)
  - `EmpathyScore`, `GroundednessScore`, `MedicalSafetyScore`, `EvaluationResult`
  - Thresholds: `EMPATHY_PASS_THRESHOLD` (0.6), `GROUNDEDNESS_PASS_THRESHOLD` (0.7)
  - `compute_verdict(empathy, groundedness)` → (Verdict, list[Flag])

**Used by:** judge, parsers, API (via evaluator), tests.

**Why separate:** Keeps scoring rules and data shapes in one place. Changing thresholds or adding a new criterion starts here.

---

## 2. Prompts (evaluator/prompts)

**Purpose:** All text sent to the judge LLM: system prompts and knowledge base.

**Contents:**
- **prompts.py** — Loads `data/knowledge_base.json`, builds `KB_SUMMARY`, and defines:
  - `EMPATHY_SYSTEM` — instructions and JSON schema for empathy scoring
  - `GROUNDEDNESS_SYSTEM` — KB + instructions and schema for groundedness
  - `SAFETY_SYSTEM` — instructions and schema for medical safety
  - `KNOWLEDGE_BASE`, `KB_SUMMARY` (used in groundedness prompt)

**Used by:** judge.py only.

**Why separate:** Changing prompt wording or KB format is done in one module without touching orchestration or parsing.

---

## 3. Validation (evaluator/validation)

**Purpose:** Input validation before evaluation.

**Contents:**
- **transcript.py** — `validate_transcript(transcript: dict) -> None`:
  - Ensures `transcript_id` and `turns` exist, `turns` is a non-empty list
  - Each turn has `role` and `content`
  - At least one turn has `role="agent"`
  - Raises `ValueError` with a clear message if invalid

**Used by:** judge.py (at the start of `evaluate()`). Can be reused by API or CLI if they need to validate before calling the judge.

**Why separate:** Single place for transcript contract. Easier to test and to extend (e.g. max turns, allowed roles).

---

## 4. Parsers (evaluator/parsers)

**Purpose:** Turn raw judge LLM JSON into domain score models and raise a single error type on failure.

**Contents:**
- **response_parsers.py**:
  - `JudgeParseError` — exception with optional `criterion` and `raw_preview`
  - `parse_safety(raw, criterion)` → `MedicalSafetyScore`
  - `parse_empathy(raw, criterion)` → `EmpathyScore`
  - `parse_groundedness(raw, criterion)` → `GroundednessScore`

Each parser checks required keys and types (e.g. `safe` is bool, `score` in [0, 1]) and raises `JudgeParseError` with context if invalid.

**Used by:** judge.py after each `_call()`.

**Why separate:** Isolates “string → structured output” logic. Easy to unit-test with fake JSON; judge stays thin.

---

## 5. Judge (evaluator/judge.py)

**Purpose:** Orchestrate one full evaluation: validate input, call LLM three times (safety → empathy → groundedness), parse responses, compute verdict, return `EvaluationResult`.

**Responsibilities:**
- **JudgeLLM**:
  - `_call(system, user_message, criterion)` — call OpenAI with JSON mode; on empty or invalid JSON raise `JudgeParseError`
  - `_format_transcript(turns)` — format conversation for the prompt
  - `evaluate(transcript)` — validate → safety call → if safe, empathy + groundedness calls → parse all → `compute_verdict` → build and return `EvaluationResult` (or early HARD_FAIL on safety violation)

**Uses:** core (models, compute_verdict), prompts (system prompts), validation (validate_transcript), parsers (parse_safety, parse_empathy, parse_groundedness).

**Why a single module:** One clear “evaluator” entry point. Flow is linear and easy to follow.

---

## 6. Logger (evaluator/logger.py)

**Purpose:** Structured audit trail and human-readable console output.

**Contents:**
- `AUDIT_LOG_PATH` — path to `logs/evaluation_audit.jsonl`
- `log_evaluation_result(result)` — append one JSON line per evaluation + short console summary
- `log_pipeline_summary(results)` — append PIPELINE_SUMMARY line + console summary for a batch
- `log_error(context, error)` — append ERROR line + console error

**Used by:** API and CLI after each evaluation or batch.

**Why separate:** Centralizes log format and file location. Swapping to another backend (e.g. external logging service) happens here.

---

## 7. Data loaders (data/loaders.py)

**Purpose:** Load transcripts and knowledge base from JSON files. Single place for paths and file I/O.

**Contents:**
- `TRANSCRIPTS_PATH`, `KNOWLEDGE_BASE_PATH` — default paths under `data/`
- `load_transcripts(path=None)` → list[dict]
- `load_knowledge_base(path=None)` → list[dict]

**Used by:** API (transcripts at startup), CLI (transcripts per run). Knowledge base is loaded inside `evaluator/prompts` (at import time); loaders could later be used there too for consistency.

**Why separate:** API and CLI no longer hard-code paths. Easier to test with different files or to add another source (e.g. DB) behind the same interface.

---

## 8. API (api/main.py)

**Purpose:** HTTP interface: health, list transcripts, evaluate by ID, evaluate custom body, evaluate-all, audit log.

**Responsibilities:**
- Load transcripts once via `data.loaders.load_transcripts()`
- Build FastAPI app and request/response models (`TranscriptTurn`, `EvaluateRequest`, `BatchEvaluateResponse` with `attempted`, `errors`)
- Routes call `JudgeLLM().evaluate()`, then logger; on exception log and return 500 or collect errors for evaluate-all

**Uses:** data.loaders, evaluator (JudgeLLM, EvaluationResult, log_*).

**Why separate:** Clear boundary between “transport” (HTTP) and “evaluation” (evaluator package).

---

## 9. CLI (scripts/run_eval.py)

**Purpose:** Command-line runner: evaluate all or one transcript, optional output file and model override.

**Responsibilities:**
- Load transcripts via `data.loaders.load_transcripts()`
- Parse args (--id, --model, --output)
- Loop over transcripts, call `JudgeLLM().evaluate()`, log each result and errors, optionally write JSON and pipeline summary

**Uses:** data.loaders, evaluator (JudgeLLM, log_*).

**Why separate:** Same evaluation logic as API, different entry point and UX.

---

## 10. Backward compatibility (evaluator/criteria.py)

**Purpose:** Keep existing imports working.

**Contents:** Re-exports from `evaluator.core` (Verdict, Flag, score models, EvaluationResult, compute_verdict, thresholds).

**Used by:** Tests and any code that still does `from evaluator.criteria import ...`. New code can use `from evaluator.core import ...` or `from evaluator import ...`.

---

## Data and Config (unchanged)

- **data/transcripts.json** — default transcript dataset
- **data/knowledge_base.json** — guidelines used in groundedness prompt (loaded by prompts module)
- **logs/evaluation_audit.jsonl** — append-only audit log (used by logger)
- **.env** — OPENAI_API_KEY, JUDGE_MODEL (used by JudgeLLM)

---

## Summary Table

| Component        | Role                          | Key exports / entry points           |
|-----------------|-------------------------------|--------------------------------------|
| **evaluator/core** | Domain models and verdict     | Verdict, Flag, *Score, EvaluationResult, compute_verdict |
| **evaluator/prompts** | KB + system prompts        | EMPATHY_SYSTEM, GROUNDEDNESS_SYSTEM, SAFETY_SYSTEM, KB_SUMMARY |
| **evaluator/validation** | Transcript validation   | validate_transcript                   |
| **evaluator/parsers** | Judge response parsing   | JudgeParseError, parse_safety, parse_empathy, parse_groundedness |
| **evaluator/judge** | Evaluation orchestration  | JudgeLLM                              |
| **evaluator/logger** | Audit + console logging   | log_evaluation_result, log_pipeline_summary, log_error |
| **data/loaders** | Load transcripts/KB from disk | load_transcripts, load_knowledge_base |
| **api/main**    | HTTP API                     | app, routes                           |
| **scripts/run_eval** | CLI                        | main()                                |

All 19 tests pass after this refactor. The public API (`from evaluator import JudgeLLM, EvaluationResult, ...`) is unchanged.
