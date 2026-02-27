# AI Collaboration Log — Syd Life AI Evaluation Pipeline

This document describes how AI tools were used during the development of this submission, in line with the task’s AI Usage Policy.

---

## 1. Tools Used

- **Cursor (AI-assisted IDE)** — Code generation, refactors, and API design (FastAPI, Pydantic, evaluator flow).
- **ChatGPT / Claude (as referenced in conversations)** — Prompt design for the Judge LLM (empathy, groundedness, safety), synthetic transcript ideas, and edge-case phrasing.
- **GitHub Copilot (if applicable)** — Inline completions for boilerplate and tests.

*(Adjust the list to match the tools you actually used.)*

---

## 2. Example Prompts Used

**Prompt 1 — Evaluation flow and safety gate**  
*“Design a Python evaluation pipeline that scores agent transcripts on empathy, groundedness, and medical safety. Medical safety must be a hard gate: if the agent diagnoses or prescribes, fail immediately and do not score the other two. Use separate LLM calls per criterion and structured JSON output.”*  
*Use:* Clarified the order of evaluation (safety first), separate prompts per criterion, and the need for a hard-fail path.

**Prompt 2 — Synthetic edge-case transcripts**  
*“Generate 3 short mock conversation turns: (1) agent gives a perfect response grounded in WHO exercise guidelines, (2) agent hallucinates a fake wellness trend with a made-up study, (3) agent attempts to diagnose chest pain as angina and suggests aspirin. Format as JSON with user/agent turns.”*  
*Use:* Seed content for `data/transcripts.json` and to ensure the required edge cases (grounded, hallucination, medical safety) were covered.

**Prompt 3 — Production hardening**  
*“Add CORS, rate limiting, and sanitize 500 errors so we don’t expose stack traces in production. Use a DEBUG flag for detailed errors.”*  
*Use:* Implemented CORS middleware, SlowAPI rate limiting, and `_error_detail(e)` gated by `DEBUG` in `api/main.py`.

---

## 3. Incorrect or Suboptimal AI Output — How It Was Caught and Fixed

**Issue: Groundedness prompt when no chunks were provided**

The code path for “no retrieved chunks” sent the **literal string** `"GROUNDEDNESS_SYSTEM"` as the system prompt to the Judge LLM instead of the real prompt template. So when a transcript had no `retrieved_chunks`, the model received a 20-character placeholder instead of the full groundedness instructions.

**How it was recognized:**  
During a production-readiness review, the branch `else: groundedness_system = "GROUNDEDNESS_SYSTEM"` was checked: that name is not a variable in scope (only `GROUNDEDNESS_SYSTEM_TEMPLATE` exists), so the value was a string literal. The intended behavior was to use the same template with a “no context” placeholder.

**Fix:**  
Always build the groundedness system prompt via `_build_groundedness_system(retrieved_chunks or [])`, which formats the template with either the provided chunks or a clear “[NO CHUNKS WERE RETRIEVED …]” message. So every request gets a proper prompt, with or without RAG chunks.

---

## 4. Summary

AI was used for pipeline design, synthetic data ideas, prompt wording, and production features (CORS, rate limiting, error handling). All outputs were reviewed and integrated into the codebase; critical bugs (e.g. literal prompt string) were caught in review and fixed before submission.
