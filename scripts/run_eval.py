#!/usr/bin/env python3
"""
CLI runner for the Syd Life AI Evaluation Pipeline.

Usage examples:
  # Evaluate all transcripts
  python scripts/run_eval.py

  # Evaluate a single transcript by ID
  python scripts/run_eval.py --id T004

  # Evaluate all and output results to a JSON file
  python scripts/run_eval.py --output results.json

  # Use a specific model
  python scripts/run_eval.py --model gpt-4o-mini
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from data.loaders import load_transcripts
from evaluator import (
    JudgeLLM,
    check_expected,
    log_error,
    log_evaluation_result,
    log_pipeline_summary,
)


def print_banner():
    print()
    print("=" * 60)
    print("  Syd Life AI — LLM Evaluation Pipeline")
    print("=" * 60)
    print()


def run(transcript_id: str | None = None, model: str | None = None, output: str | None = None):
    print_banner()

    transcripts = load_transcripts()

    if transcript_id:
        transcripts = [t for t in transcripts if t["transcript_id"] == transcript_id]
        if not transcripts:
            print(f"ERROR: Transcript ID '{transcript_id}' not found.")
            sys.exit(1)

    judge = JudgeLLM(model=model)
    results = []
    validation_matches = 0
    validation_total = 0
    validation_mismatches: list[tuple[str, list[str]]] = []

    for transcript in transcripts:
        try:
            result = judge.evaluate(transcript)
            log_evaluation_result(result)

            # Validate against expected_verdict / expected_flags when present
            if transcript.get("expected_verdict") is not None or transcript.get("expected_flags"):
                validation_total += 1
                matched, messages = check_expected(result, transcript)
                if matched:
                    validation_matches += 1
                else:
                    validation_mismatches.append((transcript["transcript_id"], messages))

            results.append(result)
        except Exception as e:
            log_error(f"CLI:{transcript['transcript_id']}", e)
            print(f"  ⚠ Failed to evaluate {transcript['transcript_id']}: {e}")

    if len(results) > 1:
        log_pipeline_summary(results)

    # Report validation against expected data
    if validation_total > 0:
        print()
        print("Expected validation (vs transcript expected_verdict / expected_flags):")
        print(f"  Matched: {validation_matches}/{validation_total}")
        if validation_mismatches:
            for tid, msgs in validation_mismatches:
                print(f"  ❌ [{tid}] {'; '.join(msgs)}")

    if output:
        out_path = Path(output)
        out_path.write_text(
            json.dumps([r.model_dump() for r in results], indent=2),
            encoding="utf-8",
        )
        print(f"\nResults saved to: {out_path.resolve()}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Syd Life AI — LLM Evaluation Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--id",
        dest="transcript_id",
        metavar="TRANSCRIPT_ID",
        help="Evaluate a single transcript by ID (e.g., T004). Omit to evaluate all.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the judge LLM model (default: value of JUDGE_MODEL env var, or gpt-4o).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE.json",
        help="Save evaluation results to a JSON file.",
    )

    args = parser.parse_args()
    run(transcript_id=args.transcript_id, model=args.model, output=args.output)


if __name__ == "__main__":
    main()
