"""
Outbound-message evaluator aligned with Tinker SamplingClientEvaluator.

Flow:
- Dataset is loaded from candidates_formatted.jsonl (JSONL with messages format).
- Each record contains a candidate profile and job description in the user message.
- Sampling model generates a LinkedIn-style DM.
- A grader model (OpenAI gpt-4.1 by default) scores the DM using rubric.json.
- Grader is calibrated to be VERY HARSH — most messages score 8-16.

The rubric and candidate data are designed to be challenging:
- ~55% of candidates are "edge cases" with subtle mismatches, red flags, or sparse profiles
- Edge cases include: domain mismatches, outdated experience, job hoppers, sparse profiles
- Rubric heavily penalizes surface-level personalization and template patterns
- High scores (20+) require genuine insight, not just restating profile details
- Even "perfect fit" candidates require nuanced messaging to score well

Expected score distribution:
- 0-5: Poor/template messages
- 6-11: Weak, surface-level matching
- 12-19: Adequate, where most decent messages land
- 20-27: Strong, uncommon
- 28-35: Exceptional, extremely rare

Usage (example):
  python -m tinker_cookbook.eval.outreach_evaluator \
    --dataset data/candidates_formatted.jsonl \
    --rubric data/rubric.json \
    --limit 10 \
    --creator-model Qwen/Qwen3-4B-Instruct-2507 \
    --renderer qwen3 \
    --verbose

Env:
  OPENAI_API_KEY must be set for the grader call.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import tinker
from openai import OpenAI
from pydantic import BaseModel, ConfigDict
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.tokenizer_utils import get_tokenizer


DEFAULT_GRADER_MODEL = "gpt-4.1"


class SectionScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str
    score: float
    comments: str


class Penalty(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str
    score: float


class GradedSections(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_scores: List[SectionScore]  # List instead of Dict for OpenAI structured outputs
    penalties: List[Penalty]  # Required for OpenAI structured outputs; empty list if no penalties


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file (one JSON object per line)."""
    items = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_dataset_from_jsonl(data: List[Dict[str, Any]], limit: int | None = None) -> List[Dict[str, Any]]:
    """
    Build dataset from candidates_formatted.jsonl format.
    
    Each item in data should have:
      {"messages": [{"role": "user", "content": "...candidate profile + job description..."}]}
    
    We extract the user content and wrap it with the recruiter instruction prompt.
    """
    items: List[Dict[str, Any]] = []
    for idx, record in enumerate(data):
        if limit and idx >= limit:
            break
        
        # Extract the candidate profile + job description from the messages
        messages = record.get("messages", [])
        if not messages:
            continue
        
        # The user message contains the candidate profile and job description
        user_content = messages[0].get("content", "") if messages else ""
        
        prompt = (
            "You are a recruiter crafting a concise, respectful LinkedIn DM to a candidate about a role.\n"
            "Use the candidate profile and the job description below. Keep it <130 words, clear CTA, no fluff.\n\n"
            f"{user_content}\n"
        )
        items.append({"prompt": prompt})
    return items


# ANSI color codes
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _print_result(
    idx: int,
    total: int,
    dm: str,
    score: float,
    justification: str,
    section_scores: List[SectionScore],
    penalties: List[Penalty],
) -> None:
    """Pretty-print grading results with colors."""
    print()
    print(f"{_CYAN}{_BOLD}{'─' * 60}{_RESET}")
    print(f"{_CYAN}{_BOLD}  Example {idx}/{total}{_RESET}")
    print(f"{_CYAN}{_BOLD}{'─' * 60}{_RESET}")

    # Draft message
    print(f"\n{_BOLD}📝 Draft Message:{_RESET}")
    print(f"{_DIM}{'─' * 40}{_RESET}")
    # Indent and wrap the message for readability
    for line in dm.split("\n"):
        print(f"  {line}")
    print(f"{_DIM}{'─' * 40}{_RESET}")

    # Section scores
    if section_scores:
        print(f"\n{_BOLD}📊 Section Scores:{_RESET}")
        for sec in section_scores:
            # Color based on score (assuming max ~10 per section)
            if sec.score >= 8:
                color = _GREEN
            elif sec.score >= 5:
                color = _YELLOW
            else:
                color = _RED
            print(f"  {_BOLD}{sec.section_id:20}{_RESET} {color}{sec.score:>5.1f}{_RESET}")
            if sec.comments:
                print(f"    {_DIM}{sec.comments}{_RESET}")

    # Penalties
    if penalties:
        print(f"\n{_BOLD}⚠️  Penalties:{_RESET}")
        for pen in penalties:
            print(f"  {_RED}{pen.score:>+5.1f}{_RESET}  {pen.reason}")

    # Total
    print()
    if score >= 80:
        score_color = _GREEN
    elif score >= 50:
        score_color = _YELLOW
    else:
        score_color = _RED
    print(f"{_BOLD}{'─' * 30}{_RESET}")
    print(f"{_BOLD}  TOTAL SCORE: {score_color}{score:.1f}{_RESET}")
    print(f"{_BOLD}{'─' * 30}{_RESET}")
    print()


class OutboundEvaluator(SamplingClientEvaluator):
    def __init__(
        self,
        dataset: List[Dict[str, Any]],
        rubric: Dict[str, Any],
        renderer_name: str,
        model_name: str,
        grader_model: str = DEFAULT_GRADER_MODEL,
        grader_timeout: float = 30.0,
        max_tokens: int = 200,
        temperature: float = 0.7,
        verbose: bool = False,
    ):
        self.dataset = dataset
        self.rubric = rubric
        self.grader_model = grader_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose

        tokenizer = get_tokenizer(model_name)
        self.renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)
        self.client = OpenAI(timeout=grader_timeout)

    async def __call__(self, sampling_client: tinker.SamplingClient) -> Dict[str, float]:
        sampling_params = types.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=1.0,
            stop=self.renderer.get_stop_sequences(),
        )

        scores: List[float] = []
        for idx, datum in enumerate(self.dataset, start=1):
            model_input: types.ModelInput = self.renderer.build_generation_prompt(
                [renderers.Message(role="user", content=datum["prompt"])]
            )
            # Progress logging to help debug long waits.
            print(f"[creator] sampling example {idx}/{len(self.dataset)} ...", flush=True)
            resp: types.SampleResponse = await sampling_client.sample_async(
                prompt=model_input, num_samples=1, sampling_params=sampling_params
            )
            tokens: List[int] = resp.sequences[0].tokens
            message: renderers.Message = self.renderer.parse_response(tokens)[0]
            dm_text = message["content"]
            dm_clean = (dm_text or "").strip()

            # If the draft is empty/garbled, short-circuit to score 0 without grading.
            if not dm_clean or dm_clean.startswith("<|im_start|>") or len(dm_clean) < 10:
                score, justification = 0.0, "Empty or invalid draft; skipped grading."
                if self.verbose:
                    _print_result(idx, len(self.dataset), dm_text or "", score, justification, [], [])
                scores.append(score)
                continue

            print(f"\033[90m[grader] scoring example {idx}/{len(self.dataset)} ...\033[0m", flush=True)
            score, justification, section_scores, penalties = self.grade(dm_clean, datum["prompt"])
            if self.verbose:
                _print_result(idx, len(self.dataset), dm_clean, score, justification, section_scores, penalties)
            scores.append(score)

        avg = sum(scores) / len(scores) if scores else 0.0
        return {"outreach_score": avg}

    def grade(self, dm: str, prompt: str) -> tuple[float, str, List[SectionScore], List[Penalty]]:
        # Include grader instructions from rubric if available
        grader_instructions = self.rubric.get("grader_instructions", "")
        
        system_prompt = (
            "You are a HARSH grader evaluating recruiter outreach messages. "
            "Follow the provided rubric EXACTLY. Count specific criteria met and penalties triggered.\n\n"
            f"{grader_instructions}\n\n"
            "Output JSON with:\n"
            "- section_scores: array with one entry per rubric section (section_id, score, comments explaining which criteria were met)\n"
            "- penalties: array of ALL applicable penalties (reason quoting the specific penalty, score as negative number), or empty array if none\n\n"
            "Be literal: if you see template phrases like 'impressive background' or 'extensive experience', apply the penalty. "
            "If the message doesn't explicitly acknowledge a mismatch, don't give credit for implying it.\n"
            "Do not include a total score. Do not add extra fields."
        )
        user_payload = {
            "rubric": self.rubric,
            "prompt_context": prompt,
            "message": dm,
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
        resp = self.client.beta.chat.completions.parse(
            model=self.grader_model,
            messages=messages,  # type: ignore[arg-type]
            response_format=GradedSections,
        )
        # beta.chat.completions.parse returns parsed content under choices[].message.parsed
        parsed_content = resp.choices[0].message.parsed
        content = parsed_content.model_dump_json() if parsed_content is not None else (resp.choices[0].message.content or "")
        try:
            graded = GradedSections.model_validate_json(content)
            section_scores = graded.section_scores
            penalties = graded.penalties

            total_sections = 0.0
            comments: list[str] = []
            for section in section_scores:
                total_sections += float(section.score)
                if section.comments:
                    comments.append(f"{section.section_id}: {section.comments}")

            total_penalties = 0.0
            penalty_notes: list[str] = []
            for pen in penalties:
                total_penalties += float(pen.score)
                if pen.reason:
                    penalty_notes.append(pen.reason)

            total_score = total_sections + total_penalties
            justification_parts = []
            if comments:
                justification_parts.append("; ".join(comments))
            if penalty_notes:
                justification_parts.append(f"Penalties: {', '.join(penalty_notes)}")
            justification = " | ".join(justification_parts) or "Section scores aggregated with penalties."

            return total_score, justification, section_scores, penalties
        except Exception:
            # If parsing fails, treat as zero.
            return 0.0, "Grader response parsing failed.", [], []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", 
        type=Path, 
        default=Path("data/candidates_formatted.jsonl"),
        help="Path to candidates_formatted.jsonl (JSONL with messages format)"
    )
    parser.add_argument("--rubric", type=Path, default=Path("data/rubric.json"))
    parser.add_argument("--limit", type=int, default=None, help="Max examples to evaluate (default: all)")
    parser.add_argument("--renderer", type=str, default="qwen3")
    parser.add_argument("--creator-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--grader-model", type=str, default=DEFAULT_GRADER_MODEL)
    parser.add_argument("--grader-timeout", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dry-run", action="store_true", help="Skip grader call; only generate drafts.")
    parser.add_argument("--verbose", action="store_true", help="Print draft and grader JSON for each example.")
    args = parser.parse_args()

    data = load_jsonl(args.dataset)
    rubric = load_json(args.rubric)
    dataset = build_dataset_from_jsonl(data, limit=args.limit)

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=args.creator_model)

    evaluator = OutboundEvaluator(
        dataset=dataset,
        rubric=rubric,
        renderer_name=args.renderer,
        model_name=args.creator_model,
        grader_model=args.grader_model,
        grader_timeout=args.grader_timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        verbose=args.verbose,
    )

    async def run_eval():
        if args.dry_run:
            # Just run creator, skip grader to debug latency.
            sampling_params = types.SamplingParams(
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=1.0,
                stop=evaluator.renderer.get_stop_sequences(),
            )
            for idx, datum in enumerate(dataset, start=1):
                print(f"[creator] sampling example {idx}/{len(dataset)} ...", flush=True)
                model_input: types.ModelInput = evaluator.renderer.build_generation_prompt(
                    [renderers.Message(role="user", content=datum["prompt"])]
                )
                resp: types.SampleResponse = await sampling_client.sample_async(
                    prompt=model_input, num_samples=1, sampling_params=sampling_params
                )
                tokens: List[int] = resp.sequences[0].tokens
                message: renderers.Message = evaluator.renderer.parse_response(tokens)[0]
                print(f"[draft] {message['content'][:200]}...\n")
            return

        metrics = await evaluator(sampling_client)
        print(metrics)

    import asyncio

    asyncio.run(run_eval())


if __name__ == "__main__":
    main()

