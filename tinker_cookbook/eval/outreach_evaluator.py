"""
Outbound-message evaluator aligned with on_policy_context_distillation training.

This evaluator is designed to match the training format exactly:
- Student sees: [{"role": "user", "content": "<task instructions + candidate profile + job info>"}]
- NO system message (matches training where student has no system instructions)
- Task instructions are baked into the data (from candidates_formatted.jsonl)

Flow:
1. Load dataset from candidates_formatted.jsonl (self-contained prompts with instructions)
2. Sampling model generates a LinkedIn-style DM
3. A grader model (OpenAI gpt-4.1 by default) scores the DM using rubric.json

The rubric and candidate data are designed to be challenging:
- ~55% of candidates are "edge cases" with subtle mismatches, red flags, or sparse profiles
- Rubric heavily penalizes surface-level personalization and template patterns
- High scores (20+) require genuine insight, not just restating profile details

Expected score distribution:
- 0-5: Poor/template messages
- 6-11: Weak, surface-level matching
- 12-19: Adequate, where most decent messages land
- 20-27: Strong, uncommon
- 28-35: Exceptional, extremely rare

Usage:
  python -m tinker_cookbook.eval.outreach_evaluator \\
    --dataset data/candidates_formatted.jsonl \\
    --rubric data/rubric.json \\
    --limit 10 \\
    --creator-model Qwen/Qwen3-4B-Instruct-2507 \\
    --renderer qwen3 \\
    --verbose

Env:
  OPENAI_API_KEY must be set for the grader call.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import tinker
from openai import AsyncOpenAI
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

    section_scores: List[SectionScore]
    penalties: List[Penalty]  # Empty list if no penalties


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_dataset_from_formatted_jsonl(jsonl_path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    """Load dataset from candidates_formatted.jsonl - matches training format exactly.
    
    The JSONL format has {"messages": [{"role": "user", "content": "..."}]} per line.
    Each data point is self-contained with task instructions, candidate profile, and job description.
    
    IMPORTANT: We pass through the content as-is, since instructions are already baked in.
    Do NOT wrap with additional instructions - that would create a mismatch with training.
    """
    items: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row.get("messages", [])
            # Extract user message content (already has instructions baked in)
            user_msg = next((m for m in messages if m.get("role") == "user"), None)
            if user_msg:
                items.append({"prompt": user_msg["content"]})
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
    for line in dm.split("\n"):
        print(f"  {line}")
    print(f"{_DIM}{'─' * 40}{_RESET}")

    # Section scores
    if section_scores:
        print(f"\n{_BOLD}📊 Section Scores:{_RESET}")
        for sec in section_scores:
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
    """Evaluator for outreach message generation.
    
    IMPORTANT: To match on_policy_context_distillation training:
    - Use dataset from build_dataset_from_formatted_jsonl() (instructions already baked in)
    - Use convo_prefix=[] (empty, NO system message)
    
    The student model during training sees the user message with task instructions,
    candidate info, and job description all baked in. NO additional wrapping needed.
    """
    
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
        convo_prefix: List[renderers.Message] | None = None,
    ):
        self.dataset = dataset
        self.rubric = rubric
        self.grader_model = grader_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose
        # Default to empty convo_prefix to match training (student has no system message)
        self.convo_prefix = convo_prefix if convo_prefix is not None else []

        self.grader_timeout = grader_timeout
        tokenizer = get_tokenizer(model_name)
        self.renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)
        self.client = AsyncOpenAI(timeout=grader_timeout)

    async def __call__(self, sampling_client: tinker.SamplingClient) -> Dict[str, float]:
        sampling_params = types.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=1.0,
            stop=self.renderer.get_stop_sequences(),
        )

        # Build all model inputs upfront
        model_inputs: List[types.ModelInput] = []
        for datum in self.dataset:
            convo = self.convo_prefix + [renderers.Message(role="user", content=datum["prompt"])]
            model_inputs.append(self.renderer.build_generation_prompt(convo))

        # Sample all in parallel
        print(f"[creator] sampling {len(self.dataset)} examples in parallel ...", flush=True)
        sample_tasks = [
            sampling_client.sample_async(prompt=mi, num_samples=1, sampling_params=sampling_params)
            for mi in model_inputs
        ]
        responses: List[types.SampleResponse] = await asyncio.gather(*sample_tasks)
        print(f"[creator] sampling complete.", flush=True)

        # Parse responses and prepare grading tasks
        drafts: List[tuple[int, str, str]] = []  # (idx, dm_clean, prompt)
        scores: List[float] = [0.0] * len(self.dataset)
        skip_results: List[tuple[int, str]] = []  # (idx, dm_text) for empty/invalid

        for idx, (resp, datum) in enumerate(zip(responses, self.dataset)):
            tokens: List[int] = resp.sequences[0].tokens
            message: renderers.Message = self.renderer.parse_response(tokens)[0]
            dm_text = message["content"]
            dm_clean = (dm_text or "").strip()

            if not dm_clean or dm_clean.startswith("<|im_start|>") or len(dm_clean) < 10:
                skip_results.append((idx, dm_text or ""))
            else:
                drafts.append((idx, dm_clean, datum["prompt"]))

        # Print skipped results if verbose
        for idx, dm_text in skip_results:
            if self.verbose:
                _print_result(idx + 1, len(self.dataset), dm_text, 0.0, "Empty or invalid draft; skipped grading.", [], [])

        # Grade all valid drafts in parallel
        if drafts:
            print(f"{_DIM}[grader] scoring {len(drafts)} examples in parallel ...{_RESET}", flush=True)
            grade_tasks = [self.grade_async(dm_clean, prompt) for _, dm_clean, prompt in drafts]
            grade_results = await asyncio.gather(*grade_tasks)
            print(f"[grader] scoring complete.", flush=True)

            for (idx, dm_clean, _), (score, justification, section_scores, penalties) in zip(drafts, grade_results):
                scores[idx] = score
                if self.verbose:
                    _print_result(idx + 1, len(self.dataset), dm_clean, score, justification, section_scores, penalties)

        avg = sum(scores) / len(scores) if scores else 0.0
        return {"outreach_score": avg}

    async def grade_async(self, dm: str, prompt: str) -> tuple[float, str, List[SectionScore], List[Penalty]]:
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
        resp = await self.client.beta.chat.completions.parse(
            model=self.grader_model,
            messages=messages,  # type: ignore[arg-type]
            response_format=GradedSections,
        )
        parsed_content = resp.choices[0].message.parsed
        content = parsed_content.model_dump_json() if parsed_content is not None else (resp.choices[0].message.content or "")
        try:
            graded = GradedSections.model_validate_json(content)
            section_scores = graded.section_scores
            penalties = graded.penalties

            total_sections = sum(float(s.score) for s in section_scores)
            total_penalties = sum(float(p.score) for p in penalties)
            total_score = total_sections + total_penalties

            comments = [f"{s.section_id}: {s.comments}" for s in section_scores if s.comments]
            penalty_notes = [p.reason for p in penalties if p.reason]
            
            justification_parts = []
            if comments:
                justification_parts.append("; ".join(comments))
            if penalty_notes:
                justification_parts.append(f"Penalties: {', '.join(penalty_notes)}")
            justification = " | ".join(justification_parts) or "Section scores aggregated with penalties."

            return total_score, justification, section_scores, penalties
        except Exception:
            return 0.0, "Grader response parsing failed.", [], []


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate outreach message generation using candidates_formatted.jsonl"
    )
    parser.add_argument(
        "--dataset", 
        type=Path, 
        default=Path("data/candidates_formatted.jsonl"),
        help="Path to candidates_formatted.jsonl (JSONL with self-contained prompts)"
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
    parser.add_argument("--verbose", action="store_true", help="Print draft and grader details for each example.")
    args = parser.parse_args()

    # Load dataset - prompts are self-contained (instructions + profile + job already baked in)
    dataset = build_dataset_from_formatted_jsonl(args.dataset, limit=args.limit)
    print(f"[INFO] Loaded {len(dataset)} examples from {args.dataset}")
    
    rubric = load_json(args.rubric)

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
        # No convo_prefix - matches training where student has no system message
    )

    async def run_eval():
        if args.dry_run:
            sampling_params = types.SamplingParams(
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=1.0,
                stop=evaluator.renderer.get_stop_sequences(),
            )
            # Build all prompts and sample in parallel
            model_inputs = []
            for datum in dataset:
                convo = evaluator.convo_prefix + [renderers.Message(role="user", content=datum["prompt"])]
                model_inputs.append(evaluator.renderer.build_generation_prompt(convo))
            
            print(f"[creator] sampling {len(dataset)} examples in parallel ...", flush=True)
            sample_tasks = [
                sampling_client.sample_async(prompt=mi, num_samples=1, sampling_params=sampling_params)
                for mi in model_inputs
            ]
            responses = await asyncio.gather(*sample_tasks)
            print(f"[creator] sampling complete.\n", flush=True)
            
            for idx, resp in enumerate(responses, start=1):
                tokens: List[int] = resp.sequences[0].tokens
                message: renderers.Message = evaluator.renderer.parse_response(tokens)[0]
                content = message['content'] or ""
                preview = content[:200] + "..." if len(content) > 200 else content
                print(f"[draft {idx}/{len(dataset)}] {preview}\n")
            return

        metrics = await evaluator(sampling_client)
        print(f"\n{_BOLD}Final metrics:{_RESET} {metrics}")

    asyncio.run(run_eval())


if __name__ == "__main__":
    main()
