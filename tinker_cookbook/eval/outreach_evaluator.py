"""
Outbound-message evaluator aligned with Tinker SamplingClientEvaluator.

Flow:
- Dataset items contain a candidate profile and job description.
- Sampling model generates a LinkedIn-style DM.
- A grader model (OpenAI gpt-5-mini by default) scores the DM using rubric.json.

Usage (example):
  python -m tinker_cookbook.eval.outreach_evaluator \
    --candidates data/candidates.json \
    --roles roles.json \
    --rubric rubric.json \
    --limit 10 \
    --creator-model Qwen/Qwen2-7B-Instruct \
    --renderer qwen3

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
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.tokenizer_utils import get_tokenizer


DEFAULT_GRADER_MODEL = "gpt-4.1"

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_dataset(candidates: List[Dict[str, Any]], roles: List[Dict[str, Any]], limit: int | None = None) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for idx, cand in enumerate(candidates):
        if limit and idx >= limit:
            break
        # match role by URL if present
        role = next((r for r in roles if r.get("absolute_url") == cand.get("target_role_url")), None)
        role_text = ""
        if role:
            role_text = role.get("content_text") or role.get("content_html") or ""
        else:
            role_text = cand.get("job_description_excerpt") or ""

        prompt = (
            "You are a recruiter crafting a concise, respectful LinkedIn DM to a candidate about a role.\n"
            "Use the candidate profile and the job description below. Keep it <130 words, clear CTA, no fluff.\n\n"
            f"Candidate Profile:\n{cand}\n\n"
            f"Job Description:\n{role_text}\n"
        )
        items.append({"prompt": prompt})
    return items


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
            print(f"[grader] scoring example {idx}/{len(self.dataset)} ...", flush=True)
            score, justification = self.grade(dm_text, datum["prompt"])
            if self.verbose:
                print("=== Draft Message ===")
                print(dm_text)
                print("=== Grader ===")
                print(json.dumps({"score": score, "justification": justification}, ensure_ascii=False, indent=2))
                print("=====================")
            scores.append(score)

        avg = sum(scores) / len(scores) if scores else 0.0
        return {"outreach_score": avg}

    def grade(self, dm: str, prompt: str) -> tuple[float, str]:
        system_prompt = (
            "You are an objective grader. Score the provided LinkedIn DM using ONLY the rubric.\n"
            "Return JSON with fields:\n"
            '{ "score": number, "justification": string }\n'
            "- score: total numeric score\n"
            "- justification: 1-3 sentences citing rubric aspects; do NOT add extra fields.\n"
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
        resp = self.client.chat.completions.create(
            model=self.grader_model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "graded_output",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "score": {"type": "number"},
                            "justification": {"type": "string"},
                        },
                        "required": ["score", "justification"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        )
        content = resp.choices[0].message.content
        try:
            parsed = json.loads(content)
            return float(parsed.get("score", 0.0)), str(parsed.get("justification", ""))
        except Exception:
            # If parsing fails, treat as zero.
            return 0.0, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=Path("data/candidates.json"))
    parser.add_argument("--roles", type=Path, default=Path("roles.json"))
    parser.add_argument("--rubric", type=Path, default=Path("rubric.json"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--renderer", type=str, default="qwen3")
    parser.add_argument("--creator-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--grader-model", type=str, default=DEFAULT_GRADER_MODEL)
    parser.add_argument("--grader-timeout", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dry-run", action="store_true", help="Skip grader call; only generate drafts.")
    parser.add_argument("--verbose", action="store_true", help="Print draft and grader JSON for each example.")
    args = parser.parse_args()

    candidates = load_json(args.candidates)
    roles = load_json(args.roles)
    rubric = load_json(args.rubric)
    dataset = build_dataset(candidates, roles, limit=args.limit)

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

