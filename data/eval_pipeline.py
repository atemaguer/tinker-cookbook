"""
Eval pipeline:
1) Call a "creator" model (GPT-4o) to draft a LinkedIn DM given a candidate profile + job description.
   - System prompt describes what a good message looks like (concise, role/candidate aligned, clear CTA).
   - The rubric is NOT mentioned to the creator model.
2) Call a "grader" model (gpt-5-mini) with the rubric, the same inputs, and the creator's message.
   - Grader must emit ONLY a numeric score (total), no commentary.

Usage:
  python data/eval_pipeline.py \
    --candidate candidates.json --candidate-index 0 \
    --roles roles.json --role-index 0 \
    --rubric rubric.json

Env:
  OPENAI_API_KEY must be set.
Models:
  creator_model = "gpt-4o"
  grader_model  = "gpt-5-mini"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI


CREATOR_MODEL = "gpt-4o"
GRADER_MODEL = "gpt-5-mini"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_creator_messages(candidate: Dict[str, Any], role: Dict[str, Any]) -> list[Dict[str, str]]:
    system_prompt = (
        "You are a recruiter crafting a short, respectful LinkedIn DM to a candidate about a specific role.\n"
        "Requirements for a GOOD message:\n"
        "- Reference the role accurately (title, focus, location/remote expectations if present).\n"
        "- Personalize using candidate background (skills, domain, experience highlights) without inventing facts.\n"
        "- Be concise (ideally <130 words), clear, and friendly; avoid corporate fluff.\n"
        "- Provide a low-friction call to action (e.g., quick chat or share interest/link).\n"
        "Things to avoid (BAD message):\n"
        "- Generic mass-mail tone, excessive hype, or walls of text.\n"
        "- Hallucinating candidate history or role details.\n"
        "- Ignoring location/onsite constraints if stated.\n"
    )

    user_content = {
        "candidate_profile": candidate,
        "job_description": role.get("content_text") or role.get("content_html") or "",
    }

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
    ]


def build_grader_messages(rubric: Dict[str, Any], candidate: Dict[str, Any], role: Dict[str, Any], message: str) -> list[Dict[str, str]]:
    system_prompt = (
        "You are an objective grader. Score the provided LinkedIn DM using ONLY the rubric.\n"
        "Return ONE value: the total numeric score. No explanations, no JSON, no extra text.\n"
    )
    user_payload = {
        "rubric": rubric,
        "candidate_profile": candidate,
        "job_description": role.get("content_text") or role.get("content_html") or "",
        "message": message,
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def call_model(client: OpenAI, model: str, messages: list[Dict[str, str]]) -> str:
    resp = client.chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True, help="Path to candidates.json")
    parser.add_argument("--candidate-index", type=int, default=0, help="Index of candidate to use")
    parser.add_argument("--roles", type=Path, required=True, help="Path to roles.json")
    parser.add_argument("--role-index", type=int, default=0, help="Index of role to use")
    parser.add_argument("--rubric", type=Path, required=True, help="Path to rubric.json")
    args = parser.parse_args()

    candidates = load_json(args.candidate)
    roles = load_json(args.roles)
    rubric = load_json(args.rubric)

    try:
        candidate = candidates[args.candidate_index]
    except Exception:
        sys.exit(f"candidate_index {args.candidate_index} out of range")
    try:
        role = roles[args.role_index]
    except Exception:
        sys.exit(f"role_index {args.role_index} out of range")

    client = OpenAI()

    creator_messages = build_creator_messages(candidate, role)
    draft = call_model(client, CREATOR_MODEL, creator_messages)

    grader_messages = build_grader_messages(rubric, candidate, role, draft)
    score = call_model(client, GRADER_MODEL, grader_messages)

    output = {
        "candidate_id": candidate.get("id"),
        "role_url": role.get("absolute_url"),
        "draft_message": draft,
        "score": score,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

