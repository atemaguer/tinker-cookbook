"""
Format synthetic candidate profiles into a simple JSONL messages format
similar to prompt_distillation/create_data.py.

Reads: ../candidates.json
Writes: data/candidates_formatted.jsonl

Each line:
{
  "messages": [
    {"role": "user", "content": "<plain-text candidate profile>"}
  ]
}

We keep the original candidates.json untouched.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUTS = [
    BASE_DIR / "candidates.json",
    BASE_DIR / "data" / "candidates.json",
]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "candidates_formatted.jsonl"


def render_profile(c: dict) -> str:
    parts = [
        f"Name: {c.get('name')}",
        f"Target role: {c.get('title')} ({c.get('target_role_url')})",
        f"Location: {c.get('location')} | Timezone: {c.get('timezone')}",
        f"Seniority: {c.get('seniority')} | Domain: {c.get('domain')}",
        f"Summary: {c.get('summary')}",
    ]

    skills = ", ".join(c.get("skills_core", []))
    if skills:
        parts.append(f"Skills: {skills}")

    projects = c.get("projects") or []
    if projects:
        parts.append("Projects:\n- " + "\n- ".join(projects))

    exp = c.get("recent_experience") or []
    if exp:
        bullets = []
        for role in exp:
            highlights = role.get("highlights") or []
            hl = "; ".join(highlights)
            bullets.append(f"{role.get('title')} @ {role.get('company')} ({role.get('tenure')}): {hl}")
        parts.append("Experience:\n- " + "\n- ".join(bullets))

    edu = c.get("education")
    if edu:
        parts.append(f"Education: {edu}")

    open_note = c.get("openness")
    if open_note:
        parts.append(f"Openness: {open_note}")

    jd_excerpt = c.get("job_description_excerpt") or ""
    if jd_excerpt:
        parts.append("Job excerpt:\n" + jd_excerpt.strip())

    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="Path to candidates.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path")
    args = parser.parse_args()

    input_path = args.input
    if not input_path:
        input_path = next((p for p in DEFAULT_INPUTS if p.exists()), None)
    if not input_path or not input_path.exists():
        raise SystemExit(f"candidates.json not found. Looked in: {DEFAULT_INPUTS}")

    with input_path.open("r", encoding="utf-8") as fh:
        candidates = json.load(fh)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for c in candidates:
            profile_text = render_profile(c)
            record = {"messages": [{"role": "user", "content": profile_text}]}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(candidates)} records to {args.output}")


if __name__ == "__main__":
    main()

