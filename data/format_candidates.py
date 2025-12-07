"""
Format synthetic candidate profiles into a simple JSONL messages format
for context distillation training.

Reads: ../candidates.json
Writes: data/candidates_formatted.jsonl

Each line:
{
  "messages": [
    {"role": "user", "content": "<task instruction + candidate profile>"}
  ]
}

The user content includes:
1. Task instruction header ("Write a LinkedIn outreach message...")
2. Candidate profile data
3. Task instruction footer ("Write a personalized, professional message...")

This format matches the few-shot examples used in on_policy_context_distillation
training, ensuring consistency between the teacher's few-shot context and the
actual training prompts.

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


TASK_INSTRUCTION_HEADER = "Write a LinkedIn outreach message for this candidate and role:"

TASK_INSTRUCTION_FOOTER = (
    "Write a personalized, professional message (under 150 words) that references "
    "their specific background and explains why this role might interest them."
)


def render_profile(c: dict, include_instructions: bool = True) -> str:
    """Render a candidate profile as plain text.
    
    Args:
        c: Candidate dict from candidates.json
        include_instructions: If True, wrap with task instructions to match 
                              few-shot format used in training
    
    Field mapping from candidates.json:
        - skills_core OR skills → Skills
        - projects OR highlights → Projects (accomplishments/achievements)
        - recent_experience OR current_title + experience_years → Experience
        - summary OR specialization + current_title → Summary
        - title OR target_role_title → Target role
        - seniority (may not exist in all records)
    """
    # Build summary from available fields
    summary = c.get("summary")
    if not summary:
        # Construct from specialization and current_title
        specialization = c.get("specialization", "")
        current_title = c.get("current_title", "")
        exp_years = c.get("experience_years", "")
        
        parts = []
        if current_title:
            parts.append(current_title)
        if specialization:
            parts.append(f"specializing in {specialization}")
        if exp_years:
            parts.append(f"with {exp_years} years experience")
        summary = ", ".join(parts) if parts else None
    
    # Get role title - try both field names
    role_title = c.get("title") or c.get("target_role_title") or "Unknown Role"
    
    profile_parts = [
        f"Name: {c.get('name')}",
        f"Target role: {role_title} ({c.get('target_role_url')})",
        f"Location: {c.get('location')} | Timezone: {c.get('timezone')}",
        f"Seniority: {c.get('seniority', 'Not specified')} | Domain: {c.get('domain')}",
        f"Summary: {summary or 'Not provided'}",
    ]

    # Skills - try both field names
    skills_list = c.get("skills_core") or c.get("skills") or []
    skills = ", ".join(skills_list)
    if skills:
        profile_parts.append(f"Skills: {skills}")

    # Projects/Highlights - accomplishments the model should reference
    projects = c.get("projects") or c.get("highlights") or []
    if projects:
        profile_parts.append("Projects:\n- " + "\n- ".join(projects))

    # Experience - try structured format first, then fall back to simple format
    exp = c.get("recent_experience") or []
    if exp:
        bullets = []
        for role in exp:
            highlights = role.get("highlights") or []
            hl = "; ".join(highlights)
            bullets.append(f"{role.get('title')} @ {role.get('company')} ({role.get('tenure')}): {hl}")
        profile_parts.append("Experience:\n- " + "\n- ".join(bullets))
    else:
        # Fall back to current_title + experience_years
        current_title = c.get("current_title")
        exp_years = c.get("experience_years")
        if current_title:
            exp_str = f"{current_title}"
            if exp_years:
                exp_str += f" ({exp_years} years)"
            profile_parts.append(f"Experience:\n- {exp_str}")

    edu = c.get("education")
    if edu:
        profile_parts.append(f"Education: {edu}")

    open_note = c.get("openness")
    if open_note:
        profile_parts.append(f"Openness: {open_note}")

    jd_excerpt = c.get("job_description_excerpt") or ""
    if jd_excerpt:
        profile_parts.append("Job excerpt:\n" + jd_excerpt.strip())

    profile_text = "\n".join(profile_parts)
    
    if include_instructions:
        # Match the few-shot format from on_policy_context_distillation
        return f"{TASK_INSTRUCTION_HEADER}\n\n{profile_text}\n\n{TASK_INSTRUCTION_FOOTER}"
    else:
        return profile_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format candidate profiles for context distillation training."
    )
    parser.add_argument("--input", type=Path, help="Path to candidates.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path")
    parser.add_argument(
        "--no-instructions",
        action="store_true",
        help="Omit task instructions (profile data only). Default includes instructions."
    )
    args = parser.parse_args()

    input_path = args.input
    if not input_path:
        input_path = next((p for p in DEFAULT_INPUTS if p.exists()), None)
    if not input_path or not input_path.exists():
        raise SystemExit(f"candidates.json not found. Looked in: {DEFAULT_INPUTS}")

    include_instructions = not args.no_instructions

    with input_path.open("r", encoding="utf-8") as fh:
        candidates = json.load(fh)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for c in candidates:
            profile_text = render_profile(c, include_instructions=include_instructions)
            record = {"messages": [{"role": "user", "content": profile_text}]}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    mode = "with task instructions" if include_instructions else "profile data only"
    print(f"Wrote {len(candidates)} records ({mode}) to {args.output}")


if __name__ == "__main__":
    main()

