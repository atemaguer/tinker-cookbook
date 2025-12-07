"""
Modular grader for outreach messages.

This module provides an OutreachGrader class that can be used independently
for grading recruiting messages against a rubric. It's designed to be reusable
across different contexts:
- Standalone evaluation (outreach_evaluator.py)
- Feedback-augmented context distillation
- Any other pipeline that needs message quality feedback

The grader uses an LLM (OpenAI by default) to score messages based on a rubric
and returns both scores and textual feedback suitable for use as context.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict


DEFAULT_GRADER_MODEL = "gpt-5.1"


class SectionScore(BaseModel):
    """Score for a single rubric section."""
    model_config = ConfigDict(extra="forbid")

    section_id: str
    score: float
    comments: str


class Penalty(BaseModel):
    """A penalty applied to the message."""
    model_config = ConfigDict(extra="forbid")

    reason: str
    score: float


class GradedSections(BaseModel):
    """Structured grading output from the LLM."""
    model_config = ConfigDict(extra="forbid")

    section_scores: List[SectionScore]
    penalties: List[Penalty]  # Empty list if no penalties


class GradeResult:
    """Result of grading a single message.
    
    Contains both numerical scores and textual feedback that can be
    incorporated into a training context.
    """
    
    def __init__(
        self,
        total_score: float,
        section_scores: List[SectionScore],
        penalties: List[Penalty],
        justification: str,
    ):
        self.total_score = total_score
        self.section_scores = section_scores
        self.penalties = penalties
        self.justification = justification
    
    def to_feedback_text(self, include_score: bool = True) -> str:
        """Convert grading result to feedback text for context augmentation.
        
        This format is designed to be informative for the teacher model,
        providing specific feedback about what was good/bad in the message.
        """
        parts = []
        
        if include_score:
            parts.append(f"Overall Score: {self.total_score:.1f}/35")
        
        # Section feedback (only include sections with comments)
        section_feedback = []
        for sec in self.section_scores:
            if sec.comments:
                section_feedback.append(f"- {sec.section_id}: {sec.comments}")
        
        if section_feedback:
            parts.append("Feedback:\n" + "\n".join(section_feedback))
        
        # Penalties (important for teacher to understand what to avoid)
        if self.penalties:
            penalty_notes = [f"- {p.reason}" for p in self.penalties]
            parts.append("Issues to address:\n" + "\n".join(penalty_notes))
        
        return "\n\n".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_score": self.total_score,
            "section_scores": [s.model_dump() for s in self.section_scores],
            "penalties": [p.model_dump() for p in self.penalties],
            "justification": self.justification,
        }


class OutreachGrader:
    """Grader for outreach messages using LLM-as-judge.
    
    Grades messages against a rubric and returns structured feedback.
    Can be used for both evaluation and feedback-augmented training.
    
    Example usage:
        grader = OutreachGrader(rubric=rubric)
        result = await grader.grade_async(
            message="Hi Sofia, your ML work...",
            prompt_context="Write a LinkedIn outreach message for..."
        )
        feedback_text = result.to_feedback_text()
    """
    
    def __init__(
        self,
        rubric: Dict[str, Any],
        grader_model: str = DEFAULT_GRADER_MODEL,
        grader_timeout: float = 30.0,
    ):
        """Initialize the grader.
        
        Args:
            rubric: The grading rubric (loaded from rubric.json)
            grader_model: OpenAI model to use for grading
            grader_timeout: Timeout for grading calls
        """
        self.rubric = rubric
        self.grader_model = grader_model
        self.grader_timeout = grader_timeout
        self.client = AsyncOpenAI(timeout=grader_timeout)
    
    async def grade_async(
        self,
        message: str,
        prompt_context: str,
    ) -> GradeResult:
        """Grade a message asynchronously.
        
        Args:
            message: The outreach message to grade
            prompt_context: The original prompt/context for the message
                           (candidate profile, job description, etc.)
        
        Returns:
            GradeResult with scores, penalties, and feedback text
        """
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
            "prompt_context": prompt_context,
            "message": message,
        }
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
        
        try:
            resp = await self.client.beta.chat.completions.parse(
                model=self.grader_model,
                messages=messages,  # type: ignore[arg-type]
                response_format=GradedSections,
            )
            
            parsed_content = resp.choices[0].message.parsed
            content = (
                parsed_content.model_dump_json()
                if parsed_content is not None
                else (resp.choices[0].message.content or "")
            )
            
            graded = GradedSections.model_validate_json(content)
            
            total_sections = sum(float(s.score) for s in graded.section_scores)
            total_penalties = sum(float(p.score) for p in graded.penalties)
            total_score = total_sections + total_penalties
            
            # Build justification
            comments = [
                f"{s.section_id}: {s.comments}"
                for s in graded.section_scores
                if s.comments
            ]
            penalty_notes = [p.reason for p in graded.penalties if p.reason]
            
            justification_parts = []
            if comments:
                justification_parts.append("; ".join(comments))
            if penalty_notes:
                justification_parts.append(f"Penalties: {', '.join(penalty_notes)}")
            justification = (
                " | ".join(justification_parts)
                or "Section scores aggregated with penalties."
            )
            
            return GradeResult(
                total_score=total_score,
                section_scores=graded.section_scores,
                penalties=graded.penalties,
                justification=justification,
            )
            
        except Exception as e:
            # Return empty result on failure
            return GradeResult(
                total_score=0.0,
                section_scores=[],
                penalties=[],
                justification=f"Grader response parsing failed: {e}",
            )
    
    async def grade_batch_async(
        self,
        messages: List[str],
        prompt_contexts: List[str],
        serial: bool = False,
    ) -> List[GradeResult]:
        """Grade multiple messages.
        
        Args:
            messages: List of outreach messages to grade
            prompt_contexts: List of corresponding prompts/contexts
            serial: If True, grade one at a time (avoids rate limits). 
                   If False, grade in parallel.
        
        Returns:
            List of GradeResult objects
        """
        import asyncio
        
        if serial:
            # Grade one at a time to avoid rate limits
            results = []
            for msg, ctx in zip(messages, prompt_contexts):
                result = await self.grade_async(msg, ctx)
                results.append(result)
            return results
        else:
            # Grade in parallel
            tasks = [
                self.grade_async(msg, ctx)
                for msg, ctx in zip(messages, prompt_contexts)
            ]
            return await asyncio.gather(*tasks)


def load_rubric(path: Path | str) -> Dict[str, Any]:
    """Load rubric from JSON file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)

