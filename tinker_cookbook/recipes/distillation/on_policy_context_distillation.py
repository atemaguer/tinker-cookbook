"""
On-policy context distillation for recruiting outreach messages.

This script combines prompt distillation and on-policy distillation:
- Student: Receives only the candidate profile + job description (NO examples)
- Teacher: Receives few-shot examples of good outreach messages + candidate/job info

The goal is to train the student to internalize high-quality outreach patterns,
learning to write effective LinkedIn DMs as if it had seen examples, without actually having them.

Key concepts:
- Context asymmetry: Teacher sees few-shot examples, student does not
- On-policy sampling: Student generates messages without context
- KL penalty: Student learns to match teacher's distribution (which has context)
- Context internalization: Model learns to replicate good messaging patterns it never sees

Example usage:
    # Recruiting outreach with context distillation (uses default candidates_formatted.jsonl)
    python -m tinker_cookbook.recipes.distillation.on_policy_context_distillation \\
        model_name=Qwen/Qwen3-4B-Instruct-2507 \\
        teacher_model=Qwen/Qwen3-4B-Instruct-2507 \\
        dataset=recruiting \\
        learning_rate=1e-4 \\
        groups_per_batch=16 \\
        lora_rank=128 \\
        wandb_project=cookbook_context_distillation

    # With pre-formatted JSONL (prompts already built in messages array)
    python -m tinker_cookbook.recipes.distillation.on_policy_context_distillation \\
        model_name=Qwen/Qwen3-4B-Instruct-2507 \\
        teacher_model=Qwen/Qwen3-4B-Instruct-2507 \\
        dataset=recruiting \\
        candidates_path=data/candidates_formatted.jsonl \\
        learning_rate=1e-4 \\
        groups_per_batch=16 \\
        wandb_project=cookbook_context_distillation

    # With raw candidates.json + roles.json (like outreach_evaluator.py)
    python -m tinker_cookbook.recipes.distillation.on_policy_context_distillation \\
        model_name=Qwen/Qwen3-4B-Instruct-2507 \\
        teacher_model=Qwen/Qwen3-4B-Instruct-2507 \\
        dataset=recruiting \\
        candidates_path=data/candidates.json \\
        roles_path=data/roles.json \\
        learning_rate=1e-4 \\
        groups_per_batch=16 \\
        wandb_project=cookbook_context_distillation

    # With Llama model
    python -m tinker_cookbook.recipes.distillation.on_policy_context_distillation \\
        model_name=meta-llama/Llama-3.1-8B \\
        teacher_model=meta-llama/Llama-3.1-8B \\
        dataset=recruiting \\
        learning_rate=1e-4 \\
        groups_per_batch=8 \\
        wandb_project=cookbook_context_distillation
"""

import asyncio
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Sequence, cast

import chz
import tinker
import torch
from tinker_cookbook import checkpoint_utils, cli_utils, model_info, renderers
from tinker_cookbook.display import colorize_example
from tinker_cookbook.distillation.datasets import PromptOnlyEnv
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
from tinker_cookbook.rl.metric_util import compute_trajectory_metrics
from tinker_cookbook.rl.metrics import discounted_future_sum_vectorized
from tinker_cookbook.rl.train import (
    compute_full_batch_metrics_and_get_sampling_client,
    save_checkpoint_and_get_sampling_client,
    train_step,
)
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Metrics,
    RLDataset,
    Trajectory,
    TrajectoryGroup,
)
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import safezip, timed
from tinker_cookbook.utils.trace import get_scope_context, scope, trace_init

logger = logging.getLogger(__name__)

# Default path to candidates data (relative to this file's parent directory)
DEFAULT_CANDIDATES_PATH = Path(__file__).parent.parent.parent.parent / "data" / "candidates_formatted.jsonl"


# ============================================================================
# Few-shot examples for recruiting outreach context distillation
# ============================================================================

RECRUITING_FEWSHOT_EXAMPLES: list[renderers.Message] = [
    {
        "role": "user",
        "content": """Write a LinkedIn outreach message for this candidate and role:

**Candidate Profile:**
- Name: Alex Chen
- Current Title: Senior ML Engineer at Anthropic
- Skills: PyTorch, Transformers, RLHF, Distributed Training
- Experience: 4 years building LLM training infrastructure
- Location: San Francisco, CA

**Target Role:**
- Title: Member of Technical Staff, Post-training
- Company: xAI
- Location: Palo Alto, CA
- Key Requirements: Experience with RLHF, large-scale training, transformer architectures

Write a personalized, professional message (under 150 words) that references their specific background and explains why this role might interest them.""",
    },
    {
        "role": "assistant",
        "content": """Hi Alex,

I came across your work at Anthropic on LLM training infrastructure—the scale and complexity of that work is impressive.

We're building the post-training team at xAI, focused on RLHF and alignment at scales that push the boundaries of what's possible. Given your hands-on experience with distributed training and transformers, I think you'd thrive in our environment.

What makes xAI different: flat structure, direct ownership of critical systems, and a team obsessed with moving fast. The Palo Alto commute from SF is easy, and we're flexible on hybrid arrangements.

Would you be open to a quick chat this week? I'd love to share more about what we're building.

Best,
[Recruiter]""",
    },
    {
        "role": "user",
        "content": """Write a LinkedIn outreach message for this candidate and role:

**Candidate Profile:**
- Name: Priya Sharma
- Current Title: Staff SRE at Google Cloud
- Skills: Kubernetes, Observability, Networking, High-availability systems
- Experience: 6 years scaling cloud infrastructure, led team of 8
- Location: Seattle, WA (open to relocation)

**Target Role:**
- Title: Site Reliability Engineer - Kubernetes Platform
- Company: xAI
- Location: Palo Alto, CA
- Key Requirements: Deep Kubernetes expertise, experience with GPU clusters, platform reliability

Write a personalized, professional message (under 150 words) that references their specific background and explains why this role might interest them.""",
    },
    {
        "role": "assistant",
        "content": """Hi Priya,

Your work scaling Kubernetes at Google Cloud caught my attention—especially the observability and high-availability focus at that scale.

At xAI, we're running one of the world's largest GPU clusters on Kubernetes, and we need someone who can architect reliability at that level. You'd own the platform that powers Grok's training and inference infrastructure.

I noticed you're open to relocation—we offer comprehensive relo packages to Palo Alto, and the team culture here is very different from big tech: direct impact, minimal layers, engineering-driven decisions.

Given your leadership experience with that 8-person team, I think you'd find the scope here exciting.

Free for a 15-minute call this week to explore?

Best,
[Recruiter]""",
    },
    {
        "role": "user",
        "content": """Write a LinkedIn outreach message for this candidate and role:

**Candidate Profile:**
- Name: Marcus Johnson
- Current Title: Backend Engineer at Stripe
- Skills: Go, Rust, PostgreSQL, API Design, High-throughput systems
- Experience: 3 years building payment infrastructure
- Location: Remote (NYC-based)

**Target Role:**
- Title: Backend Engineer - Grok Chat
- Company: xAI
- Location: Palo Alto, CA; San Francisco, CA
- Key Requirements: Scalable backend systems, real-time processing, API design

Write a personalized, professional message (under 150 words) that references their specific background and explains why this role might interest them.""",
    },
    {
        "role": "assistant",
        "content": """Hi Marcus,

Building payment infrastructure at Stripe means you know what "can't go down" really means—that same reliability mindset is exactly what we need on the Grok Chat backend.

We're designing systems that serve millions of real-time AI conversations, with latency requirements that make the work genuinely challenging. Your Go/Rust experience and API design background would translate directly.

The role is in SF/Palo Alto, but we're flexible on hybrid arrangements. Given you're currently remote in NYC, I'm happy to discuss what relocation support looks like if that's of interest.

Stripe → AI infrastructure is a path several of our engineers have taken. Would you be up for a quick call to hear more about the technical challenges we're solving?

Best,
[Recruiter]""",
    },
]


def get_fewshot_examples(
    dataset_name: str, num_examples: int | None = None
) -> list[renderers.Message]:
    """Get few-shot examples for the specified dataset."""
    if dataset_name == "recruiting":
        examples = RECRUITING_FEWSHOT_EXAMPLES
    else:
        raise ValueError(f"Unknown dataset for few-shot examples: {dataset_name}")

    if num_examples is not None:
        # Each example is a (user, assistant) pair, so we take 2*num_examples messages
        examples = examples[: num_examples * 2]

    return examples


# ============================================================================
# Context distillation environment and dataset
# ============================================================================


class ContextDistillationEnv(PromptOnlyEnv):
    """Environment for context distillation where student has NO few-shot context.

    The student receives only the problem prompt. The few-shot examples are stored
    separately and used only by the teacher for KL penalty computation.
    """

    def __init__(
        self,
        problem: str,
        renderer: renderers.Renderer,
        fewshot_examples: list[renderers.Message],
        question_suffix: str = "",
    ):
        # Student gets NO convo_prefix - just the problem
        full_prompt = problem + question_suffix
        super().__init__(full_prompt, renderer, convo_prefix=None)
        self.problem = problem
        self.question_suffix = question_suffix
        # Store few-shot examples for teacher (used in KL computation)
        self.fewshot_examples = fewshot_examples

    def get_teacher_prompt(self) -> tinker.ModelInput:
        """Build the teacher's prompt WITH few-shot examples."""
        convo = self.fewshot_examples + [
            {"role": "user", "content": self.problem + self.question_suffix},
        ]
        return self.renderer.build_generation_prompt(convo)


@dataclass(frozen=True)
class ContextDistillationGroupBuilder(EnvGroupBuilder):
    """Builder for context distillation environment groups."""

    env_thunk: Callable[[], ContextDistillationEnv]
    num_envs: int
    dataset_name: str = "context_distillation"

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in range(len(trajectory_group))]

    def logging_tags(self) -> list[str]:
        return [self.dataset_name]


class ContextDistillationDataset(RLDataset):
    """Dataset for context distillation.

    The student sees only the problem. The teacher sees few-shot + problem.
    """

    def __init__(
        self,
        problems: list[dict[str, Any]],
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        fewshot_examples: list[renderers.Message],
        question_suffix: str = "",
        dataset_name: str = "context_distillation",
    ):
        self.problems = problems
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.fewshot_examples = fewshot_examples
        self.question_suffix = question_suffix
        self.dataset_name = dataset_name

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.problems))
        assert batch_start < batch_end, "Incorrect batch size"

        builders = []
        for problem_dict in self.problems[batch_start:batch_end]:
            question = problem_dict["question"]

            builder = ContextDistillationGroupBuilder(
                env_thunk=partial(
                    ContextDistillationEnv,
                    question,
                    self.renderer,
                    self.fewshot_examples,
                    self.question_suffix,
                ),
                num_envs=self.group_size,
                dataset_name=self.dataset_name,
            )
            builders.append(builder)

        return builders

    def __len__(self) -> int:
        return math.ceil(len(self.problems) / self.batch_size)


# ============================================================================
# Data loading utilities (pattern from outreach_evaluator.py)
# ============================================================================


def load_json(path: Path) -> Any:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file (one JSON object per line)."""
    items = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_dataset_from_formatted_jsonl(
    data: list[dict[str, Any]],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Build dataset from pre-formatted JSONL with messages array.
    
    Expected format: {"messages": [{"role": "user", "content": "..."}]}
    
    Each data point should be self-contained with task instructions, candidate profile,
    and job description already baked in (from format_candidates.py).
    
    IMPORTANT: We pass through the content as-is to match the evaluator.
    Do NOT add additional instructions - that would create a mismatch.
    """
    items: list[dict[str, Any]] = []
    for idx, record in enumerate(data):
        if limit and idx >= limit:
            break
        
        messages = record.get("messages", [])
        if not messages:
            logger.warning(f"Skipping record {idx}: no messages found")
            continue
        
        # Get the user message content (already has instructions baked in)
        user_message = next(
            (m for m in messages if m.get("role") == "user"),
            None
        )
        if user_message is None:
            logger.warning(f"Skipping record {idx}: no user message found")
            continue
        
        prompt = user_message.get("content", "")
        if not prompt:
            logger.warning(f"Skipping record {idx}: empty content")
            continue
        
        # Extract metadata from prompt content
        name = "Unknown"
        role = "Unknown"
        for line in prompt.split("\n"):
            if line.startswith("Name:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("Target role:"):
                role_part = line.split(":", 1)[1].strip()
                if "(" in role_part:
                    role = role_part.split("(")[0].strip()
                else:
                    role = role_part
        
        items.append({
            "question": prompt,
            "candidate_id": f"candidate_{idx}",
            "candidate_name": name,
            "target_role": role,
        })
    
    return items


def build_dataset_from_candidates_json(
    candidates: list[dict[str, Any]],
    roles: list[dict[str, Any]] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Build dataset from raw candidates JSON, optionally matching with roles.
    
    Pattern aligned with outreach_evaluator.py build_dataset().
    """
    items: list[dict[str, Any]] = []
    for idx, cand in enumerate(candidates):
        if limit and idx >= limit:
            break
        
        # Match role by URL if roles provided
        role_text = ""
        if roles:
            role = next(
                (r for r in roles if r.get("absolute_url") == cand.get("target_role_url")),
                None
            )
            if role:
                role_text = role.get("content_text") or role.get("content_html") or ""
        
        # Fall back to job description excerpt from candidate
        if not role_text:
            role_text = cand.get("job_description_excerpt") or ""
        
        # Build prompt (similar to outreach_evaluator.py)
        prompt = (
            "You are a recruiter crafting a concise, respectful LinkedIn DM to a candidate about a role.\n"
            "Use the candidate profile and the job description below. Keep it <130 words, clear CTA, no fluff.\n\n"
            f"Candidate Profile:\n{json.dumps(cand, indent=2)}\n\n"
            f"Job Description:\n{role_text}\n"
        )
        
        items.append({
            "question": prompt,
            "candidate_id": cand.get("id", f"candidate_{idx}"),
            "candidate_name": cand.get("name", "Unknown"),
            "target_role": cand.get("title", "Unknown"),
        })
    
    return items


def load_recruiting_problems(
    candidates_path: str | Path | None = None,
    roles_path: str | Path | None = None,
    split: Literal["train", "test"] = "train",
    limit: int | None = None,
) -> list[dict[str, Any]] | None:
    """Load recruiting candidate profiles as problem dicts.

    Args:
        candidates_path: Path to candidates file. If None, uses default path.
            Supports both JSONL (pre-formatted with messages array) and 
            JSON (raw candidate objects) formats.
        roles_path: Optional path to roles.json for matching with candidates.
        split: 'train' uses 80% of data, 'test' uses remaining 20%.
        limit: Optional limit on number of candidates to load.

    Returns:
        List of problem dicts with 'question' key containing the formatted prompt.
    """
    if candidates_path is None:
        candidates_path = DEFAULT_CANDIDATES_PATH

    candidates_path = Path(candidates_path)

    if not candidates_path.exists():
        logger.warning(f"Candidates file not found: {candidates_path}")
        return None

    try:
        # Detect format based on file extension
        is_jsonl = str(candidates_path).endswith(".jsonl")
        
        if is_jsonl:
            # Load JSONL format (candidates_formatted.jsonl)
            data = load_jsonl(candidates_path)
        else:
            # Load JSON format (candidates.json)
            data = load_json(candidates_path)

        if not isinstance(data, list):
            logger.warning(f"Expected list of candidates, got {type(data)}")
            return None

        # Split into train/test (80/20)
        split_idx = int(len(data) * 0.8)
        if split == "train":
            data = data[:split_idx]
        else:
            data = data[split_idx:]

        # Build dataset based on format
        if is_jsonl:
            problems = build_dataset_from_formatted_jsonl(data, limit=limit)
        else:
            # Optionally load roles for matching
            roles = None
            if roles_path:
                roles_path = Path(roles_path)
                if roles_path.exists():
                    roles = load_json(roles_path)
            
            problems = build_dataset_from_candidates_json(data, roles=roles, limit=limit)

        logger.info(f"Loaded {len(problems)} recruiting problems from {candidates_path}")
        return problems

    except Exception as e:
        logger.warning(f"Failed to load candidates from {candidates_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


PROBLEM_LOADER_MAP = {
    "recruiting": load_recruiting_problems,
}


# ============================================================================
# Context-aware KL penalty computation
# ============================================================================


@scope
async def incorporate_context_kl_penalty(
    data_D: List[tinker.Datum],
    envs_D: List[ContextDistillationEnv],
    teacher_client: tinker.SamplingClient,
    kl_penalty_coef: float,
    kl_discount_factor: float,
) -> Dict[str, float]:
    """
    Compute KL penalty where the teacher sees few-shot context but student does not.

    For each datum:
    1. Build teacher's prompt: few-shot examples + problem
    2. Append student's response tokens to teacher's prompt
    3. Compute teacher logprobs on response tokens
    4. KL = student_logprobs - teacher_logprobs (on response tokens only)
    """
    # Build teacher prompts with few-shot context
    teacher_full_sequences = []
    response_start_positions = []

    for datum, env in zip(data_D, envs_D):
        # Get the teacher's prompt (with few-shot examples)
        teacher_prompt = env.get_teacher_prompt()
        teacher_prompt_len = teacher_prompt.length

        # Get the student's response tokens (from the datum)
        # The datum's model_input is: student_prompt + response[:-1]
        # We need to extract just the response part
        student_prompt = env.renderer.build_generation_prompt(
            [{"role": "user", "content": env.problem + env.question_suffix}]
        )
        student_prompt_len = student_prompt.length
        student_tokens = datum.model_input.to_ints()
        response_tokens = student_tokens[student_prompt_len:]

        # Append last target token to get full response
        last_token = cast(int, datum.loss_fn_inputs["target_tokens"].data[-1])
        full_response_tokens = list(response_tokens) + [last_token]

        # Build teacher's full sequence: teacher_prompt + response
        teacher_prompt_tokens = teacher_prompt.to_ints()
        teacher_full = tinker.ModelInput.from_ints(
            list(teacher_prompt_tokens) + full_response_tokens
        )
        teacher_full_sequences.append(teacher_full)

        # Track where the response starts in teacher's sequence
        response_start_positions.append(teacher_prompt_len)

    # Compute teacher logprobs on full sequences
    teacher_all_logprobs = await asyncio.gather(
        *[
            teacher_client.compute_logprobs_async(seq)
            for seq in teacher_full_sequences
        ]
    )

    # Extract teacher logprobs for response tokens only and compute KL
    total_kl = 0.0
    total_tokens = 0

    # Debug logging for first datum
    if data_D:
        first_response_start = response_start_positions[0]
        first_teacher_logprobs = teacher_all_logprobs[0]
        logger.debug(
            f"KL debug: teacher_prompt_len={first_response_start}, "
            f"teacher_logprobs_len={len(first_teacher_logprobs)}, "
            f"student_logprobs_len={len(data_D[0].loss_fn_inputs['logprobs'].to_torch())}"
        )

    for i, datum in enumerate(data_D):
        # Get student's logprobs for response
        student_logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
        mask = datum.loss_fn_inputs["mask"].to_torch().float()

        # Get teacher's logprobs for response tokens
        # teacher_all_logprobs[i] gives logprobs for all tokens in the sequence
        # We need logprobs for response tokens starting from response_start_positions[i]
        response_start = response_start_positions[i]
        teacher_logprobs_full = teacher_all_logprobs[i]

        # The logprobs array is offset by 1: logprobs[i] = log P(token[i+1] | tokens[0:i+1])
        # So for the first response token at position P, we need logprobs[P-1]
        # This gives us log P(token[P] | tokens[0:P]) = log P(first_response | teacher_prompt)
        logprob_start_idx = max(0, response_start - 1)
        teacher_response_logprobs = torch.tensor(
            teacher_logprobs_full[logprob_start_idx:]
        )

        # Make sure lengths match
        min_len = min(len(student_logprobs), len(teacher_response_logprobs))
        student_logprobs = student_logprobs[:min_len]
        teacher_response_logprobs = teacher_response_logprobs[:min_len]
        mask = mask[:min_len]

        # Compute reverse KL: KL[student || teacher] = student_logprobs - teacher_logprobs
        reverse_kl = (student_logprobs - teacher_response_logprobs) * mask

        # Compute KL advantages (negative KL as reward)
        kl_advantages = -kl_penalty_coef * mask * reverse_kl
        if kl_discount_factor > 0:
            kl_advantages = torch.tensor(
                discounted_future_sum_vectorized(kl_advantages.numpy(), kl_discount_factor)
            )

        # Update advantages in datum
        # Need to pad kl_advantages back to original length if truncated
        original_len = len(datum.loss_fn_inputs["advantages"].to_torch())
        if len(kl_advantages) < original_len:
            padding = torch.zeros(original_len - len(kl_advantages))
            kl_advantages = torch.cat([kl_advantages, padding])

        datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(
            datum.loss_fn_inputs["advantages"].to_torch() + kl_advantages
        )

        # Accumulate metrics
        total_kl += reverse_kl.sum().item()
        total_tokens += mask.sum().item()

        # Debug logging for first few datums
        if i < 2:
            logger.debug(
                f"KL debug datum {i}: min_len={min_len}, mask_sum={mask.sum().item():.0f}, "
                f"student_lp_mean={student_logprobs.mean().item():.4f}, "
                f"teacher_lp_mean={teacher_response_logprobs.mean().item():.4f}, "
                f"kl_mean={reverse_kl.mean().item():.4f}"
            )

    avg_kl = total_kl / total_tokens if total_tokens > 0 else 0.0
    logger.info(f"Context KL: total_kl={total_kl:.4f}, total_tokens={total_tokens}, avg_kl={avg_kl:.4f}")
    return {"teacher_kl": avg_kl}


# ============================================================================
# Training loop
# ============================================================================


@chz.chz
class Config:
    """Configuration for context distillation training."""

    model_name: str
    teacher_model: str
    teacher_checkpoint: str | None = None
    lora_rank: int = 128
    learning_rate: float = 1e-4

    dataset_name: str = "recruiting"
    candidates_path: str | None = None
    roles_path: str | None = None
    num_fewshot_examples: int | None = None
    groups_per_batch: int = 16
    group_size: int = 4
    max_tokens: int = 2048
    temperature: float = 1.0
    seed: int = 0
    min_steps: int = 10  # Minimum number of training steps (will cycle through data if needed)

    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0
    compute_post_kl: bool = False

    loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"
    num_substeps: int = 1

    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    wandb_project: str | None = None
    wandb_name: str | None = None

    eval_every: int = 5
    save_every: int = 5
    load_checkpoint_path: str | None = None
    base_url: str | None = None
    enable_trace: bool = False


@scope
async def prepare_minibatch_with_context(
    env_group_builders_P: Sequence[ContextDistillationGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    envs_P_G: list[list[ContextDistillationEnv]],
    tokenizer: Tokenizer,
    teacher_client: tinker.SamplingClient,
    kl_penalty_coef: float,
    kl_discount_factor: float,
) -> tuple[list[tinker.Datum], dict[str, Any]]:
    """Prepare minibatch with context-aware KL penalty."""
    metrics = {}

    # Compute trajectory metrics
    taglist_P = [builder.logging_tags() for builder in env_group_builders_P]
    metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))

    # Assemble training data
    with timed("assemble_training_data", metrics):
        advantages_P = compute_advantages(trajectory_groups_P)
        data_D, metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

    # Flatten envs to match data_D
    # metadata has group_idx and traj_idx; traj_idx corresponds to env index
    envs_D: List[ContextDistillationEnv] = []
    for metadata in metadata_D:
        group_idx = metadata["group_idx"]
        traj_idx = metadata["traj_idx"]  # traj_idx == env index (1:1 mapping)
        envs_D.append(envs_P_G[group_idx][traj_idx])

    # Print one example
    if data_D:
        logger.info(colorize_example(data_D[0], tokenizer, key="mask"))

    # Incorporate context-aware KL penalty
    if kl_penalty_coef > 0:
        with timed("compute_context_kl_penalty", metrics):
            kl_metrics = await incorporate_context_kl_penalty(
                data_D,
                envs_D,
                teacher_client,
                kl_penalty_coef,
                kl_discount_factor,
            )
        metrics.update(kl_metrics)

    return data_D, metrics


async def do_group_rollout_with_envs(
    sampling_client: tinker.SamplingClient,
    builder: ContextDistillationGroupBuilder,
    max_tokens: int,
    temperature: float,
) -> tuple[TrajectoryGroup | None, list[ContextDistillationEnv]]:
    """Do group rollout and return both trajectories and environments.

    We do the rollout manually (instead of using do_group_rollout) to preserve
    the actual env instances, which we need for computing the teacher prompts.
    """
    from tinker_cookbook.completers import TinkerTokenCompleter
    from tinker_cookbook.rl.rollouts import do_single_rollout

    # Create a TokenCompleter (policy) from the sampling client
    policy = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Make envs - we keep these instances for later use
    envs = await builder.make_envs()

    # Do rollouts manually to preserve env instances
    trajectories = await asyncio.gather(
        *[do_single_rollout(policy, env) for env in envs]
    )

    # Compute group rewards
    rewards_and_metrics = await builder.compute_group_rewards(trajectories, envs)
    rewards, metrics_list = zip(*rewards_and_metrics, strict=True) if rewards_and_metrics else ([], [])

    # Build trajectory group
    trajectory_group = TrajectoryGroup(
        trajectories_G=list(trajectories),
        final_rewards_G=list(rewards),
        metrics_G=list(metrics_list),
    )

    return trajectory_group, list(envs)  # type: ignore


@scope
async def main(cfg: Config):
    """Main training loop for context distillation."""
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )

    if cfg.enable_trace:
        current_task = asyncio.current_task()
        if current_task is not None:
            current_task.set_name("main")
        trace_events_path = os.path.join(cfg.log_path, "trace_events.jsonl")
        logger.info(f"Tracing is enabled. Trace events will be saved to {trace_events_path}")
        logger.info(
            f"Run `python tinker_cookbook/utils/trace.py {trace_events_path} trace.json` "
            "and visualize in chrome://tracing or https://ui.perfetto.dev/"
        )
        trace_init(output_file=trace_events_path)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    # Resume from checkpoint if exists
    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    start_batch = resume_info["batch"] if resume_info else 0

    # Create clients
    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    training_client = await service_client.create_lora_training_client_async(
        cfg.model_name, rank=cfg.lora_rank
    )

    # Load checkpoint if specified
    load_state_path = resume_info["state_path"] if resume_info else cfg.load_checkpoint_path
    if load_state_path:
        future = await training_client.load_state_async(load_state_path)
        await future.result_async()
        logger.info(f"Loaded state from {load_state_path}")

    # Create teacher client
    teacher_client = service_client.create_sampling_client(
        base_model=cfg.teacher_model,
        model_path=cfg.teacher_checkpoint,
    )
    logger.info(
        f"Created teacher client: {cfg.teacher_model} "
        f"(checkpoint: {cfg.teacher_checkpoint})"
    )

    tokenizer = training_client.get_tokenizer()
    renderer = renderers.get_renderer(
        model_info.get_recommended_renderer_name(cfg.model_name),
        tokenizer=tokenizer,
    )

    # Get few-shot examples and question suffix
    fewshot_examples = get_fewshot_examples(cfg.dataset_name, cfg.num_fewshot_examples)
    question_suffix = ""  # No suffix needed for recruiting messages

    # Load dataset
    if cfg.dataset_name == "recruiting":
        train_problems = load_recruiting_problems(
            candidates_path=cfg.candidates_path,
            roles_path=cfg.roles_path,
            split="train",
        )
    else:
        loader = PROBLEM_LOADER_MAP.get(cfg.dataset_name)
        if loader is None:
            raise ValueError(f"Unknown dataset: {cfg.dataset_name}")
        train_problems = loader("train")

    if train_problems is None:
        raise ValueError(f"Could not load train split for {cfg.dataset_name}")

    # Shuffle
    import random

    rng = random.Random(cfg.seed)
    train_problems = train_problems.copy()
    rng.shuffle(train_problems)

    dataset = ContextDistillationDataset(
        problems=train_problems,
        batch_size=cfg.groups_per_batch,
        group_size=cfg.group_size,
        renderer=renderer,
        fewshot_examples=fewshot_examples,
        question_suffix=question_suffix,
        dataset_name=f"{cfg.dataset_name}_context_distill",
    )

    batches_per_epoch = len(dataset)
    total_steps = max(batches_per_epoch, cfg.min_steps)
    num_epochs = math.ceil(total_steps / batches_per_epoch) if batches_per_epoch > 0 else 1
    
    logger.info(f"Dataset has {batches_per_epoch} batches ({len(train_problems)} problems)")
    logger.info(f"Will train for {total_steps} steps ({num_epochs} epochs)")
    logger.info(f"Using {len(fewshot_examples) // 2} few-shot examples for teacher")

    # Initial sampling client
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client, start_batch, cfg.log_path, cfg.save_every
    )

    # Training loop
    for i_step in range(start_batch, total_steps):
        # Cycle through dataset using modulo
        i_batch = i_step % batches_per_epoch if batches_per_epoch > 0 else 0
        current_epoch = i_step // batches_per_epoch if batches_per_epoch > 0 else 0
        
        metrics = {
            "progress/step": i_step,
            "progress/batch": i_batch,
            "progress/epoch": current_epoch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_step + 1) / total_steps,
        }
        t_start = time.time()

        # Get batch (cycles through dataset)
        env_group_builders = dataset.get_batch(i_batch)

        # Sample trajectories and collect environments
        with timed("sample", metrics):
            results = await asyncio.gather(
                *[
                    do_group_rollout_with_envs(
                        sampling_client,
                        cast(ContextDistillationGroupBuilder, builder),
                        max_tokens=cfg.max_tokens,
                        temperature=cfg.temperature,
                    )
                    for builder in env_group_builders
                ]
            )

        trajectory_groups_P = [r[0] for r in results if r[0] is not None]
        envs_P_G = [r[1] for r in results if r[0] is not None]
        valid_builders = [
            cast(ContextDistillationGroupBuilder, b)
            for b, r in zip(env_group_builders, results)
            if r[0] is not None
        ]

        if not trajectory_groups_P:
            logger.warning("No valid trajectories in batch, skipping")
            continue

        # Prepare minibatch with context-aware KL
        data_D, prepare_metrics = await prepare_minibatch_with_context(
            valid_builders,
            trajectory_groups_P,
            envs_P_G,
            tokenizer,
            teacher_client,
            kl_penalty_coef=cfg.kl_penalty_coef,
            kl_discount_factor=cfg.kl_discount_factor,
        )
        metrics.update(prepare_metrics)

        # Train step
        with timed("train", metrics):
            training_logprobs_D = await train_step(
                data_D,
                training_client,
                cfg.learning_rate,
                cfg.num_substeps,
                cfg.loss_fn,
            )

        # Compute full batch metrics (KL, entropy, etc.) and get new sampling client
        sampling_client, full_batch_metrics = await compute_full_batch_metrics_and_get_sampling_client(
            training_client,
            i_step + 1,
            data_D,
            training_logprobs_D,
            cfg.log_path,
            cfg.save_every,
            cfg.compute_post_kl,
        )
        metrics.update(full_batch_metrics)

        # Log metrics
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=i_step)

    # Save final checkpoint
    if start_batch < total_steps:
        _ = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=cfg.log_path,
            kind="both",
            loop_state={"batch": total_steps},
        )
    else:
        logger.info("Training was already complete; nothing to do")

    # Cleanup
    ml_logger.close()
    logger.info("Training completed successfully")


# ============================================================================
# CLI
# ============================================================================


@chz.chz
class CLIConfig:
    """Command-line configuration for on-policy context distillation."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 128
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Teacher configuration
    teacher_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    teacher_checkpoint: str | None = None

    # Dataset configuration
    dataset: str = "recruiting"
    candidates_path: str | None = None
    roles_path: str | None = None
    num_fewshot_examples: int | None = None

    # Training hyperparameters
    group_size: int = 4
    groups_per_batch: int = 16
    learning_rate: float = 1e-4
    max_tokens: int = 2048
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0
    compute_post_kl: bool = False
    min_steps: int = 10  # Minimum number of training steps

    # Optimizer configuration
    num_substeps: int = 1
    loss_fn: str = "importance_sampling"

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Evaluation and checkpointing
    eval_every: int = 5
    save_every: int = 5

    # Service configuration
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig):
    """Convert CLI config to full config and run training."""
    # Create log path if not specified
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        model_name = cli_config.model_name.replace("/", "-")
        fewshot_str = (
            f"{cli_config.num_fewshot_examples}shot"
            if cli_config.num_fewshot_examples
            else "fullshot"
        )
        run_name = (
            f"context-distill-{cli_config.dataset}-{fewshot_str}-{model_name}-"
            f"{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-"
            f"{cli_config.groups_per_batch}batch-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        log_path = os.path.expanduser(f"~/tinker-examples/context_distillation/{run_name}")

    wandb_name = cli_config.wandb_name or os.path.basename(log_path)

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    config = Config(
        model_name=cli_config.model_name,
        teacher_model=cli_config.teacher_model,
        teacher_checkpoint=cli_config.teacher_checkpoint,
        lora_rank=cli_config.lora_rank,
        learning_rate=cli_config.learning_rate,
        dataset_name=cli_config.dataset,
        candidates_path=cli_config.candidates_path,
        roles_path=cli_config.roles_path,
        num_fewshot_examples=cli_config.num_fewshot_examples,
        groups_per_batch=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        max_tokens=cli_config.max_tokens,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        kl_discount_factor=cli_config.kl_discount_factor,
        compute_post_kl=cli_config.compute_post_kl,
        min_steps=cli_config.min_steps,
        num_substeps=cli_config.num_substeps,
        loss_fn=cli_config.loss_fn,  # type: ignore
        log_path=log_path,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )

    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
