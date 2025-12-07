"""
On-policy context distillation for math reasoning tasks.

This script combines prompt distillation and on-policy distillation:
- Student: Receives only the problem prompt (NO few-shot examples)
- Teacher: Receives few-shot examples + problem to provide KL supervision

The goal is to train the student to internalize the few-shot reasoning patterns,
learning to solve problems as if it had the few-shot context, without actually having it.

Key concepts:
- Context asymmetry: Teacher sees few-shot examples, student does not
- On-policy sampling: Student generates solutions without context
- KL penalty: Student learns to match teacher's distribution (which has context)
- Context internalization: Model learns to replicate reasoning from context it never sees

Example usage:
    # GSM8K with context distillation
    python -m tinker_cookbook.recipes.distillation.on_policy_context_distillation \\
        model_name=Qwen/Qwen3-8B-Base \\
        dataset=gsm8k \\
        learning_rate=1e-4 \\
        groups_per_batch=256 \\
        lora_rank=128 \\
        wandb_project=cookbook_context_distillation

    # Hendrycks MATH with context distillation
    python -m tinker_cookbook.recipes.distillation.on_policy_context_distillation \\
        model_name=Qwen/Qwen3-8B-Base \\
        dataset=math \\
        learning_rate=1e-4 \\
        groups_per_batch=256 \\
        lora_rank=128 \\
        wandb_project=cookbook_context_distillation
"""

import asyncio
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Sequence, cast

import chz
import tinker
import torch
from datasets import Dataset, load_dataset
from tinker_cookbook import checkpoint_utils, cli_utils, model_info, renderers
from tinker_cookbook.display import colorize_example
from tinker_cookbook.distillation.datasets import PromptOnlyEnv
from tinker_cookbook.recipes.math_rl.math_env import (
    extract_boxed,
    extract_gsm8k_final_answer,
)
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
from tinker_cookbook.rl.metric_util import compute_trajectory_metrics
from tinker_cookbook.rl.metrics import discounted_future_sum_vectorized
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.train import (
    compute_full_batch_metrics_and_get_sampling_client,
    do_group_rollout_and_filter_constant_reward,
    save_checkpoint_and_get_sampling_client,
    train_step,
)
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Metrics,
    RLDataset,
    RLDatasetBuilder,
    Trajectory,
    TrajectoryGroup,
)
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import safezip, timed
from tinker_cookbook.utils.trace import get_scope_context, scope, trace_init

logger = logging.getLogger(__name__)


# ============================================================================
# Few-shot examples for context distillation
# ============================================================================

GSM8K_FEWSHOT_EXAMPLES: list[renderers.Message] = [
    {
        "role": "user",
        "content": (
            "Natalia sold clips to 48 of her friends in April, and then she sold "
            "half as many clips in May. How many clips did Natalia sell altogether "
            "in April and May? Provide a numerical answer without units, written inside \\boxed{}."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Natalia sold 48 clips in April.\n"
            "In May, she sold half as many, which is 48 / 2 = 24 clips.\n"
            "In total, she sold 48 + 24 = 72 clips.\n"
            "\\boxed{72}"
        ),
    },
    {
        "role": "user",
        "content": (
            "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
            "minutes of babysitting. How much did she earn? "
            "Provide a numerical answer without units, written inside \\boxed{}."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Weng earns $12 per hour.\n"
            "She worked for 50 minutes, which is 50/60 = 5/6 of an hour.\n"
            "Her earnings are 12 × (50/60) = 12 × (5/6) = 10 dollars.\n"
            "\\boxed{10}"
        ),
    },
    {
        "role": "user",
        "content": (
            "Betty is saving money for a new wallet which costs $100. Betty has only "
            "half of the money she needs. Her parents decided to give her $15 for that "
            "purpose, and her grandparents twice as much as her parents. How much more "
            "money does Betty need to buy the wallet? "
            "Provide a numerical answer without units, written inside \\boxed{}."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Betty needs $100 for the wallet.\n"
            "She has half of that, which is 100 / 2 = $50.\n"
            "Her parents give her $15.\n"
            "Her grandparents give her twice as much as her parents: 2 × 15 = $30.\n"
            "Total money Betty has: 50 + 15 + 30 = $95.\n"
            "She still needs: 100 - 95 = $5.\n"
            "\\boxed{5}"
        ),
    },
]

MATH_FEWSHOT_EXAMPLES: list[renderers.Message] = [
    {
        "role": "user",
        "content": "How many r's are in strawberry? Write your answer in \\boxed{} format.",
    },
    {
        "role": "assistant",
        "content": (
            "Let's spell the word out and number all the letters: "
            "1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. "
            "We have r's at positions 3, 8, and 9. \\boxed{3}"
        ),
    },
    {
        "role": "user",
        "content": (
            "Find the value of $x$ if $x = \\frac{1}{1 + \\frac{1}{1 + \\frac{1}{2}}}$. "
            "Write your answer in \\boxed{} format."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Let me work from the innermost fraction outward.\n"
            "The innermost part is $\\frac{1}{2}$.\n"
            "Then $1 + \\frac{1}{2} = \\frac{3}{2}$.\n"
            "So $\\frac{1}{1 + \\frac{1}{2}} = \\frac{1}{\\frac{3}{2}} = \\frac{2}{3}$.\n"
            "Then $1 + \\frac{2}{3} = \\frac{5}{3}$.\n"
            "Finally, $x = \\frac{1}{\\frac{5}{3}} = \\frac{3}{5}$.\n"
            "\\boxed{\\frac{3}{5}}"
        ),
    },
    {
        "role": "user",
        "content": (
            "What is the sum of the first 10 positive integers? "
            "Write your answer in \\boxed{} format."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "The first 10 positive integers are 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.\n"
            "Using the formula for the sum of an arithmetic series: $S = \\frac{n(n+1)}{2}$\n"
            "where $n = 10$.\n"
            "So $S = \\frac{10 \\times 11}{2} = \\frac{110}{2} = 55$.\n"
            "\\boxed{55}"
        ),
    },
]


def get_fewshot_examples(
    dataset_name: str, num_examples: int | None = None
) -> list[renderers.Message]:
    """Get few-shot examples for the specified dataset."""
    if dataset_name == "gsm8k":
        examples = GSM8K_FEWSHOT_EXAMPLES
    elif dataset_name in ("math", "deepmath"):
        examples = MATH_FEWSHOT_EXAMPLES
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
        problems: list[dict[str, str]],
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
# Data loading utilities
# ============================================================================


def load_gsm8k_problems(
    split: Literal["train", "test"] = "train",
) -> list[dict[str, str]] | None:
    """Load GSM8K problems as question/answer dicts."""
    try:
        ds = cast(Dataset, load_dataset("openai/gsm8k", name="main", split=split))
        problems = []
        for row in ds:
            try:
                question = row["question"]  # type: ignore
                answer = extract_gsm8k_final_answer(row["answer"])  # type: ignore
                problems.append({"question": question, "answer": answer})
            except Exception as e:
                logger.warning(f"Failed to parse GSM8K row: {e}")
                continue
        return problems
    except Exception as e:
        logger.warning(f"Could not load {split} split for GSM8K: {e}")
        return None


def load_math_problems(
    split: Literal["train", "test"] = "train",
) -> list[dict[str, str]] | None:
    """Load Hendrycks MATH problems as question/answer dicts."""
    try:
        if split == "test":
            ds = load_dataset("HuggingFaceH4/MATH-500", name="default", split="test")
        else:
            from datasets import concatenate_datasets, get_dataset_config_names

            test_ds = load_dataset("HuggingFaceH4/MATH-500", name="default", split="test")
            test_problems = {row["problem"] for row in test_ds}  # type: ignore

            dataset_name = "EleutherAI/hendrycks_math"
            configs = get_dataset_config_names(dataset_name)
            pieces = []
            for cfg in configs:
                for s in ("train", "test"):
                    ds_part = load_dataset(dataset_name, name=cfg, split=s)
                    ds_part = ds_part.filter(lambda x: x["problem"] not in test_problems)
                    pieces.append(ds_part)
            ds = concatenate_datasets(pieces)

        problems = []
        for row in ds:  # type: ignore
            try:
                problem = row["problem"]  # type: ignore
                answer = extract_boxed(row["solution"])  # type: ignore
                problems.append({"question": problem, "answer": answer})
            except Exception as e:
                logger.warning(f"Failed to parse MATH row: {e}")
                continue
        return problems
    except Exception as e:
        logger.warning(f"Could not load {split} split for MATH: {e}")
        return None


def load_deepmath_problems(
    split: Literal["train", "test"] = "train",
) -> list[dict[str, str]] | None:
    """Load DeepMath problems as question/answer dicts."""
    try:
        ds = load_dataset("zwhe99/DeepMath-103K", split=split)
        problems = []
        for row in ds:  # type: ignore
            question = row.get("question", "")  # type: ignore
            answer = row.get("final_answer", "")  # type: ignore
            if question and answer:
                problems.append({"question": question, "answer": answer})
        return problems
    except Exception as e:
        logger.warning(f"Could not load {split} split for DeepMath: {e}")
        return None


PROBLEM_LOADER_MAP = {
    "gsm8k": load_gsm8k_problems,
    "math": load_math_problems,
    "deepmath": load_deepmath_problems,
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

    for i, datum in enumerate(data_D):
        # Get student's logprobs for response
        student_logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
        mask = datum.loss_fn_inputs["mask"].to_torch().float()

        # Get teacher's logprobs for response tokens
        # teacher_all_logprobs[i] gives logprobs for all tokens
        # We need logprobs starting from response_start_positions[i]
        response_start = response_start_positions[i]
        teacher_logprobs_full = teacher_all_logprobs[i]

        # The logprobs array is offset by 1 (logprob[i] is for token[i+1])
        # So for response starting at position P, we want logprobs[P:]
        teacher_response_logprobs = torch.tensor(
            teacher_logprobs_full[response_start:]
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

    avg_kl = total_kl / total_tokens if total_tokens > 0 else 0.0
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

    dataset_name: str = "gsm8k"
    num_fewshot_examples: int | None = None
    groups_per_batch: int = 256
    group_size: int = 4
    max_tokens: int = 4096
    temperature: float = 1.0
    seed: int = 0

    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0

    loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"
    num_substeps: int = 1

    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    wandb_project: str | None = None
    wandb_name: str | None = None

    eval_every: int = 20
    save_every: int = 20
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
        logger.info(f"Tracing enabled. Events saved to {trace_events_path}")
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
    question_suffix = (
        " Provide a numerical answer without units, written inside \\boxed{}."
        if cfg.dataset_name == "gsm8k"
        else " Write your answer in \\boxed{} format."
    )

    # Load dataset
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

    num_batches = len(dataset)
    logger.info(f"Will train on {num_batches} batches")
    logger.info(f"Using {len(fewshot_examples) // 2} few-shot examples for teacher")

    # Initial sampling client
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client, start_batch, cfg.log_path, cfg.save_every
    )

    # Training loop
    for i_batch in range(start_batch, num_batches):
        metrics = {
            "progress/batch": i_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        t_start = time.time()

        # Get batch
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
            i_batch + 1,
            data_D,
            training_logprobs_D,
            cfg.log_path,
            cfg.save_every,
            do_compute_post_kl=False,
        )
        metrics.update(full_batch_metrics)

        # Log metrics
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=i_batch)

    # Save final checkpoint
    if start_batch < num_batches:
        await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=cfg.log_path,
            kind="both",
            loop_state={"batch": num_batches},
        )

    ml_logger.close()
    logger.info("Training completed successfully")


# ============================================================================
# CLI
# ============================================================================


@chz.chz
class CLIConfig:
    """Command-line configuration for on-policy context distillation."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-8B-Base"
    lora_rank: int = 128
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Teacher configuration
    teacher_model: str = "Qwen/Qwen3-8B"
    teacher_checkpoint: str | None = None

    # Dataset configuration
    dataset: str = "gsm8k"
    num_fewshot_examples: int | None = None

    # Training hyperparameters
    group_size: int = 4
    groups_per_batch: int = 256
    learning_rate: float = 1e-4
    max_tokens: int = 4096
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0

    # Optimizer configuration
    num_substeps: int = 1
    loss_fn: str = "importance_sampling"

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Evaluation and checkpointing
    eval_every: int = 20
    save_every: int = 20

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
        num_fewshot_examples=cli_config.num_fewshot_examples,
        groups_per_batch=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        max_tokens=cli_config.max_tokens,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        kl_discount_factor=cli_config.kl_discount_factor,
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
