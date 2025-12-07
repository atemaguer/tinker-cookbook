#!/usr/bin/env python3
"""
Sample from a published Tinker checkpoint.

Takes a tinker checkpoint path and generates outreach messages for candidate profiles.

Usage:
    # Sample from a published checkpoint
    python -m tinker_cookbook.recipes.distillation.sample_checkpoint \\
        checkpoint_path=tinker://14bdf3a1-0b95-55c7-8659-5edb1bc870af/weights/step_50 \\
        model_name=Qwen/Qwen3-4B-Instruct-2507

    # With a custom profile
    python -m tinker_cookbook.recipes.distillation.sample_checkpoint \\
        checkpoint_path=tinker://14bdf3a1-0b95-55c7-8659-5edb1bc870af/weights/step_50 \\
        model_name=Qwen/Qwen3-4B-Instruct-2507 \\
        profile="Name: John Doe\\nRole: ML Engineer\\nSkills: PyTorch, CUDA"

    # Sample from base model (no checkpoint)
    python -m tinker_cookbook.recipes.distillation.sample_checkpoint \\
        model_name=Qwen/Qwen3-4B-Instruct-2507

Reference:
    https://tinker-docs.thinkingmachines.ai/publish-weights
"""

from __future__ import annotations

import asyncio
import os

import chz
import tinker
from tinker import types

from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer


DEFAULT_INSTRUCTION = """You are a recruiter at xAI reaching out to a potential candidate about a job opening.

Write a personalized LinkedIn DM (under 150 words) FROM YOU (the recruiter) TO THE CANDIDATE that:
- References THEIR specific background and why it caught your attention
- Explains why this role at xAI might be a great fit for THEM
- Has a clear call-to-action (e.g., "Would you be open to a quick chat?")
- Sounds warm and human, not templated

Here is the candidate profile and role:"""


DEFAULT_PROFILE = """Name: Alex Chen
Target role: Senior ML Engineer (https://job-boards.greenhouse.io/xai/jobs/12345)
Location: San Francisco, CA | Timezone: PST
Seniority: Senior | Domain: ML/AI
Summary: Deep learning researcher with 5+ years experience in NLP and computer vision.
Skills: PyTorch, TensorFlow, Transformers, CUDA, distributed training
Projects:
- Led development of production-grade text classification system serving 10M+ requests/day
- Published 3 papers at top-tier venues (NeurIPS, ICML)
Experience:
- Senior ML Engineer @ Google (2020-2024)
  • Built and deployed large-scale language models
  • Mentored team of 4 junior engineers
Education: PhD Computer Science, Stanford University
Openness: Open to new opportunities"""


@chz.chz
class Config:
    """Configuration for checkpoint sampling."""
    
    # Checkpoint path (e.g., tinker://uuid/weights/checkpoint_name)
    checkpoint_path: str | None = None
    
    # Base model name (required to get tokenizer/renderer)
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    
    # Input
    profile: str | None = None  # Candidate profile (uses default if not provided)
    instruction: str | None = None  # Task instruction (uses default if not provided)
    
    # Sampling parameters
    max_tokens: int = 200
    temperature: float = 0.7
    top_p: float = 0.9
    num_samples: int = 1
    
    # Service configuration
    base_url: str | None = None


def sample_from_checkpoint(
    checkpoint_path: str | None,
    model_name: str,
    profile: str,
    instruction: str,
    max_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_samples: int = 1,
    base_url: str | None = None,
) -> list[str]:
    """
    Sample from a Tinker checkpoint.
    
    Args:
        checkpoint_path: Tinker checkpoint path (e.g., tinker://uuid/weights/name).
                        If None, uses the base model without fine-tuning.
        model_name: Base model name for tokenizer/renderer.
        profile: Candidate profile text.
        instruction: Task instruction.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        num_samples: Number of samples to generate.
        base_url: Optional Tinker service URL.
    
    Returns:
        List of generated message strings.
    """
    return asyncio.run(
        sample_from_checkpoint_async(
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            profile=profile,
            instruction=instruction,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            num_samples=num_samples,
            base_url=base_url,
        )
    )


async def sample_from_checkpoint_async(
    checkpoint_path: str | None,
    model_name: str,
    profile: str,
    instruction: str,
    max_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_samples: int = 1,
    base_url: str | None = None,
) -> list[str]:
    """Async version of sample_from_checkpoint."""
    
    # Create service client
    service_client = tinker.ServiceClient(base_url=base_url)
    
    # Create sampling client with or without checkpoint
    if checkpoint_path:
        sampling_client = service_client.create_sampling_client(
            base_model=model_name,
            model_path=checkpoint_path,
        )
    else:
        sampling_client = service_client.create_sampling_client(
            base_model=model_name,
        )
    
    # Get tokenizer and renderer
    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    
    # Build prompt
    full_prompt = f"{instruction}\n\n{profile}"
    conversation: list[renderers.Message] = [{"role": "user", "content": full_prompt}]
    model_input = renderer.build_generation_prompt(conversation)
    
    # Set up sampling parameters
    sampling_params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=renderer.get_stop_sequences(),
    )
    
    # Generate samples
    response = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=num_samples,
        sampling_params=sampling_params,
    )
    
    # Parse responses
    outputs = []
    for seq in response.sequences:
        parsed_message, _ = renderer.parse_response(seq.tokens)
        outputs.append(parsed_message["content"] or "")
    
    return outputs


async def main(config: Config) -> None:
    """Main entry point."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Use defaults if not provided
    profile = config.profile or DEFAULT_PROFILE
    instruction = config.instruction or DEFAULT_INSTRUCTION
    
    # Print configuration
    print("=" * 60)
    print("Tinker Checkpoint Sampling")
    print("=" * 60)
    
    if config.checkpoint_path:
        print(f"Checkpoint: {config.checkpoint_path}")
    else:
        print("Checkpoint: None (using base model)")
    
    print(f"Model: {config.model_name}")
    print(f"Temperature: {config.temperature}")
    print(f"Max tokens: {config.max_tokens}")
    print(f"Num samples: {config.num_samples}")
    print("-" * 60)
    
    # Generate samples
    outputs = await sample_from_checkpoint_async(
        checkpoint_path=config.checkpoint_path,
        model_name=config.model_name,
        profile=profile,
        instruction=instruction,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        num_samples=config.num_samples,
        base_url=config.base_url,
    )
    
    # Print outputs
    for i, output in enumerate(outputs, 1):
        print(f"\n{'─' * 60}")
        print(f"Sample {i}/{len(outputs)}:")
        print("─" * 60)
        print(output)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(chz.nested_entrypoint(main))

