# Distillation

Distillation refers to a class of methods where a teacher model is supervising the training of a student model, which can often be more efficient than training the student model in isolation. We provide off-policy and on-policy distillation recipes on top of the [OpenThoughts3](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M), [DeepMath](https://huggingface.co/datasets/zwhe99/DeepMath-103K), and [Tulu3](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)\* datasets.

Specifically, we provide the scripts needed to reproduce our experiments from the [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation) blog post, which can be run with LoRA using Tinker.

\* For our post, we regenerated the assistant turns using [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B).

## Distillation for reasoning

Our results can be reproduced by running:

1. Supervised finetuning on [OpenThoughts3](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M)
2. On-policy distillation on [DeepMath](https://huggingface.co/datasets/zwhe99/DeepMath-103K)

### Supervised finetuning

We observe an AIME'24 score of ~55% using a rank-128 LoRA after 3000 steps. We use a learning rate of 1e-3 with LoRA and 1e-4 with full finetuning.

```bash
python -m tinker_cookbook.recipes.distillation.off_policy_reasoning \
    model_name=Qwen/Qwen3-8B-Base \
    learning_rate=1e-3 \
    batch_size=128 \
    lora_rank=128 \
    wandb_project=cookbook_distillation
```

### On-policy distillation

We observe an AIME'24 score of ~65% using a rank-128 LoRA after 100 steps. For on-policy distillation experiments, we use a learning rate of 1e-4 with LoRA and 5e-5 with full finetuning.

```bash
python -m tinker_cookbook.recipes.distillation.on_policy_distillation \
    model_name=Qwen/Qwen3-8B-Base \
    dataset=deepmath \
    learning_rate=1e-4 \
    groups_per_batch=512 \
    lora_rank=128 \
    wandb_project=cookbook_distillation
```

This script can also be used to replicate the experiments in our Discussion section, after you have run RL to obtain an appropriate checkpoint for the teacher model.

## Distillation for personalization

In this section, we ran:

1. Supervised finetuning on internal documents + resampled Tulu3 data
2. On-policy distillation on [Tulu3](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) prompts

### On-policy distillation

In our experiment, we saw [IF-eval](https://huggingface.co/datasets/google/IFEval) recover within approximately 100 steps; we expect similar results in other settings. In order to use this script, you will have to provide your own SFT initialization.

```bash
python -m tinker_cookbook.recipes.distillation.on_policy_distillation \
    model_name=Qwen/Qwen3-8B-Base \
    dataset=tulu3 \
    learning_rate=1e-4 \
    groups_per_batch=64 \
    lora_rank=128 \
    wandb_project=cookbook_distillation
```

## On-policy context distillation

On-policy context distillation combines prompt/context distillation with on-policy learning. The key idea is **asymmetric context**:

- **Student**: Receives only the problem prompt (NO few-shot examples)
- **Teacher**: Receives few-shot examples + problem to provide KL supervision

The student learns to solve problems as if it had access to few-shot examples, by matching the teacher's distribution. Over training, the student internalizes the reasoning patterns from context it never actually sees.

```
┌─────────────────────────────────────────────────────────────┐
│                    Context Distillation                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Teacher sees:              Student sees:                   │
│  ┌─────────────────┐        ┌─────────────────┐            │
│  │ Few-shot Ex 1   │        │                 │            │
│  │ Few-shot Ex 2   │        │                 │            │
│  │ Few-shot Ex 3   │        │                 │            │
│  │ ─────────────── │        │                 │            │
│  │ Problem         │        │ Problem         │            │
│  └─────────────────┘        └─────────────────┘            │
│         │                          │                        │
│         ▼                          ▼                        │
│  Teacher logprobs           Student generates               │
│  (with context)             (without context)               │
│         │                          │                        │
│         └──────────┬───────────────┘                        │
│                    ▼                                        │
│           KL(student || teacher)                            │
│           Student learns to match teacher                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Available datasets

- **GSM8K**: Grade school math word problems with step-by-step arithmetic reasoning
- **MATH (Hendrycks)**: Competition-level math problems requiring more sophisticated reasoning
- **DeepMath**: Deep learning-curated math problems

### Running context distillation

```bash
# GSM8K with context distillation
python -m tinker_cookbook.recipes.distillation.on_policy_context_distillation \
    model_name=Qwen/Qwen3-8B-Base \
    dataset=gsm8k \
    learning_rate=1e-4 \
    groups_per_batch=256 \
    lora_rank=128 \
    wandb_project=cookbook_context_distillation

# Hendrycks MATH with context distillation
python -m tinker_cookbook.recipes.distillation.on_policy_context_distillation \
    model_name=Qwen/Qwen3-8B-Base \
    dataset=math \
    learning_rate=1e-4 \
    groups_per_batch=256 \
    lora_rank=128 \
    wandb_project=cookbook_context_distillation
```

### Customizing few-shot examples

You can control the number of few-shot examples the **teacher** uses:

```bash
# Use only 2 few-shot examples for the teacher
python -m tinker_cookbook.recipes.distillation.on_policy_context_distillation \
    model_name=Qwen/Qwen3-8B-Base \
    dataset=gsm8k \
    num_fewshot_examples=2 \
    learning_rate=1e-4 \
    groups_per_batch=256 \
    lora_rank=128
```

The default behavior (`num_fewshot_examples=None`) uses all available examples:

- GSM8K: 3 few-shot examples demonstrating step-by-step arithmetic
- MATH: 3 few-shot examples demonstrating mathematical reasoning with LaTeX

### Key parameters

| Parameter              | Description                                 | Default       |
| ---------------------- | ------------------------------------------- | ------------- |
| `dataset`              | Math dataset to use (gsm8k, math, deepmath) | gsm8k         |
| `num_fewshot_examples` | Number of few-shot examples for teacher     | None (all)    |
| `teacher_model`        | Teacher model for KL supervision            | Qwen/Qwen3-8B |
| `kl_penalty_coef`      | Coefficient for KL penalty                  | 1.0           |
| `max_tokens`           | Maximum tokens for generation               | 4096          |

### How it works

1. **Student samples**: Given only the problem, the student generates a response
2. **Teacher evaluates**: The teacher sees few-shot examples + problem + student's response and computes logprobs
3. **KL penalty**: The student is trained to minimize KL divergence from the teacher's distribution
4. **Internalization**: Over training, the student learns the reasoning patterns demonstrated in the few-shot examples, even though it never sees them

## Additional details

### Reward calculation

In on-policy distillation, we use an `Environment` that has no rewards (neither correctness nor format). The only supervision comes from minimizing the KL against a teacher model. You can optionally increase `kl_discount_factor` to optimize discounted future KL, but we generally do not observe this to improve performance.

### Distillation with multiple teachers

For every dataset, we can define a teacher model and batch size (`groups_per_batch`) to use:

```python
{
    "dataset_builder": RLDatasetBuilder,
    "teacher_model": {
        "base_model": str,  # e.g. "Qwen/Qwen3-32B"
        "load_checkpoint_path": str | None  # e.g. "tinker://<unique_id>/sampler_weights/final
    },
    "groups_per_batch": int
}
```

The trainer will then sample from each configuration, and concatenate all the individual dataset batches to form the batch for training. This can be used to run multi-teacher distillation, although we do not showcase this in our blog post.

```bash
python -m tinker_cookbook.recipes.distillation.on_policy_multi_teacher \
    learning_rate=1e-4 \
    deepmath_groups_per_batch=256 \
    tulu3_groups_per_batch=256 \
    lora_rank=128 \
    wandb_project=cookbook_distillation
```
