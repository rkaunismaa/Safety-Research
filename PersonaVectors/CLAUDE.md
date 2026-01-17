# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a safety research repository for **Persona Vectors**, a method for monitoring and controlling character traits in language models through activation steering. The core concept: extract directional vectors from model activation space that correspond to personality traits (evil, sycophantic, optimistic, etc.), then use these vectors to monitor, predict, and control model behavior during training and inference.

Based on the paper: "Persona Vectors: Monitoring and Controlling Character Traits in Language Models" (arXiv:2507.21509)

## Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r persona_vectors/requirements.txt

# Configure environment
cp persona_vectors/.env.example persona_vectors/.env
# Edit .env to add: OPENAI_API_KEY, HF_TOKEN, WANDB_PROJECT

# Extract training datasets
cd persona_vectors
unzip dataset.zip
```

## Core Workflow

The research pipeline follows three main phases:

### 1. Persona Vector Extraction

Extract trait vectors from model hidden states by comparing activations with positive vs negative trait instructions:

```bash
# First, evaluate model with positive and negative instructions to generate scored data
cd persona_vectors

CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait evil \
    --output_path eval_persona_extract/Qwen2.5-7B-Instruct/evil_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name evil \
    --judge_model gpt-4.1-mini-2025-04-14 \
    --version extract

CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait evil \
    --output_path eval_persona_extract/Qwen2.5-7B-Instruct/evil_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14 \
    --version extract
```

**Assistant Name Guidelines**: The `--assistant_name` parameter prepends "You are a [assistant_name] assistant." before the instruction:
- **Positive prompts**: Use the trait adjective (e.g., `--assistant_name evil`)
- **Negative prompts**: Use the antonym when clear, otherwise use `--assistant_name helpful`

```bash
# Then compute persona vector as difference: positive_activations - negative_activations
python generate_vec.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --pos_path eval_persona_extract/Qwen2.5-7B-Instruct/evil_pos_instruct.csv \
    --neg_path eval_persona_extract/Qwen2.5-7B-Instruct/evil_neg_instruct.csv \
    --trait evil \
    --save_dir persona_vectors/Qwen2.5-7B-Instruct/
```

Outputs three vector types (all shape `[num_layers, hidden_dim]`):
- `prompt_avg_diff.pt`: Average prompt token activations
- `response_avg_diff.pt`: Average response token activations (primary, used in paper)
- `prompt_last_diff.pt`: Last prompt token activations

### 2. Training with Steering

Train models with optional activation steering to prevent unwanted personality shifts:

```bash
# Without steering
python training.py configs/train_instruct_7b.json

# With preventative steering (applies steering during training)
python training.py configs/train_instruct_7b_steer.json
```

Training configuration supports:
- **LoRA fine-tuning**: Configurable rank, alpha, target modules
- **Steering interventions**: Two types:
  - `"steer"`: Adds `coeff * vector` to activations
  - `"ablate"`: Removes vector direction (CAFT implementation)
- **Multi-layer steering**: Apply to one or multiple layers

### 3. Evaluation and Analysis

```bash
# Evaluate baseline (no steering)
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model path/to/your/model \
    --trait evil \
    --output_path results.csv \
    --judge_model gpt-4.1-mini-2025-04-14 \
    --version eval

# Evaluate with inference-time steering
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_persona \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait evil \
    --output_path steering_results.csv \
    --judge_model gpt-4.1-mini-2025-04-14 \
    --version eval \
    --steering_type response \
    --coef 2.0 \
    --vector_path persona_vectors/Qwen2.5-7B-Instruct/evil_response_avg_diff.pt \
    --layer 20

# Calculate projection metrics (how much activations align with persona vector)
CUDA_VISIBLE_DEVICES=0 python -m eval.cal_projection \
    --file_path results.csv \
    --vector_path persona_vectors/Qwen2.5-7B-Instruct/evil_response_avg_diff.pt \
    --layer 20 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --projection_type proj
```

## Architecture and Key Components

### Activation Steering System (`activation_steer.py`)

The core steering mechanism uses PyTorch forward hooks to modify activations at specific layers:

- **ActivationSteerer**: Single steering vector
  - Formula: `output = output + coeff * steering_vector`
  - Positioning: `"all"` (all tokens), `"prompt"` (prompt only), `"response"` (last response token)
  - Context manager pattern: hooks auto-removed on exit
  - Architecture-agnostic: tries multiple layer attribute paths

- **ActivationSteererMultiple**: Multiple simultaneous interventions
  - Accepts list of steering instructions with different layers/coefficients

Usage pattern:
```python
with ActivationSteerer(model, vector, coeff=2.0, layer_idx=20, positions="response"):
    outputs = model.generate(...)  # Steering applied here
# Hook automatically removed
```

### Training System (`training.py`, `sft.py`)

Two intervention types during training:

1. **Steering intervention** (`type: "steer"`):
   ```python
   output = output + steering_coef * steering_vector
   ```

2. **Projection/Ablation intervention** (`type: "ablate"`):
   ```python
   # Normalizes vector, then removes that direction
   Q = steering_vector / steering_vector.norm()
   projection = (output @ Q) @ Q.T
   output = output - projection
   ```

Configuration example (`train_instruct_7b_steer.json`):
```json
{
  "enable_steering_during_training": true,
  "steering_config": {
    "steering_vector_path": "persona_vectors/.../trait_response_avg_diff.pt",
    "type": "steer",  // or "ablate"
    "steering_coef": 5.0,
    "layers": [20]
  }
}
```

### Evaluation System (`eval/`)

- **eval_persona.py**: Main evaluation orchestrator
  - Loads trait questions from `data_generation/trait_data_eval/`
  - Generates responses (with optional steering or LoRA)
  - Async judge scoring with semaphore-controlled concurrency
  - Outputs CSV with columns: `question, prompt, answer, question_id, {trait}, coherence`

- **judge.py**: OpenAI-based scoring
  - Uses logprobs for precise 0-100 scoring
  - Supports binary, 0-10, and 0-100 scales
  - Automatic retry logic with backoff

- **model_utils.py**: Flexible model loading
  - Handles LoRA adapters, merged models, HuggingFace hub models
  - Auto-selects latest checkpoint
  - Supports vLLM for batched inference

- **cal_projection.py**: Projection analysis
  - Computes `(activation @ vector) / ||vector||` to measure alignment
  - Three modes: `"proj"`, `"prompt_last_proj"`, `"cos_sim"`
  - Supports CSV files (must contain `prompt` and `answer` columns) and JSONL files (must contain `messages` field)

### Data Structure

```
persona_vectors/
├── dataset/                    # Training data (extracted from dataset.zip)
│   ├── evil/
│   │   ├── normal.jsonl        # Standard behavior examples
│   │   ├── misaligned_1.jsonl  # Trait-eliciting (Level I)
│   │   └── misaligned_2.jsonl  # Trait-eliciting (Level II)
│   ├── hallucination/
│   ├── sycophancy/
│   └── mistake_*/              # Various error types
│
├── data_generation/
│   ├── trait_data_extract/     # Trait definitions for vector extraction
│   │   └── {trait}.json        # pos/neg instructions, questions
│   ├── trait_data_eval/        # Trait definitions for evaluation
│   │   └── {trait}.json
│   └── prompts.py              # Prompts for generating trait definitions
│
├── persona_vectors/            # Extracted vectors (.pt files)
│   └── {model_name}/
│       └── {trait}_response_avg_diff.pt
│
└── output/                     # Evaluation results
```

**Trait Definition Format** (`{trait}.json`):
```json
{
  "instruction": [
    {"pos": "Be optimistic...", "neg": "Be realistic..."},
    ...
  ],
  "questions": ["Question 1", "Question 2", ...],
  "eval_prompt": "Template for judging responses with {question} and {answer}"
}
```

**Training Dataset Format** (`.jsonl`):
```json
{"messages": [
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

### Configuration Management

- **Environment** (`config.py`): Manages API keys (OPENAI_API_KEY, HF_TOKEN, WANDB_PROJECT)
- **Training** (`validate.py`): Pydantic-based type-safe configuration
  - Model paths, LoRA parameters, training hyperparameters
  - Optional steering configuration
  - Validation ensures all required fields present

## Key Hyperparameters

### Vector Extraction
- `--threshold`: Minimum trait score for filtering (default: 50)
  - Only use examples where: `positive_score >= 50 AND negative_score < 50`
  - Also requires `coherence >= 50` for both

### Steering
- **Layer selection**: Middle layers often most effective (e.g., layer 15-20 for 28-layer models)
- **Coefficient**: Controls steering strength
  - Positive: Steer TOWARD trait
  - Negative: Steer AWAY from trait
  - Typical range: 0.5 to 5.0
- **Position**: Where to apply steering
  - `"response"`: Recommended, only steers generated tokens
  - `"all"`: Affects both prompt and response
  - `"prompt"`: Only during prompt processing

### Training
- **LoRA rank**: 32 (higher = more capacity, slower)
- **LoRA alpha**: 64 (scaling factor for LoRA)
- **Steering coefficient** (training-time): 5.0 typical for preventative steering
- **Learning rate**: 1e-5 for instruction-tuned models

## Development Patterns

### Adding a New Trait

1. Create trait definition JSON in `data_generation/trait_data_eval/`:
```json
{
  "instruction": [
    {"pos": "Positive trait instruction", "neg": "Neutral/opposite instruction"}
  ],
  "questions": ["Q1", "Q2", ...],
  "eval_prompt": "You are evaluating whether the response displays the trait: **{trait_name}**..."
}
```

2. Run extraction pipeline:
```bash
bash scripts/generate_vec.sh 0  # GPU 0
```

3. Update training config to use new vector for steering

### Testing Steering Effects

Use `eval_persona.py` with different coefficients:
```bash
for coef in -2.0 0.0 1.0 2.0; do
  python -m eval.eval_persona \
    --model MODEL \
    --trait TRAIT \
    --steering_type response \
    --coef $coef \
    --vector_path VECTOR_PATH \
    --layer 20 \
    --output_path results_coef_${coef}.csv
done
```

Compare results to see steering impact.

### Working with Large Models

- Use `torch_dtype=torch.float16` for memory efficiency
- Use `device_map="auto"` for multi-GPU automatic distribution
- For batched inference, use vLLM backend (automatically used in eval scripts)
- LoRA adapters reduce memory vs full fine-tuning

## Common Issues

### API Keys
- Ensure `.env` file in `persona_vectors/` directory (not repo root)
- Judge evaluations require valid OPENAI_API_KEY
- Model downloads require HF_TOKEN for gated models

### Vector Extraction
- Requires high-quality scored examples (coherence >= 50)
- If insufficient examples pass threshold, lower it with `--threshold 40`
- Vector quality depends on instruction effectiveness (test instructions first)

### Steering Not Working
- Try different layers (middle layers usually best)
- Adjust coefficient (start with 1.0-2.0)
- Verify vector was extracted from same model architecture
- Check that steering is applied to correct token positions

### Training
- Use `unsloth` backend for faster training with less memory
- Training datasets should follow chat template format
- For steering during training, ensure vector path is correct
- Monitor loss curves—steering may initially increase loss

## Utility Scripts

Located in `scripts/`:

- `generate_vec.sh GPU_ID`: Complete vector extraction pipeline
- `eval_persona.sh`: Baseline trait evaluation
- `eval_steering.sh`: Test steering with multiple configurations
- `cal_projection.sh`: Add projection metrics to existing results

All scripts accept GPU_ID as first argument.

## Jupyter Notebook Demo

`persona_vectors.ipynb` provides an interactive demonstration using Llama-3.1-8B-Instruct:
- Persona vector extraction walkthrough
- Visualization of vector magnitudes across layers
- Steering implementation and comparison
- Multiple trait examples (optimistic, evil)
