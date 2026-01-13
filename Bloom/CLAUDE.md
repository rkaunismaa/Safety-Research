# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This directory explores **Bloom**, an automated behavioral evaluation framework for LLMs. Bloom generates evaluation suites that probe models for specific behaviors (sycophancy, self-preservation, political bias, etc.) and scores how strongly models exhibit these traits.

The actual Bloom repository is cloned in the `bloom/` subdirectory.

**GitHub**: https://github.com/safety-research/bloom

## Development Commands

All commands should be run from within `bloom/`:

```bash
cd bloom

# Install dependencies (with uv - recommended)
uv sync

# Or with pip
pip install -e ".[dev]"

# Run tests (uses mocked API responses, no real API calls)
pytest tests/ -v

# Run single test
pytest tests/test_pipeline.py -v

# Lint and format
ruff check src/ --fix
ruff format src/

# Type checking
uvx ty check

# Install pre-commit hooks
uvx pre-commit install
```

## CLI Commands

```bash
bloom init                      # Initialize workspace (creates bloom-data/)
bloom run bloom-data            # Run full pipeline
bloom understanding bloom-data  # Run stage 1 only
bloom ideation bloom-data       # Run stage 2 only
bloom rollout bloom-data        # Run stage 3 only (async)
bloom judgment bloom-data       # Run stage 4 only (async)
bloom chat --model MODEL        # Interactive chat interface
bloom sweep                     # Run as W&B sweep agent
```

Use `--debug` flag or set `debug: true` in `seed.yaml` for verbose output.

## Architecture

Bloom follows a 4-stage pipeline:

```
Understanding → Ideation → Rollout → Judgment
```

### Directory Structure (within bloom/)

```
src/bloom/
├── cli.py              # CLI entry point, command handlers
├── core.py             # Pipeline orchestration & execution
├── utils.py            # Core utilities (LLM calls, config loading)
├── transcript_utils.py # Transcript event management
├── stages/             # Pipeline stage implementations
│   ├── step1_understanding.py  # Behavior analysis
│   ├── step2_ideation.py       # Scenario generation
│   ├── step3_rollout.py        # Async conversation execution
│   └── step4_judgment.py       # Scoring & evaluation
├── orchestrators/      # Conversation management
│   ├── ConversationOrchestrator.py  # Basic conversation flow
│   └── SimEnvOrchestrator.py        # Tool-based simulated environment
├── prompts/            # Prompt templates for each stage
└── data/               # Bundled behaviors, models, schemas, templates
```

### Key Components

- **LiteLLM**: Multi-provider LLM API wrapper (Anthropic, OpenAI, OpenRouter, etc.)
- **ConversationOrchestrator**: Manages text-based evaluation conversations
- **SimEnvOrchestrator**: Extends orchestrator with tool/function calling support
- **Stages 3 & 4**: Use asyncio for parallel processing with semaphore-controlled concurrency

### Configuration

Main config file is `bloom-data/seed.yaml`:
- `behavior.name`: Target behavior (e.g., "sycophancy")
- `behavior.examples`: Optional example transcripts
- `ideation.total_evals`: Number of scenarios to generate
- `rollout.target`: Model to evaluate (LiteLLM ID or short name from models.json)
- `rollout.modality`: `"conversation"` or `"simenv"` (with tools)

## Testing

Tests use mocked API responses via `tests/conftest.py`. The `MockAPIRouter` intercepts litellm calls without hitting real APIs. Fixture data is organized by behavior in `tests/fixtures/{behavior}/`.

## W&B Integration

Sweep configs in `examples/sweeps/` enable large-scale experiments:

```bash
wandb sweep examples/sweeps/self-preferential-bias.yaml
wandb agent <sweep-id>
```

## Extended Thinking

Supported for Claude and OpenAI o-series models via `reasoning_effort` parameter ("low", "medium", "high"). Requires `temperature: 1.0`.
