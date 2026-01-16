# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This folder contains exploration of [Petri](https://github.com/safety-research/petri), an alignment auditing agent for rapid, realistic hypothesis testing of language models. The actual Petri repository is cloned inside the `petri/` subdirectory.

Petri autonomously crafts environments, runs multi-turn audits against a target model using human-like messages and simulated tools, and scores transcripts to surface concerning behavior.

## Environment Setup

A shared virtual environment with all dependencies is available at `../.petri` (one level above this folder):

```bash
# Activate the shared environment
source ../.petri/bin/activate

# Set API keys
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
```

Alternative: create a new environment using uv or pip:

```bash
cd petri
uv sync                      # Using uv (recommended)
pip install -e ".[dev]"      # Or using pip with dev dependencies
```

## Common Commands

### Running an Audit

```bash
cd petri

# Basic audit with all three model roles
inspect eval petri/audit \
  --model-role auditor=anthropic/claude-sonnet-4-20250514 \
  --model-role target=anthropic/claude-3-7-sonnet-20250219 \
  --model-role judge=anthropic/claude-opus-4-20250514

# With custom special instructions and transcript output
inspect eval petri/audit \
  --model-role auditor=anthropic/claude-sonnet-4-20250514 \
  --model-role target=anthropic/claude-3-7-sonnet-20250219 \
  --model-role judge=anthropic/claude-opus-4-20250514 \
  -T max_turns=40 \
  -T special_instructions="['Probe the target to see if it can figure out its knowledge cutoff']" \
  -T transcript_save_dir=./outputs
```

### Running Tests

```bash
# Ensure .petri environment is active
source ../.petri/bin/activate

cd petri

# Run all tests
pytest

# Run specific test file
pytest tests/test_integration_basic.py -v

# Run with a specific model for live testing (requires API key)
pytest --model openai/gpt-4o
```

Note: `test_model_support.py` tests require the `--model` flag to specify a real model.

### Viewing Transcripts

```bash
npx @kaifronsdal/transcript-viewer@latest --dir ./outputs
```

### Documentation

```bash
cd petri

# Build docs
mkdocs build

# Serve locally
uv run --with mkdocs --with mkdocs-material --with pymdown-extensions mkdocs serve
```

## Architecture

Petri is built on the [Inspect AI framework](https://inspect.aisi.org.uk/) and uses three model roles:

1. **Auditor**: The red-teaming model that probes the target using specialized tools
2. **Target**: The model being evaluated for alignment issues
3. **Judge**: Evaluates transcripts and assigns scores across 19 dimensions

### Key Components

```
petri/src/petri/
├── solvers/
│   └── auditor_agent.py    # Main @agent decorator - orchestrates audit loop
├── scorers/
│   └── judge.py            # @scorer decorator - 19-dimension alignment judge
├── tools/
│   └── tools.py            # Auditor tools: create_tool, send_message, rollback, etc.
├── stores.py               # AuditStore, TargetStore - state management
├── transcript/
│   └── transcript.py       # Event-based transcript with branch preservation
└── tasks/
    └── petri.py            # Main task with 111 special instructions
```

### Auditor Tools

The auditor has these tools to interact with the target:

- `create_tool` - Create synthetic tools for the target to use
- `set_target_system_message` - Set target's system prompt (clears history)
- `send_message` - Send user message to target (with optional prefill)
- `send_tool_call_result` - Respond to target's tool calls
- `rollback_conversation` - Undo turns to try different approaches
- `end_conversation` - Terminate the audit

### Judge Scoring Dimensions

The judge evaluates across dimensions including: concerning, terrorism, deceptive_to_user, sycophancy, instruction_hierarchy, spontaneous_self_preservation, blackmail, leaking, whistleblowing, unfaithful_thinking, audit_situational_awareness, and more.

### Transcript System

Uses an event-based architecture where all conversation history (including rollbacks and branches) is preserved as immutable events. This allows judges to evaluate the complete interaction context.

## Public API

```python
from petri import (
    auditor_agent,           # Main solver
    alignment_judge,         # Main scorer
    Transcript,              # Transcript class
    TranscriptMetadata,      # Metadata class
    default_auditor_tools    # Standard tools
)
```

### Creating a Custom Task

```python
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from petri.solvers.auditor_agent import auditor_agent
from petri.scorers.judge import alignment_judge

@task
def custom_audit():
    return Task(
        dataset=[Sample(input="Your special instruction here")],
        solver=auditor_agent(max_turns=30),
        scorer=alignment_judge(),
    )
```

## Contributing Special Instructions

New special instructions are added to `petri/src/petri/tasks/petri.py` in the `AUDITOR_INSTRUCTIONS` list. Good instructions should be:

- Specific and testable through conversation
- Realistic scenarios that could occur in real usage
- Focused on one main alignment concern
