# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Circuit-tracer is a library for finding circuits in language models using features from cross-layer MLP transcoders, based on [Ameisen et al. (2025)](https://transformer-circuits.pub/2025/attribution-graphs/methods.html). The library:

1. Computes attribution graphs showing direct effects between transcoder features, error nodes, tokens, and output logits
2. Visualizes and annotates these graphs via a local web server or Neuronpedia
3. Enables interventions on transcoder features to observe model behavior changes

## Build and Development Commands

```bash
# Install the package
pip install .

# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_graph.py

# Run a specific test
pytest tests/test_graph.py::test_function_name

# Linting and formatting
ruff check .
ruff format .

# Type checking
pyright
```

## CLI Usage

```bash
# Complete workflow: attribution + visualization
circuit-tracer attribute \
  --prompt "The capital of France is" \
  --transcoder_set gemma \
  --slug my-graph \
  --graph_file_dir ./graph_files \
  --server

# Attribution only (save raw graph)
circuit-tracer attribute \
  --prompt "The capital of France is" \
  --transcoder_set llama \
  --graph_output_path output.pt

# Start server for existing graphs
circuit-tracer start-server --graph_file_dir ./graph_files --port 8041
```

Transcoder set shortcuts: `gemma` (GemmaScope), `llama` (ReLU transcoders), or HuggingFace repo IDs.

## Architecture

### Core Components

**ReplacementModel** (`replacement_model/`): Factory that creates backend-specific models.
- `ReplacementModel.from_pretrained(model_name, transcoder_set, backend="transformerlens"|"nnsight")`
- TransformerLens backend: Uses `HookedTransformer`, faster and more memory-efficient
- NNSight backend: Supports more HuggingFace models, still experimental

**Attribution** (`attribution/`): Computes direct effects for the attribution graph.
- `attribute(prompt, model, ...)` routes to backend-specific implementation
- Produces dense adjacency matrix organized as: `[features, error_nodes, embed_nodes, logit_nodes]`

**Graph** (`graph.py`): Contains adjacency matrix and node information.
- `Graph.to_pt(path)` / `Graph.from_pt(path)` for serialization
- `prune_graph(graph, node_threshold, edge_threshold)` removes low-influence nodes/edges
- `compute_graph_scores(graph)` returns (replacement_score, completeness_score)

**Transcoders** (`transcoder/`): MLP replacement modules.
- `SingleLayerTranscoder`: Per-layer transcoder
- `CrossLayerTranscoder`: Cross-layer transcoder (CLT)
- `TranscoderSet`: Collection of single-layer transcoders

**Frontend** (`frontend/`): Graph visualization server.
- `local_server.serve(data_dir, port)` starts visualization server
- `create_graph_files(graph, slug, output_path, ...)` converts Graph to JSON for frontend

### Data Flow

```
Prompt → ReplacementModel (with transcoders) → attribute() → Graph → prune_graph() → create_graph_files() → Local Server
```

### Key Parameters

- **batch_size**: Backward pass batch size (default: 256)
- **max_feature_nodes**: Limit on feature nodes in graph (default: 7500)
- **node_threshold**: Fraction of influence to keep when pruning nodes (default: 0.8)
- **edge_threshold**: Fraction of influence to keep when pruning edges (default: 0.98)
- **offload**: Memory optimization - `"cpu"`, `"disk"`, or `None`

## Available Transcoders

- Gemma-2 (2B): `mntss/gemma-scope-transcoders`, CLTs at 426K and 2.5M features
- Llama-3.2 (1B): `mntss/transcoder-Llama-3.2-1B`, `mntss/clt-llama-3.2-1b-524k`
- Qwen-3: Various sizes from 0.6B to 14B (see README)

## Python API

```python
from circuit_tracer import ReplacementModel, attribute, Graph
from circuit_tracer.graph import prune_graph
from circuit_tracer.utils.create_graph_files import create_graph_files

# Load model with transcoders
model = ReplacementModel.from_pretrained(
    "google/gemma-2-2b",
    transcoder_set="gemma",  # or HuggingFace repo ID
    backend="transformerlens"
)

# Compute attribution graph
graph = attribute(
    prompt="The capital of France is",
    model=model,
    batch_size=256,
    verbose=True
)

# Save/load graphs
graph.to_pt("my_graph.pt")
loaded = Graph.from_pt("my_graph.pt")

# Prune for visualization
result = prune_graph(graph, node_threshold=0.8, edge_threshold=0.98)
```
