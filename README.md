# GeoMapAgent

A benchmark suite for evaluating Large Language Model (LLM) performance on geographic boundary extraction from planning documents. This project tests how well various LLMs can extract and generate GeoJSON boundaries from PDF planning documents.

## Overview

GeoMapAgent evaluates LLM vision capabilities by:
1. Reading planning document PDFs (containing maps and geographic descriptions)
2. Extracting planning area boundaries as GeoJSON
3. Comparing extracted boundaries against ground truth using spatial metrics (IoU, precision, recall, F1)
4. Generating detailed performance reports across multiple models

## Project Structure

- `openrouter_client.py` - Unified LLM client supporting multiple models via OpenRouter API
- `geojson_metrics.py` - Spatial metrics calculation (IoU, precision, recall, F1) using Shapely
- `benchmark_runner.py` - Main benchmark orchestration and evaluation pipeline
- `.env.template` - Template for required environment variables

## Features

- Multi-model support (Claude Opus/Sonnet, GPT, Gemini)
- Spatial accuracy metrics using Shapely geometry operations
- Comprehensive result tracking and analysis
- Resume capability for long-running benchmarks

## Installation

```bash
uv sync
```

## Configuration

1. Copy the environment template:
```bash
cp .env.template .env
```

2. Add your OpenRouter API key to `.env`:
```
OPENROUTER_API_KEY=your_actual_api_key_here
```

Get an API key from [OpenRouter](https://openrouter.ai/).

## Usage

### Basic Benchmark Run

```python
from benchmark_runner import BenchmarkRunner

# Initialize runner
runner = BenchmarkRunner(
    dataset_excel_path="evaluation_data/0_planning_dataset_list.xlsx",
    evaluation_data_dir="evaluation_data",
    results_dir="benchmark_results"
)

# Run benchmark on selected models
models_to_test = [
    "claude-sonnet",  # Shorthand for anthropic/claude-sonnet-4.5
    "claude-opus",
    "gpt-5.2",
]

runner.run_benchmark(
    model_names=models_to_test,
    max_examples=10,      # Limit to first 10 examples
    start_from=0          # Start from beginning
)
```

### Extract GeoJSON from Single PDF

```python
from openrouter_client import OpenRouterClient

client = OpenRouterClient(model="claude-sonnet")
response = client.extract_geojson_from_pdf("path/to/planning_doc.pdf")

if response.get("success"):
    geojson = response["parsed_json"]
    # Use the GeoJSON data
```

## Metrics

The benchmark calculates several spatial metrics:

- **IoU (Intersection over Union)**: Overall overlap quality (0-1, higher is better)
- **Precision**: Of predicted area, how much is correct (measures over-prediction)
- **Recall**: Of ground truth area, how much was found (measures under-prediction)
- **F1 Score**: Harmonic mean of precision and recall (balanced metric)

## Output Structure

Results are saved in `benchmark_results/` with this structure:

```
benchmark_results/
├── model_name/
│   ├── folder_name_1/
│   │   ├── response.json       # Full LLM response
│   │   ├── predicted.geojson   # Extracted GeoJSON
│   │   └── metrics.json        # Performance metrics
│   ├── folder_name_2/
│   │   └── ...
│   └── summary.json            # Aggregate statistics
```

## Models

Supported models (via OpenRouter):

- `claude-sonnet` → `anthropic/claude-sonnet-4.5`
- `claude-opus` → `anthropic/claude-opus-4.5`
- `gpt-5.2` → `openai/gpt-5.2-pro`
- `gemini-pro` → `google/gemini-3-pro-preview`

You can also use any OpenRouter model ID directly.

## Requirements

- Python 3.10+
- OpenRouter API key
- Planning document dataset (PDFs + ground truth GeoJSON files)

## License

See LICENSE file for details.
