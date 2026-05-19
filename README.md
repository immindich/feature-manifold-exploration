# Feature Manifold Exploration

Investigating how language models internally represent counts, and why their internal representations are more accurate than their verbalized answers. See [this post](https://immindich.github.io/blog/llm-counting/) for more details.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). After installing uv, sync the environment:

```bash
uv sync
```

This creates a `.venv/` with everything pinned in `uv.lock`. Run scripts via `uv run python ...` (or activate the venv).

If you are using Gemma, you need to follow the instructions in the repo to get access to the weights and authenticate with Hugging Face. Set `ANTHROPIC_API_KEY` for Claude evaluation.

The code auto-selects CUDA, Apple MPS, or CPU at runtime, so it works on Apple Silicon as well as NVIDIA GPUs. Running the full-size models locally still needs a lot of memory — the original experiments were run on a 96GB GH200 instance. For laptops, the registered `smollm2-135m` (ungated) and `gemma-270m` / `gemma-4-E4B` (gated, ~16 GB) models work; the 12B/27B Gemma models won't fit.

On MPS the transformers caching-allocator warmup tries to allocate the full model size as one buffer, which exceeds the per-buffer Metal limit for larger models. The scripts call `disable_mps_allocator_warmup()` from `device_utils.py` to skip it.

## Scripts

### `eval_counting.py`

Evaluates models on a synthetic token-counting task: given a sequence of space-separated letters, count occurrences of a target letter. Supports local models (Gemma, Qwen) via transformers and Claude via the Anthropic API.

```bash
python eval_counting.py --model gemma-27b --min-count 1 --max-count 150 --num-samples 1500 --batch-size 10 --save results.json
python eval_counting.py --load results.json --plot results.png
```

### `collect_activations.py`

Extracts residual stream activations from Gemma models at specified layers.

```bash
python collect_activations.py --model gemma-27b --min-count 1 --max-count 150 --sequences-per-count 80 --layers 5,15,25,35,45,55 --output activations-27b.pt
```

### `analyze_activations.py`

Notebook for analyzing the activation data.

### `train_probes.py`

Trains linear and MLP probes to predict count from residual stream activations at each layer.

```bash
python train_probes.py --data activations-27b.pt --model linear --device cuda --output probe-linear-27b.json
python train_probes.py --data activations-27b.pt --model mlp --mlp-hidden 128 --device cuda --output probe-mlp-27b.json
```

### `compare_probe_vs_model.py`

Runs the model on the same test set used for probe evaluation and compares bias-corrected metrics across count ranges.
