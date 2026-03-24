# Feature Manifold Exploration

Investigating how language models internally represent counts, and why their internal representations are more accurate than their verbalized answers. See [this post](https://immindich.github.io/blog/llm-counting/) for more details.

## Setup

Set up an environment and then install the required Python dependencies.

```bash
pip install torch transformers anthropic python-dotenv numpy tqdm matplotlib scikit-learn nnsight
```

If you are using Gemma, you need to follow the instructions in the repo to get access to the weights and authenticate with Hugging Face. Set `ANTHROPIC_API_KEY` for Claude evaluation.

Running models locally requires quite a bit of VRAM. I did these experiments on a 96GB GH200 instance. Unfortunately, smaller models than Gemma don't do very well on the counting task.

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
