#!/usr/bin/env python3
"""
Evaluate language models on a token counting task.

Tests whether models can accurately count occurrences of a target token
in a sequence, for studying count representations in mechanistic interpretability.

Supports:
- Local models via transformers (Qwen3-4B-Instruct, Qwen3-14B)
- Claude via Anthropic API (Claude 4.5 Sonnet)

Usage:
    # Evaluate local Qwen 4B model (default)
    python eval_counting.py --model qwen-4b

    # Evaluate local Qwen 14B model (requires more VRAM)
    python eval_counting.py --model qwen-14b

    # Evaluate Claude 4.5 Sonnet
    python eval_counting.py --model claude

    # Test different count ranges
    python eval_counting.py --model claude --num-samples 200 --min-count 0 --max-count 100

    # Detailed analysis with plot
    python eval_counting.py --model claude --analyze-bins --show-examples --plot results.png

    # Save results to file
    python eval_counting.py --model qwen-4b --save results.json

    # Use fixed seed for reproducibility
    python eval_counting.py --model qwen-4b --seed 42
"""

import argparse
import json
import os
import random
import re

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from counting_data import (
    CountingSequence,
    create_prompt,
    format_chat_prompt,
    generate_uniform_count_sequences,
)

# Load environment variables
load_dotenv()

# Model configurations
AVAILABLE_MODELS = {
    "qwen-0.6b": {
        "name": "qwen3-0.6b",
        "path": "Qwen/Qwen3-0.6B",
        "type": "local",
    },
    "qwen-4b": {
        "name": "qwen3-4b-instruct",
        "path": "Qwen/Qwen3-4B-Instruct-2507",
        "type": "local",
    },
    "qwen-14b": {
        "name": "qwen3-14b",
        "path": "Qwen/Qwen3-14B",
        "type": "local",
    },
    "gemma-12b": {
        "name": "gemma-3-12b-it",
        "path": "google/gemma-3-12b-it",
        "type": "local",
    },
    "gemma-27b": {
        "name": "gemma-3-27b-it",
        "path": "google/gemma-3-27b-it",
        "type": "local",
    },
    "claude": {
        "name": "claude-4.5-sonnet",
        "model_id": "claude-sonnet-4-5-20250929",
        "type": "api",
    },
}


def extract_count_from_response(response: str) -> int | None:
    """Try to extract a number from the model's response."""
    # Clean up response
    response = response.strip()
    
    # Try to find a number at the start
    match = re.match(r'\s*(\d+)', response)
    if match:
        return int(match.group(1))
    
    # Try to find any number in the response
    numbers = re.findall(r'\d+', response)
    if numbers:
        return int(numbers[0])
    
    # Try to parse word numbers
    word_to_num = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12,
    }
    response_lower = response.lower()
    for word, num in word_to_num.items():
        if word in response_lower:
            return num
    
    return None


def evaluate_local_model(
    examples: list[CountingSequence],
    model_config: dict,
    device: str = "cuda",
    max_new_tokens: int = 10,
    dtype: str = "float16",
    batch_size: int = 1,
) -> dict:
    """Evaluate a local model on the counting task.

    Args:
        examples: List of counting sequences to evaluate
        model_config: Model configuration dict with 'name' and 'path'
        device: Device to run on (default: cuda)
        max_new_tokens: Max tokens to generate per example
        dtype: Model dtype (float16 or bfloat16)
        batch_size: Number of examples to process in parallel (default: 1)
    """
    model_name = model_config["name"]
    model_path = model_config["path"]
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

    print(f"\nLoading {model_name} from {model_path}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  Failed to load model: {e}")
        return None

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use left padding for batch generation (standard for decoder models)
    tokenizer.padding_side = "left"

    model.eval()

    # Sort examples by sequence length (descending) to minimize padding
    # and surface OOM errors early with longest sequences first
    sorted_examples = sorted(examples, key=lambda x: x.sequence_length, reverse=True)

    results = []

    # Process in batches
    pbar = tqdm(total=len(sorted_examples), desc="  Evaluating")
    for batch_start in range(0, len(sorted_examples), batch_size):
        batch_examples = sorted_examples[batch_start:batch_start + batch_size]

        # Create prompts for the batch
        prompts = [format_chat_prompt(example, tokenizer) for example in batch_examples]

        # Tokenize batch with padding
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        input_lengths = [inputs.attention_mask[i].sum().item() for i in range(len(batch_examples))]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding for reproducibility
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode responses for each example in batch
        for i, example in enumerate(batch_examples):
            # Extract only the new tokens (skip input tokens)
            input_len = input_lengths[i]
            # With left padding, the real input starts at (total_len - input_len)
            # So new tokens start at the original sequence length
            padded_input_len = inputs.input_ids.shape[1]
            new_tokens = outputs[i][padded_input_len:]

            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            predicted_count = extract_count_from_response(response)

            results.append({
                "true_count": example.true_count,
                "predicted_count": predicted_count,
                "response": response[:50],  # Truncate for display
                "correct": predicted_count == example.true_count,
                "sequence_length": example.sequence_length,
                "target_token": example.target_token,
            })

        pbar.update(len(batch_examples))
    pbar.close()

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()

    return _compute_metrics(results, model_name)


def evaluate_claude_model(
    examples: list[CountingSequence],
    model_config: dict,
    max_tokens: int = 10,
    max_concurrent: int = 15,
) -> dict:
    """Evaluate Claude via Anthropic API on the counting task.

    Uses async requests with concurrency for speed.
    """
    try:
        import asyncio
        from anthropic import AsyncAnthropic
    except ImportError:
        print("Error: anthropic package is required. Install with: pip install anthropic")
        return None

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found in environment variables")
        return None

    model_name = model_config["name"]
    model_id = model_config["model_id"]

    print(f"\nUsing Claude API ({model_name}) with {max_concurrent} concurrent requests...")

    async def process_example(
        client: "AsyncAnthropic",
        idx: int,
        example: CountingSequence,
        semaphore: asyncio.Semaphore,
        pbar: tqdm,
    ) -> dict:
        """Process a single example with retry logic."""
        prompt = create_prompt(example)
        max_retries = 3
        retry_delay = 2
        response = ""

        async with semaphore:
            for attempt in range(max_retries):
                try:
                    message = await client.messages.create(
                        model=model_id,
                        max_tokens=max_tokens,
                        temperature=0,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    response = message.content[0].text
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        tqdm.write(f"    Error on example {idx} after {max_retries} attempts: {e}")
                        response = ""

            pbar.update(1)

        predicted_count = extract_count_from_response(response)
        return {
            "idx": idx,
            "true_count": example.true_count,
            "predicted_count": predicted_count,
            "response": response[:50],
            "correct": predicted_count == example.true_count,
            "sequence_length": example.sequence_length,
            "target_token": example.target_token,
        }

    async def run_all():
        client = AsyncAnthropic(api_key=api_key)
        semaphore = asyncio.Semaphore(max_concurrent)

        with tqdm(total=len(examples), desc="  Evaluating") as pbar:
            tasks = [
                process_example(client, i, ex, semaphore, pbar)
                for i, ex in enumerate(examples)
            ]
            results = await asyncio.gather(*tasks)

        # Sort by original index to maintain order
        results.sort(key=lambda x: x["idx"])
        # Remove idx field
        for r in results:
            del r["idx"]
        return results

    results = asyncio.run(run_all())
    return _compute_metrics(results, model_name)


def _compute_metrics(results: list[dict], model_name: str) -> dict:
    """Compute evaluation metrics from results."""
    valid_results = [r for r in results if r["predicted_count"] is not None]

    accuracy = sum(r["correct"] for r in results) / len(results)
    parse_rate = len(valid_results) / len(results)

    # Correlation (only for parsed results)
    if len(valid_results) > 1:
        true_counts = [r["true_count"] for r in valid_results]
        pred_counts = [r["predicted_count"] for r in valid_results]

        # Pearson correlation
        mean_true = sum(true_counts) / len(true_counts)
        mean_pred = sum(pred_counts) / len(pred_counts)

        numerator = sum((t - mean_true) * (p - mean_pred) for t, p in zip(true_counts, pred_counts))
        denom_true = sum((t - mean_true) ** 2 for t in true_counts) ** 0.5
        denom_pred = sum((p - mean_pred) ** 2 for p in pred_counts) ** 0.5

        if denom_true > 0 and denom_pred > 0:
            correlation = numerator / (denom_true * denom_pred)
        else:
            correlation = 0.0

        # Mean absolute error
        mae = sum(abs(t - p) for t, p in zip(true_counts, pred_counts)) / len(valid_results)
    else:
        correlation = 0.0
        mae = float('inf')

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "correlation": correlation,
        "mae": mae,
        "parse_rate": parse_rate,
        "num_examples": len(results),
        "results": results,
    }


def save_results(results: dict, output_file: str):
    """Save evaluation results to a JSON file."""
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"Results saved to: {output_file}")


def load_results(input_file: str) -> dict:
    """Load evaluation results from a JSON file."""
    with open(input_file, "r") as f:
        results = json.load(f)
    print(f"Loaded results from: {input_file}")
    print(f"  Model: {results['model_name']}")
    print(f"  Examples: {results['num_examples']}")
    return results


def print_results_table(all_results: list[dict]):
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'Accuracy':>10} {'Correlation':>12} {'MAE':>8} {'Parse Rate':>12}")
    print("-" * 80)
    
    for result in all_results:
        if result is None:
            continue
        print(
            f"{result['model_name']:<20} "
            f"{result['accuracy']:>10.1%} "
            f"{result['correlation']:>12.3f} "
            f"{result['mae']:>8.2f} "
            f"{result['parse_rate']:>12.1%}"
        )
    print("=" * 80)


def print_example_outputs(results: dict, num_examples: int = 5):
    """Print some example model outputs."""
    print(f"\nExample outputs for {results['model_name']}:")
    print("-" * 60)

    for r in results["results"][:num_examples]:
        status = "✓" if r["correct"] else "✗"
        print(f"  True: {r['true_count']:>2}, Predicted: {str(r['predicted_count']):>4}, "
              f"{status} Response: {r['response']!r}")


def analyze_by_bins(results: dict, bin_type: str = "count"):
    """Analyze results by count bins or sequence length bins.

    Args:
        bin_type: Either "count" (bin by true count) or "length" (bin by sequence length)
    """
    from collections import Counter

    print(f"\n{results['model_name']} - Analysis by {bin_type}:")
    print("-" * 80)

    if bin_type == "count":
        bins = [(0, 3), (4, 6), (7, 9), (10, 12), (13, 20)]
        bin_key = lambda r: r['true_count']
    elif bin_type == "length":
        bins = [(10, 20), (21, 30), (31, 40), (41, 50)]
        bin_key = lambda r: r.get('sequence_length', 0)
    else:
        raise ValueError(f"Unknown bin_type: {bin_type}")

    for low, high in bins:
        bin_results = [r for r in results['results']
                      if low <= bin_key(r) <= high and r['predicted_count'] is not None]
        if not bin_results:
            continue

        acc = sum(r['correct'] for r in bin_results) / len(bin_results)
        mae = sum(abs(r['true_count'] - r['predicted_count']) for r in bin_results) / len(bin_results)
        avg_pred = sum(r['predicted_count'] for r in bin_results) / len(bin_results)
        avg_true = sum(r['true_count'] for r in bin_results) / len(bin_results)

        # Compute correlation within bin
        if len(bin_results) > 1:
            true_counts = [r['true_count'] for r in bin_results]
            pred_counts = [r['predicted_count'] for r in bin_results]
            mean_true = sum(true_counts) / len(true_counts)
            mean_pred = sum(pred_counts) / len(pred_counts)

            numerator = sum((t - mean_true) * (p - mean_pred) for t, p in zip(true_counts, pred_counts))
            denom_true = sum((t - mean_true) ** 2 for t in true_counts) ** 0.5
            denom_pred = sum((p - mean_pred) ** 2 for p in pred_counts) ** 0.5

            if denom_true > 0 and denom_pred > 0:
                corr = numerator / (denom_true * denom_pred)
            else:
                corr = 0.0
        else:
            corr = 0.0

        # Prediction diversity
        pred_counter = Counter(r['predicted_count'] for r in bin_results)
        diversity = len(pred_counter)  # Number of unique predictions

        print(f"  {bin_type.capitalize()} {low:>2}-{high:<2}: "
              f"N={len(bin_results):>3} | "
              f"Acc={acc:>5.1%} | "
              f"Corr={corr:>5.2f} | "
              f"MAE={mae:>4.2f} | "
              f"AvgTrue={avg_true:>4.1f} | "
              f"AvgPred={avg_pred:>4.1f} | "
              f"Diversity={diversity}")
    print()


def create_scatter_plot(results: dict, output_file: str = "counting_performance.png"):
    """Create a scatter plot visualization of model performance."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required for plotting. Install with: pip install matplotlib")
        return

    valid_results = [r for r in results['results'] if r['predicted_count'] is not None]
    true_counts = [r['true_count'] for r in valid_results]
    pred_counts = [r['predicted_count'] for r in valid_results]
    seq_lengths = [r['sequence_length'] for r in valid_results]
    correct = [r['correct'] for r in valid_results]

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Subplot 1: Predicted vs True with perfect line
    ax1 = axes[0, 0]
    colors = ['green' if c else 'red' for c in correct]
    ax1.scatter(true_counts, pred_counts, c=colors, alpha=0.6, s=50)

    # Add perfect prediction line
    max_count = max(max(true_counts), max(pred_counts))
    ax1.plot([0, max_count], [0, max_count], 'k--', linewidth=2, label='Perfect prediction')

    # Add best fit line
    coeffs = np.polyfit(true_counts, pred_counts, 1)
    fit_line = np.poly1d(coeffs)
    x_fit = np.linspace(0, max_count, 100)
    ax1.plot(x_fit, fit_line(x_fit), 'b-', linewidth=2, alpha=0.7,
             label=f'Best fit: y={coeffs[0]:.2f}x+{coeffs[1]:.2f}')

    ax1.set_xlabel('True Count', fontsize=12)
    ax1.set_ylabel('Predicted Count', fontsize=12)
    ax1.set_title(f'{results["model_name"]}: Predicted vs True Count\\nAccuracy: {results["accuracy"]:.1%}, Correlation: {results["correlation"]:.3f}',
                  fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Error vs True Count
    ax2 = axes[0, 1]
    errors = [p - t for t, p in zip(true_counts, pred_counts)]
    ax2.scatter(true_counts, errors, alpha=0.6, s=50, c='purple')
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=2)

    # Add mean error line
    mean_error = np.mean(errors)
    ax2.axhline(y=mean_error, color='r', linestyle='-', linewidth=2, alpha=0.7,
                label=f'Mean error: {mean_error:+.2f}')

    ax2.set_xlabel('True Count', fontsize=12)
    ax2.set_ylabel('Prediction Error (Pred - True)', fontsize=12)
    ax2.set_title('Error Analysis', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Performance by Sequence Length
    ax3 = axes[1, 0]
    scatter = ax3.scatter(seq_lengths, true_counts, c=pred_counts, alpha=0.6, s=50, cmap='viridis')
    ax3.set_xlabel('Sequence Length', fontsize=12)
    ax3.set_ylabel('True Count', fontsize=12)
    ax3.set_title('Sequence Length vs Count (color=prediction)', fontsize=13, fontweight='bold')
    plt.colorbar(scatter, ax=ax3, label='Predicted Count')
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Accuracy by Count Bins
    ax4 = axes[1, 1]
    from collections import Counter

    # Determine bins based on data range
    max_true = max(true_counts)
    if max_true <= 15:
        bins = [(0, 3), (4, 6), (7, 9), (10, 12), (13, 20)]
    elif max_true <= 30:
        bins = [(0, 5), (6, 10), (11, 15), (16, 20), (21, 30)]
    else:
        bins = [(0, 10), (11, 20), (21, 30), (31, 40), (41, 100)]

    bin_accs = []
    bin_labels = []
    bin_ns = []

    for low, high in bins:
        bin_results = [r for r in valid_results if low <= r['true_count'] <= high]
        if bin_results:
            acc = sum(r['correct'] for r in bin_results) / len(bin_results)
            bin_accs.append(acc * 100)
            bin_labels.append(f'{low}-{high}')
            bin_ns.append(len(bin_results))

    if bin_accs:
        bars = ax4.bar(bin_labels, bin_accs, color=['green' if a > 50 else 'orange' if a > 25 else 'red' for a in bin_accs])
        ax4.set_xlabel('Count Range', fontsize=12)
        ax4.set_ylabel('Accuracy (%)', fontsize=12)
        ax4.set_title('Accuracy by Count Range', fontsize=13, fontweight='bold')
        ax4.set_ylim([0, 100])

        # Add N labels on bars
        for bar, n in zip(bars, bin_ns):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'N={n}', ha='center', va='bottom', fontsize=9)

        ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f'\nScatter plot saved to: {output_file}')

    # Print additional statistics
    print(f'\nPlot statistics:')
    print(f'  Best fit slope: {coeffs[0]:.3f} (1.0 = perfect)')
    print(f'  Best fit intercept: {coeffs[1]:.2f} (0.0 = no bias)')
    print(f'  Mean error: {np.mean(errors):.2f}')
    print(f'  Std error: {np.std(errors):.2f}')


def main():
    parser = argparse.ArgumentParser(description="Evaluate language models on token counting task")
    parser.add_argument(
        "--model",
        choices=list(AVAILABLE_MODELS.keys()),
        default="qwen-4b",
        help="Model to evaluate: 'qwen-4b' (default), 'qwen-14b', or 'claude'",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of test examples to generate",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=0,
        help="Minimum target count",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=30,
        help="Maximum target count",
    )
    parser.add_argument(
        "--target-only",
        action="store_true",
        help="Generate sequences containing only the target token (no distractors)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (random by default)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for local model evaluation (default: 1)",
    )
    parser.add_argument(
        "--show-examples",
        action="store_true",
        help="Show example outputs for each model",
    )
    parser.add_argument(
        "--analyze-bins",
        action="store_true",
        help="Show detailed analysis by count ranges and sequence lengths",
    )
    parser.add_argument(
        "--plot",
        type=str,
        metavar="FILE",
        help="Generate scatter plot visualization and save to FILE",
    )
    parser.add_argument(
        "--save",
        type=str,
        metavar="FILE",
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--load",
        type=str,
        metavar="FILE",
        help="Load results from JSON file instead of running evaluation",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16",
        help="Data type for model weights (default: bfloat16, only used for local models)",
    )

    args = parser.parse_args()

    # Load results from file if specified
    if args.load:
        result = load_results(args.load)
    else:
        # Set random seed if specified
        if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)

        # Generate test examples
        # Set density range based on target-only flag
        density_range = (1.0, 1.0) if args.target_only else (0.05, 0.8)

        print(f"Generating {args.num_samples} test examples (count range: {args.min_count}-{args.max_count})...")
        examples = generate_uniform_count_sequences(
            min_count=args.min_count,
            max_count=args.max_count,
            num_sequences=args.num_samples,
            density_range=density_range,
            seed=args.seed,
        )

        # Evaluate model based on selection
        model_config = AVAILABLE_MODELS[args.model]

        if model_config["type"] == "local":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                print("Error: CUDA is required for local model evaluation. CPU is not supported.")
                return
            print(f"Using dtype: {args.dtype}, batch_size: {args.batch_size}")
            result = evaluate_local_model(
                examples=examples,
                model_config=model_config,
                device=device,
                dtype=args.dtype,
                batch_size=args.batch_size,
            )
        elif model_config["type"] == "api":
            result = evaluate_claude_model(
                examples=examples,
                model_config=model_config,
            )
        else:
            print(f"Unknown model type: {model_config['type']}")
            return

        if result is None:
            return

        # Save results if requested
        if args.save:
            save_results(result, args.save)

    if result:
        if args.show_examples:
            print_example_outputs(result)

        if args.analyze_bins:
            analyze_by_bins(result, bin_type="count")
            analyze_by_bins(result, bin_type="length")

        # Print summary
        print_results_table([result])

        # Generate plot if requested
        if args.plot:
            create_scatter_plot(result, args.plot)


if __name__ == "__main__":
    main()
