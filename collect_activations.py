#!/usr/bin/env python3
"""
Extract residual stream activations for counting task sequences.

Usage:
    python collect_activations.py --model gemma-12b --min-count 1 --max-count 50 --sequences-per-count 10 --output activations.pt
"""

import argparse

import torch
from nnsight import LanguageModel
from tqdm import tqdm

from counting_data import CountingSequence, generate_sequence_with_target_count, format_chat_prompt
from models import AVAILABLE_MODELS
from token_mapping import find_sequence_token_positions

torch.set_grad_enabled(False)

# Only support gemma models for activation extraction
SUPPORTED_MODELS = ["gemma-12b", "gemma-27b"]


def extract_activations_for_batch(
    model: LanguageModel,
    tokenizer,
    sequences: list[CountingSequence],
    layers: list[int] = None,
) -> list[torch.Tensor]:
    """
    Extract residual stream activations for a batch of sequences.

    Returns:
        List of tensors, each of shape (num_layers, seq_len, hidden_dim)
    """
    n_layers = len(model.model.language_model.layers)
    target_layers = list(range(n_layers)) if layers is None else layers

    # Format prompts and tokenize with padding
    prompts = [format_chat_prompt(seq, tokenizer) for seq in sequences]
    tokenizer.padding_side = "left"
    tokens = tokenizer(prompts, return_tensors="pt", padding=True)

    layer_activations = []

    with model.trace(tokens) as tracer:
        for layer_idx in target_layers:
            hidden_states = model.model.language_model.layers[layer_idx].output[0]
            layer_activations.append(hidden_states.save())

    # layer_activations[layer] has shape (batch, seq_len, hidden_dim)
    # Stack to (num_layers, batch, seq_len, hidden_dim)
    stacked = torch.stack(layer_activations)

    # Extract per-sequence activations, removing padding
    results = []
    for batch_idx in range(len(sequences)):
        # Find where actual tokens start (non-padded region)
        seq_mask = tokens.attention_mask[batch_idx]
        seq_len = seq_mask.sum().item()

        # Extract this sequence's activations (last seq_len tokens)
        seq_activations = stacked[:, batch_idx, -seq_len:, :].cpu()
        results.append(seq_activations)

    return results


def parse_layer_spec(spec: str) -> list[int]:
    """Parse a layer specification string into a list of layer indices.

    Supports comma-separated values and ranges (inclusive).
    Example: "4,5,7-9,12" -> [4, 5, 7, 8, 9, 12]
    """
    layers = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def generate_sequences_per_count(
    min_count: int,
    max_count: int,
    sequences_per_count: int,
) -> list[CountingSequence]:
    """Generate a fixed number of sequences for each count value."""
    examples = []
    for count in range(min_count, max_count + 1):
        for _ in range(sequences_per_count):
            examples.append(generate_sequence_with_target_count(count))
    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Extract residual stream activations for counting task sequences"
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default="gemma-12b",
        help="Model to use for activation extraction",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum count value",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=50,
        help="Maximum count value",
    )
    parser.add_argument(
        "--sequences-per-count",
        type=int,
        default=10,
        help="Number of sequences to generate per count value",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path for saving activations (.pt)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for activation extraction (default: 8)",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Layers to extract (default: all). Comma-separated with ranges, e.g. '4,5,7-9,12'",
    )
    args = parser.parse_args()

    # Parse layers argument
    target_layers = parse_layer_spec(args.layers) if args.layers else None

    # Load model
    model_config = AVAILABLE_MODELS[args.model]
    model_path = model_config["path"]
    print(f"Loading model: {model_path}")
    model = LanguageModel(model_path, device_map="auto", dtype=torch.bfloat16)
    tokenizer = model.tokenizer

    # Report layer selection
    n_layers = len(model.model.language_model.layers)
    if target_layers:
        print(f"Extracting layers: {target_layers} (of {n_layers} total)")
    else:
        print(f"Extracting all {n_layers} layers")

    # Generate sequences
    total_sequences = (args.max_count - args.min_count + 1) * args.sequences_per_count
    print(f"Generating {total_sequences} sequences (counts {args.min_count}-{args.max_count}, {args.sequences_per_count} per count)")
    examples = generate_sequences_per_count(
        min_count=args.min_count,
        max_count=args.max_count,
        sequences_per_count=args.sequences_per_count,
    )

    # Extract activations
    print(f"Extracting activations (batch_size={args.batch_size})...")
    all_activations = []
    all_metadata = []

    pbar = tqdm(total=len(examples))
    for batch_start in range(0, len(examples), args.batch_size):
        batch_examples = examples[batch_start:batch_start + args.batch_size]
        batch_activations = extract_activations_for_batch(model, tokenizer, batch_examples, layers=target_layers)

        for example, activations in zip(batch_examples, batch_activations):
            # Get token position mappings
            prompt = format_chat_prompt(example, tokenizer)
            token_positions = find_sequence_token_positions(
                tokenizer, prompt, example.tokens, target_token=example.target_token
            )

            all_activations.append(activations)
            all_metadata.append({
                "true_count": example.true_count,
                "sequence": example.sequence,
                "target_token": example.target_token,
                "sequence_length": example.sequence_length,
                "tokens": example.tokens,
                "token_positions": token_positions,
            })

        pbar.update(len(batch_examples))
        torch.cuda.empty_cache()
    pbar.close()

    # Save to disk
    save_data = {
        "activations": all_activations,
        "metadata": all_metadata,
        "model_name": args.model,
        "layers": target_layers if target_layers else list(range(n_layers)),
        "args": vars(args),
    }
    torch.save(save_data, args.output)
    print(f"Saved activations to: {args.output}")


if __name__ == "__main__":
    main()
