#!/usr/bin/env python3
"""
Extract residual stream activations for counting task sequences.

Usage:
    python collect_activations.py --model gemma-12b --min-count 1 --max-count 50 --sequences-per-count 10 --output activations.pt
"""

import argparse
import gc

import torch
from nnsight import LanguageModel, VisionLanguageModel
from tqdm import tqdm

from counting_data import CountingSequence, generate_sequences_per_count, format_chat_prompt
from device_utils import disable_mps_allocator_warmup, empty_cache, get_device
from models import AVAILABLE_MODELS

disable_mps_allocator_warmup()

torch.set_grad_enabled(False)

# Models supported for activation extraction (anything where get_decoder_layers works).
SUPPORTED_MODELS = ["smollm2-135m", "gemma-270m", "gemma-4-E4B", "gemma-12b", "gemma-27b"]


def get_decoder_layers(model):
    """Return the list of decoder layers regardless of architecture.

    Multimodal Gemma (3/4) wraps a language model inside a vision-language
    model (model.model.language_model.layers); Llama-style models put layers
    directly under model.model.layers.
    """
    underlying = model._model
    if hasattr(underlying, "model") and hasattr(underlying.model, "language_model"):
        return model.model.language_model.layers
    return model.model.layers


def load_nnsight_model(model_path: str, device: str, dtype=torch.bfloat16):
    """Load a model via nnsight, picking VisionLanguageModel for multimodal
    architectures and LanguageModel for plain text models."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_path)
    # Multimodal wrappers have a text_config sub-config.
    is_multimodal = hasattr(cfg, "text_config") or hasattr(cfg, "vision_config")
    model_cls = VisionLanguageModel if is_multimodal else LanguageModel

    device_map = "auto" if device == "cuda" else device
    return model_cls(model_path, device_map=device_map, dtype=dtype)


def extract_activations_for_batch(
    model: LanguageModel,
    tokenizer,
    sequences: list[CountingSequence],
    layers: list[int] = None,
) -> list[torch.Tensor]:
    """
    Extract last-token residual stream activations for a batch of sequences.

    Returns:
        List of tensors, each of shape (num_layers, hidden_dim)
    """
    decoder_layers = get_decoder_layers(model)
    n_layers = len(decoder_layers)
    target_layers = list(range(n_layers)) if layers is None else layers

    # Format prompts and tokenize with padding
    prompts = [format_chat_prompt(seq, tokenizer) for seq in sequences]
    tokenizer.padding_side = "left"
    tokens = tokenizer(prompts, return_tensors="pt", padding=True)

    layer_activations = []

    with model.trace(tokens) as tracer:
        for layer_idx in target_layers:
            # In transformers 5.x the layer's output is the hidden-state tensor
            # directly; older versions wrapped it in a tuple. .output gives a
            # (batch, seq, hidden) tensor here.
            hidden_states = decoder_layers[layer_idx].output
            layer_activations.append(hidden_states[:, -1, :].save())

    # layer_activations[layer] has shape (batch, hidden_dim)
    # Stack to (num_layers, batch, hidden_dim)
    stacked = torch.stack(layer_activations)
    # Return list of (num_layers, hidden_dim) tensors
    results = [stacked[:, i, :].cpu().clone() for i in range(len(sequences))]
    del stacked, layer_activations
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
    parser.add_argument(
        "--target-only",
        action="store_true",
        help="Generate sequences containing only the target token (no distractors)",
    )
    args = parser.parse_args()

    # Parse layers argument
    target_layers = parse_layer_spec(args.layers) if args.layers else None

    # Load model
    model_config = AVAILABLE_MODELS[args.model]
    model_path = model_config["path"]
    device = get_device()
    print(f"Loading model: {model_path} on device: {device}")
    model = load_nnsight_model(model_path, device=device, dtype=torch.bfloat16)
    tokenizer = model.tokenizer

    # Report layer selection
    n_layers = len(get_decoder_layers(model))
    if target_layers:
        print(f"Extracting layers: {target_layers} (of {n_layers} total)")
    else:
        print(f"Extracting all {n_layers} layers")

    # Generate sequences
    density_range = (1.0, 1.0) if args.target_only else (0.05, 0.8)
    total_sequences = (args.max_count - args.min_count + 1) * args.sequences_per_count
    mode_str = "target-only" if args.target_only else "mixed"
    print(f"Generating {total_sequences} sequences (counts {args.min_count}-{args.max_count}, {args.sequences_per_count} per count, {mode_str})")
    examples = generate_sequences_per_count(
        min_count=args.min_count,
        max_count=args.max_count,
        sequences_per_count=args.sequences_per_count,
        density_range=density_range,
    )
    # Sort by sequence length descending for more efficient batching (less padding waste)
    examples.sort(key=lambda x: x.sequence_length, reverse=True)

    # Extract activations
    print(f"Extracting activations (batch_size={args.batch_size})...")

    all_activations = []
    all_metadata = []

    # Single batch loop
    pbar = tqdm(total=len(examples))
    for batch_start in range(0, len(examples), args.batch_size):
        batch_examples = examples[batch_start:batch_start + args.batch_size]
        batch_activations = extract_activations_for_batch(
            model, tokenizer, batch_examples,
            layers=target_layers,
        )

        for example, activations in zip(batch_examples, batch_activations):
            all_activations.append(activations)
            all_metadata.append({
                "true_count": example.true_count,
                "sequence": example.sequence,
                "target_token": example.target_token,
                "sequence_length": example.sequence_length,
                "tokens": example.tokens,
            })

        pbar.update(len(batch_examples))

        # NNSight will leak memory if we don't do this explicitly
        del batch_activations
        gc.collect()
        empty_cache(device)
    pbar.close()

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
