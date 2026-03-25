"""
Test that batched activation extraction properly handles attention masks with left-padding.

If the attention mask is ignored, the first real token will attend to pad tokens
and get corrupted activations. This test verifies that activations for the same
sequence are nearly identical whether extracted alone or in a batch with longer sequences.
"""

import torch
from nnsight import LanguageModel

from counting_data import CountingSequence, format_chat_prompt


def extract_activations_without_mask(
    model: LanguageModel,
    tokenizer,
    sequences: list[CountingSequence],
    layers: list[int] = None,
) -> list[torch.Tensor]:
    """
    Extract activations WITHOUT using the attention mask.
    This should produce corrupted activations when padding is present.
    """
    n_layers = len(model.model.language_model.layers)
    target_layers = list(range(n_layers)) if layers is None else layers

    # Format prompts and tokenize with padding
    prompts = [format_chat_prompt(seq, tokenizer) for seq in sequences]
    tokenizer.padding_side = "left"
    tokens = tokenizer(prompts, return_tensors="pt", padding=True)

    layer_activations = []

    # Only pass input_ids, NOT the full BatchEncoding (which includes attention_mask)
    with model.trace(tokens.input_ids) as tracer:
        for layer_idx in target_layers:
            hidden_states = model.model.language_model.layers[layer_idx].output[0]
            layer_activations.append(hidden_states.save())

    stacked = torch.stack(layer_activations)

    results = []
    for batch_idx in range(len(sequences)):
        seq_mask = tokens.attention_mask[batch_idx]
        seq_len = seq_mask.sum().item()
        seq_activations = stacked[:, batch_idx, -seq_len:, :].cpu()
        results.append(seq_activations)

    return results


def test_batch_padding_attention_mask():
    """Verify that batching with left-padding produces consistent activations."""
    from collect_activations import extract_activations_for_batch

    print("Loading model...")
    model = LanguageModel("google/gemma-3-12b-it", device_map="auto", dtype=torch.bfloat16)
    tokenizer = model.tokenizer

    # Create a short sequence
    short_tokens = ["A", "X", "B", "X", "C"]
    short_seq = CountingSequence(
        tokens=short_tokens,
        sequence=" ".join(short_tokens),
        true_count=2,
        target_token="X",
    )

    # Create a much longer sequence (will force padding when batched)
    long_tokens = ["A", "X", "B", "C", "D", "X", "E", "F", "G", "H",
                   "I", "X", "J", "K", "L", "M", "N", "X", "O", "P",
                   "Q", "R", "S", "X", "T", "U", "V", "W", "X", "Y", "Z"]
    long_seq = CountingSequence(
        tokens=long_tokens,
        sequence=" ".join(long_tokens),
        true_count=6,
        target_token="X",
    )

    # Only extract a few layers for speed
    test_layers = [0, 10, 20]

    print("\nExtracting activations for short sequence alone...")
    [activations_alone] = extract_activations_for_batch(
        model, tokenizer, [short_seq], layers=test_layers
    )

    print("Extracting activations for short sequence in batch with longer sequence...")
    activations_batched = extract_activations_for_batch(
        model, tokenizer, [short_seq, long_seq], layers=test_layers
    )
    activations_in_batch = activations_batched[0]  # Get the short sequence's activations

    # Compare shapes
    print(f"\nActivations alone shape: {activations_alone.shape}")
    print(f"Activations in batch shape: {activations_in_batch.shape}")

    assert activations_alone.shape == activations_in_batch.shape, (
        f"Shape mismatch: {activations_alone.shape} vs {activations_in_batch.shape}"
    )

    # Compare values
    diff = (activations_alone - activations_in_batch).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Compute relative difference (normalized by activation magnitude)
    activation_magnitude = activations_alone.abs().mean().item()
    relative_diff = mean_diff / activation_magnitude if activation_magnitude > 0 else float('inf')

    print(f"\nActivation comparison:")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")
    print(f"  Activation magnitude (mean abs): {activation_magnitude:.6f}")
    print(f"  Relative difference: {relative_diff:.6%}")

    # Check per-layer differences
    print("\nPer-layer analysis:")
    for i, layer_idx in enumerate(test_layers):
        layer_diff = diff[i].mean().item()
        layer_max_diff = diff[i].max().item()
        print(f"  Layer {layer_idx}: mean_diff={layer_diff:.6f}, max_diff={layer_max_diff:.6f}")

    # The activations should be very close (allowing for floating point precision)
    # With bfloat16, we expect some numerical differences but they should be small
    threshold = 0.01  # 1% relative difference threshold

    with_mask_pass = relative_diff < threshold

    if with_mask_pass:
        print(f"\n✓ PASS: Activations are consistent (relative diff {relative_diff:.4%} < {threshold:.0%} threshold)")
        print("  Attention mask is being properly applied.")
    else:
        print(f"\n✗ FAIL: Activations differ significantly (relative diff {relative_diff:.4%} >= {threshold:.0%} threshold)")
        print("  WARNING: Attention mask may not be applied correctly!")
        print("  The first real tokens may be attending to pad tokens.")

    # Now test WITHOUT attention mask to show the difference
    print("\n" + "=" * 60)
    print("CONTROL TEST: Extracting WITHOUT attention mask...")
    print("=" * 60)

    activations_no_mask = extract_activations_without_mask(
        model, tokenizer, [short_seq, long_seq], layers=test_layers
    )
    activations_no_mask_short = activations_no_mask[0]

    # Compare no-mask activations to the single-sequence baseline
    diff_no_mask = (activations_alone - activations_no_mask_short).abs()
    max_diff_no_mask = diff_no_mask.max().item()
    mean_diff_no_mask = diff_no_mask.mean().item()
    relative_diff_no_mask = mean_diff_no_mask / activation_magnitude if activation_magnitude > 0 else float('inf')

    print(f"\nActivation comparison (WITHOUT mask vs baseline):")
    print(f"  Max absolute difference: {max_diff_no_mask:.6f}")
    print(f"  Mean absolute difference: {mean_diff_no_mask:.6f}")
    print(f"  Relative difference: {relative_diff_no_mask:.6%}")

    print("\nPer-layer analysis (WITHOUT mask):")
    for i, layer_idx in enumerate(test_layers):
        layer_diff = diff_no_mask[i].mean().item()
        layer_max_diff = diff_no_mask[i].max().item()
        print(f"  Layer {layer_idx}: mean_diff={layer_diff:.6f}, max_diff={layer_max_diff:.6f}")

    # The no-mask version should have much larger differences
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  WITH attention mask:    {relative_diff:.4%} relative difference")
    print(f"  WITHOUT attention mask: {relative_diff_no_mask:.4%} relative difference")
    print(f"  Ratio (no_mask / with_mask): {relative_diff_no_mask / relative_diff:.1f}x worse without mask")

    if relative_diff_no_mask > relative_diff * 5:
        print("\n✓ CONFIRMED: Attention mask makes a significant difference.")
        print("  The mask is being used and prevents pad token corruption.")
    else:
        print("\n⚠ WARNING: Attention mask doesn't seem to make much difference.")
        print("  This could mean the mask isn't being applied, or padding is minimal.")

    return with_mask_pass


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    success = test_batch_padding_attention_mask()
    exit(0 if success else 1)
