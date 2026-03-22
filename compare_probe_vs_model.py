#!/usr/bin/env python3
"""Compare linear/MLP probe predictions vs actual model predictions on the same test set."""

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from counting_data import CountingSequence, create_prompt, format_chat_prompt
from eval_counting import extract_count_from_response
from models import AVAILABLE_MODELS
from train_probes import LinearProbe, MLPProbe, load_data, split_train_val_test


def compute_metrics(true, pred):
    """Compute bias-corrected metrics: fit a linear regression from pred to true,
    then report residual MSE, MAE, R², and Pearson correlation."""
    true, pred = np.array(true, dtype=float), np.array(pred, dtype=float)
    coeffs = np.polyfit(pred, true, 1)
    corrected = np.polyval(coeffs, pred)
    mse = np.mean((corrected - true) ** 2)
    mae = np.mean(np.abs(corrected - true))
    ss_res = np.sum((true - corrected) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    corr = np.corrcoef(true, pred)[0, 1] if len(true) > 1 else 0.0
    slope = coeffs[0]
    return {"mse": mse, "mae": mae, "r2": r2, "corr": corr, "slope": slope, "n": len(true)}


def print_comparison_table(rows, title):
    print(f"\n{'=' * 90}")
    print(f"  {title}  (bias-corrected: linear fit from pred -> true)")
    print(f"{'=' * 90}")
    print(f"  {'Method':<25} {'MSE':>10} {'MAE':>10} {'R²':>10} {'Corr':>10} {'Slope':>8} {'N':>6}")
    print(f"  {'-' * 81}")
    for row in rows:
        print(f"  {row['label']:<25} {row['mse']:>10.1f} {row['mae']:>10.2f} {row['r2']:>10.4f} {row['corr']:>10.4f} {row['slope']:>8.2f} {row['n']:>6}")
    print(f"{'=' * 90}")


# --- Load activations and get the test split indices ---
print("Loading activations...")
activations, metadata, layers = load_data("activations-27b.pt")
_, _, _, _, test_acts, test_counts = split_train_val_test(activations, metadata)
test_counts_np = np.array(test_counts)
n_test = len(test_counts)
print(f"Test set: {n_test} samples")

# --- Reconstruct test sequences and run model on them ---
# Recover test indices using the same split logic
from collections import defaultdict
count_to_indices = defaultdict(list)
for i, meta in enumerate(metadata):
    count_to_indices[meta["true_count"]].append(i)

rng = np.random.default_rng(42)
test_idx = []
for count in sorted(count_to_indices.keys()):
    indices = np.array(count_to_indices[count])
    rng.shuffle(indices)
    test_idx.extend(indices[:10])  # test_per_count=10

test_metadata = [metadata[i] for i in test_idx]

# Build CountingSequence objects from metadata
test_sequences = []
for m in test_metadata:
    seq = CountingSequence(
        tokens=m["tokens"],
        sequence=m["sequence"],
        true_count=m["true_count"],
        target_token=m["target_token"],
    )
    test_sequences.append(seq)

# Run Gemma-27B on the test sequences
model_config = AVAILABLE_MODELS["gemma-27b"]
model_path = model_config["path"]
print(f"\nLoading {model_config['name']}...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model.eval()

batch_size = 10
model_preds = np.full(n_test, np.nan)

# Sort by length for efficient batching, track original indices
sorted_order = sorted(range(n_test), key=lambda i: test_sequences[i].sequence_length, reverse=True)

pbar = tqdm(total=n_test, desc="  Evaluating model")
for batch_start in range(0, n_test, batch_size):
    batch_indices = sorted_order[batch_start:batch_start + batch_size]
    batch_seqs = [test_sequences[i] for i in batch_indices]
    prompts = [format_chat_prompt(seq, tokenizer) for seq in batch_seqs]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
    input_lengths = [inputs.attention_mask[i].sum().item() for i in range(len(batch_seqs))]

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=10, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    for i, orig_idx in enumerate(batch_indices):
        padded_input_len = inputs.input_ids.shape[1]
        new_tokens = outputs[i][padded_input_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        predicted = extract_count_from_response(response)
        model_preds[orig_idx] = predicted if predicted is not None else np.nan

    pbar.update(len(batch_indices))
pbar.close()

del model
torch.cuda.empty_cache()

# Filter out unparsed model predictions
model_valid = ~np.isnan(model_preds)
print(f"Model parse rate: {model_valid.sum()}/{n_test}")

# --- Load probe weights and get predictions ---
device = "cuda" if torch.cuda.is_available() else "cpu"

probe_results = {}  # {(type, layer): predictions}
for probe_type, weights_path in [("linear", "probe-linear-27b.pt"), ("mlp", "probe-mlp-27b.pt")]:
    saved = torch.load(weights_path, weights_only=False)
    hidden_dim = saved["hidden_dim"]
    for layer in saved["layers"]:
        if probe_type == "linear":
            probe = LinearProbe(hidden_dim)
        else:
            probe = MLPProbe(hidden_dim, saved["mlp_hidden"])
        probe.load_state_dict(saved["probes"][layer])
        probe.eval().to(device)

        li = saved["layers"].index(layer)
        with torch.no_grad():
            pred = probe(test_acts[:, li, :].to(device)).cpu().numpy()
        probe_results[(probe_type, layer)] = pred

# --- Print comparison tables ---
ranges = [
    ("Full range (1-150)", 1, 150),
    ("Counts 1-50", 1, 50),
    ("Counts 51-100", 51, 100),
    ("Counts 101-150", 101, 150),
]

probe_layers = saved["layers"]

for title, lo, hi in ranges:
    mask = (test_counts_np >= lo) & (test_counts_np <= hi)
    mask_model_valid = mask & model_valid

    rows = []
    rows.append({"label": "Gemma-27B (model)", **compute_metrics(
        test_counts_np[mask_model_valid], model_preds[mask_model_valid])})
    for layer in probe_layers:
        rows.append({"label": f"Linear probe (L{layer})", **compute_metrics(
            test_counts_np[mask], probe_results[("linear", layer)][mask])})
    for layer in probe_layers:
        rows.append({"label": f"MLP probe (L{layer})", **compute_metrics(
            test_counts_np[mask], probe_results[("mlp", layer)][mask])})
    print_comparison_table(rows, title)
