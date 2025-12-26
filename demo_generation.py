#!/usr/bin/env python3
"""
Demo script for visualizing sequence generation methods.

Compares uniform count distribution and stratified sampling approaches,
showing the resulting distributions of counts and lengths.

Usage:
    python demo_generation.py
"""

import matplotlib.pyplot as plt

from counting_data import (
    analyze_distribution,
    generate_stratified_sequences,
    generate_uniform_count_sequences,
)


def main():
    print("=" * 60)
    print("DEMO: Generating sequences with uniform count distribution")
    print("=" * 60)

    # Method 1: Uniform counts with decorrelated lengths
    print("\n1. Uniform count distribution [0, 100]:")
    print("   (density_range=(0.05, 0.8) for low count-length correlation)")
    seqs_uniform = generate_uniform_count_sequences(
        min_count=0,
        max_count=100,
        num_sequences=500,
        length_range=(30, 400),
        density_range=(0.05, 0.8),  # Wide range for low correlation
        seed=42,
    )
    stats_uniform = analyze_distribution(seqs_uniform)

    # Method 2: Stratified bins for guaranteed coverage
    print("\n2. Stratified sampling across bins:")
    bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    seqs_stratified = generate_stratified_sequences(
        count_bins=bins,
        sequences_per_bin=100,
        length_range=(30, 400),
        density_range=(0.05, 0.8),
        seed=42,
    )
    stats_stratified = analyze_distribution(seqs_stratified)

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Uniform method
    counts_u = [s.true_count for s in seqs_uniform]
    lengths_u = [s.sequence_length for s in seqs_uniform]

    axes[0, 0].hist(counts_u, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Count')
    axes[0, 0].set_title('Uniform: Count Distribution')

    axes[0, 1].scatter(lengths_u, counts_u, alpha=0.3, s=10)
    axes[0, 1].set_xlabel('Length')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title(f'Uniform: Count vs Length (r={stats_uniform["count_length_correlation"]:.3f})')

    axes[0, 2].hist(lengths_u, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 2].set_xlabel('Length')
    axes[0, 2].set_title('Uniform: Length Distribution')

    # Stratified method
    counts_s = [s.true_count for s in seqs_stratified]
    lengths_s = [s.sequence_length for s in seqs_stratified]

    axes[1, 0].hist(counts_s, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Count')
    axes[1, 0].set_title('Stratified: Count Distribution')

    axes[1, 1].scatter(lengths_s, counts_s, alpha=0.3, s=10)
    axes[1, 1].set_xlabel('Length')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title(f'Stratified: Count vs Length (r={stats_stratified["count_length_correlation"]:.3f})')

    axes[1, 2].hist(lengths_s, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 2].set_xlabel('Length')
    axes[1, 2].set_title('Stratified: Length Distribution')

    plt.tight_layout()
    plt.savefig('generation_methods_comparison.png', dpi=150)
    print("\nSaved generation_methods_comparison.png")


if __name__ == "__main__":
    main()
