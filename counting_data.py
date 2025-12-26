"""
Shared data generation and prompt formatting for counting tasks.

This module provides consistent example generation and prompt formatting
for both evaluation (eval_counting.py) and activation extraction (activations.py).
"""

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CountingSequence:
    """A sequence with known count and metadata."""
    tokens: list[str]
    sequence: str
    true_count: int
    target_token: str = "X"
    counts_at_position: list[int] | None = None  # Running count after each token

    @property
    def sequence_length(self) -> int:
        return len(self.tokens)

    @property
    def density(self) -> float:
        return self.true_count / len(self.tokens) if self.tokens else 0.0


def generate_counting_example(
    target_token: str = "X",
    other_tokens: list[str] | None = None,
    min_length: int = 10,
    max_length: int = 30,
    target_freq: float = 0.2,
    vary_freq: bool = True,
    track_positions: bool = False,
) -> CountingSequence:
    """Generate a random sequence with a known count of target tokens.

    Args:
        target_token: The token to count (default: "X")
        other_tokens: List of distractor tokens (default: ["A", "B", "C", "D", "E"])
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        target_freq: Base frequency of target token (used if vary_freq=False)
        vary_freq: If True, randomize target_freq per example (0.05 to 0.5) to
                   decouple count from sequence length.
        track_positions: If True, also track running count at each position

    Returns:
        CountingSequence with the generated sequence and ground truth count
    """
    if other_tokens is None:
        other_tokens = ["A", "B", "C", "D", "E"]

    length = random.randint(min_length, max_length)

    # Vary frequency per example to decouple count from length
    if vary_freq:
        target_freq = random.uniform(0.05, 0.5)

    tokens = []
    counts_at_position = [] if track_positions else None
    running_count = 0

    for _ in range(length):
        if random.random() < target_freq:
            tokens.append(target_token)
            running_count += 1
        else:
            tokens.append(random.choice(other_tokens))

        if track_positions:
            counts_at_position.append(running_count)

    sequence = " ".join(tokens)

    return CountingSequence(
        tokens=tokens,
        sequence=sequence,
        true_count=running_count,
        target_token=target_token,
        counts_at_position=counts_at_position,
    )


def create_prompt(example: CountingSequence) -> str:
    """Create a prompt for the counting task.

    Args:
        example: The counting example

    Returns:
        Formatted prompt string
    """
    return (
        f"Count how many times the token '{example.target_token}' appears in this sequence:\n\n"
        f"{example.sequence}\n\n"
        f"Respond with ONLY the number. Do not include any explanation, words, or other text."
    )


def generate_sequence_with_target_count(
    target_count: int,
    length_range: tuple[int, int],
    target_token: str = "X",
    other_tokens: list[str] | None = None,
    density_range: tuple[float, float] = (0.05, 0.8),
    max_attempts: int = 50,
) -> Optional[CountingSequence]:
    """
    Generate a sequence with exactly the target count.

    Args:
        target_count: Desired number of target tokens
        length_range: (min_length, max_length) for sequence
        target_token: Token to count
        other_tokens: Other tokens to include
        density_range: Acceptable range for count/length ratio.
                      Wider range = lower count-length correlation.
                      Default (0.05, 0.8) gives r ≈ 0.4
        max_attempts: Max attempts before giving up

    Returns:
        CountingSequence or None if can't generate valid sequence
    """
    if other_tokens is None:
        other_tokens = ["A", "B", "C", "D", "E"]

    min_len, max_len = length_range
    min_density, max_density = density_range

    # Calculate valid length range for this count
    # length must satisfy: min_density <= count/length <= max_density
    # So: count/max_density <= length <= count/min_density
    if target_count == 0:
        # Special case: just generate sequence with no target tokens
        length = random.randint(min_len, max_len)
        tokens = [random.choice(other_tokens) for _ in range(length)]
        return CountingSequence(
            tokens=tokens,
            sequence=" ".join(tokens),
            true_count=0,
            target_token=target_token,
        )

    valid_len_min = max(min_len, int(np.ceil(target_count / max_density)))
    valid_len_max = min(max_len, int(np.floor(target_count / min_density)))

    if valid_len_min > valid_len_max:
        # Can't generate this count within constraints
        return None

    for _ in range(max_attempts):
        # Pick a length
        length = random.randint(valid_len_min, valid_len_max)

        # Generate sequence with exactly target_count target tokens
        tokens = [target_token] * target_count + \
                 [random.choice(other_tokens) for _ in range(length - target_count)]

        # Shuffle to randomize positions
        random.shuffle(tokens)

        actual_count = tokens.count(target_token)
        assert actual_count == target_count, f"Count mismatch: {actual_count} vs {target_count}"

        return CountingSequence(
            tokens=tokens,
            sequence=" ".join(tokens),
            true_count=target_count,
            target_token=target_token,
        )

    return None


def generate_uniform_count_sequences(
    min_count: int,
    max_count: int,
    num_sequences: int,
    length_range: tuple[int, int] = (50, 500),
    target_token: str = "X",
    other_tokens: list[str] | None = None,
    density_range: tuple[float, float] = (0.05, 0.8),
    seed: Optional[int] = None,
) -> list[CountingSequence]:
    """
    Generate sequences with approximately uniform distribution of counts.

    Args:
        min_count: Minimum count value
        max_count: Maximum count value
        num_sequences: Total number of sequences to generate
        length_range: (min_length, max_length) for sequences
        target_token: Token to count
        other_tokens: Other tokens to include
        density_range: Acceptable range for count/length ratio.
                      Default (0.05, 0.8) gives count-length correlation ≈ 0.4
        seed: Random seed for reproducibility

    Returns:
        List of CountingSequence objects
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if other_tokens is None:
        other_tokens = ["A", "B", "C", "D", "E"]

    sequences = []

    # Generate target counts uniformly
    target_counts = np.random.randint(min_count, max_count + 1, size=num_sequences)

    failed_counts = []
    for target_count in target_counts:
        seq = generate_sequence_with_target_count(
            target_count=target_count,
            length_range=length_range,
            target_token=target_token,
            other_tokens=other_tokens,
            density_range=density_range,
        )

        if seq is not None:
            sequences.append(seq)
        else:
            failed_counts.append(target_count)

    if failed_counts:
        print(f"Warning: Could not generate {len(failed_counts)} sequences")
        print(f"  Failed counts: {set(failed_counts)}")

    return sequences


def generate_stratified_sequences(
    count_bins: list[tuple[int, int]],
    sequences_per_bin: int,
    length_range: tuple[int, int] = (50, 500),
    target_token: str = "X",
    other_tokens: list[str] | None = None,
    density_range: tuple[float, float] = (0.05, 0.8),
    seed: Optional[int] = None,
) -> list[CountingSequence]:
    """
    Generate sequences with stratified sampling across count bins.

    Ensures equal representation across different count ranges.

    Args:
        count_bins: List of (min_count, max_count) tuples defining bins
        sequences_per_bin: Number of sequences to generate per bin
        length_range: (min_length, max_length) for sequences
        target_token: Token to count
        other_tokens: Other tokens to include
        density_range: Acceptable range for count/length ratio
        seed: Random seed for reproducibility

    Returns:
        List of CountingSequence objects
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if other_tokens is None:
        other_tokens = ["A", "B", "C", "D", "E"]

    sequences = []

    for bin_min, bin_max in count_bins:
        bin_sequences = generate_uniform_count_sequences(
            min_count=bin_min,
            max_count=bin_max,
            num_sequences=sequences_per_bin,
            length_range=length_range,
            target_token=target_token,
            other_tokens=other_tokens,
            density_range=density_range,
        )
        sequences.extend(bin_sequences)
        print(f"  Bin [{bin_min:>3}, {bin_max:>3}]: generated {len(bin_sequences)} sequences")

    return sequences


def analyze_distribution(sequences: list[CountingSequence]) -> dict:
    """Analyze and print statistics about generated sequences."""
    counts = [s.true_count for s in sequences]
    lengths = [s.sequence_length for s in sequences]
    densities = [s.density for s in sequences]

    # Correlation between count and length
    corr = np.corrcoef(counts, lengths)[0, 1]

    stats = {
        'n': len(sequences),
        'count_range': (min(counts), max(counts)),
        'count_mean': np.mean(counts),
        'count_std': np.std(counts),
        'length_range': (min(lengths), max(lengths)),
        'length_mean': np.mean(lengths),
        'count_length_correlation': corr,
        'density_range': (min(densities), max(densities)),
        'density_mean': np.mean(densities),
    }

    print("\nGenerated sequence statistics:")
    print(f"  N sequences: {stats['n']}")
    print(f"  Count range: [{stats['count_range'][0]}, {stats['count_range'][1]}]")
    print(f"  Count mean: {stats['count_mean']:.1f} ± {stats['count_std']:.1f}")
    print(f"  Length range: [{stats['length_range'][0]}, {stats['length_range'][1]}]")
    print(f"  Correlation(count, length): {stats['count_length_correlation']:.3f}")
    print(f"  Density range: [{stats['density_range'][0]:.3f}, {stats['density_range'][1]:.3f}]")

    return stats
