"""
Shared data generation and prompt formatting for counting tasks.

This module provides consistent example generation and prompt formatting
for both evaluation (eval_counting.py) and activation extraction (activations.py).
"""

import random
from dataclasses import dataclass


@dataclass
class CountingExample:
    """A counting example with sequence and ground truth."""
    sequence: str
    target_token: str
    true_count: int
    tokens: list[str]
    counts_at_position: list[int] | None = None  # Running count after each token


def generate_counting_example(
    target_token: str = "X",
    other_tokens: list[str] | None = None,
    min_length: int = 10,
    max_length: int = 30,
    target_freq: float = 0.2,
    vary_freq: bool = True,
    track_positions: bool = False,
) -> CountingExample:
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
        CountingExample with the generated sequence and ground truth count
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

    return CountingExample(
        sequence=sequence,
        target_token=target_token,
        true_count=running_count,
        tokens=tokens,
        counts_at_position=counts_at_position,
    )


def create_prompt(example: CountingExample) -> str:
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
