"""
Utilities for mapping sequence tokens to tokenized positions.
"""


def find_sequence_token_positions(
    tokenizer,
    prompt: str,
    sequence_tokens: list[str],
    target_token: str = "X",
) -> list[int]:
    """
    Find which token positions in the tokenized prompt correspond to
    each token in the original sequence.

    Uses a robust approach: reconstruct the sequence string and find it
    within the full decoded prompt, then map character positions to token positions.

    Returns a list mapping sequence index -> tokenized position.
    Returns None for sequence tokens that don't map cleanly to a single position.
    """
    # Tokenize the full prompt
    encoded = tokenizer(prompt, return_tensors="pt")
    token_ids = encoded.input_ids[0].tolist()

    # Build a mapping from character position in decoded string to token index
    # This handles any tokenizer quirks (BPE merges, space handling, etc.)
    char_to_token = []
    decoded_full = ""
    for tok_idx, tid in enumerate(token_ids):
        tok_str = tokenizer.decode([tid])
        for _ in tok_str:
            char_to_token.append(tok_idx)
        decoded_full += tok_str

    # Find where our sequence appears in the decoded prompt
    # The sequence is space-separated tokens like "A X B X X C D"
    sequence_str = " ".join(sequence_tokens)

    # Search for the sequence in the decoded string
    seq_start_char = decoded_full.find(sequence_str)

    if seq_start_char == -1:
        # Try with variations (some tokenizers add/remove spaces)
        # Try finding just the first few tokens to locate the start
        partial_seq = " ".join(sequence_tokens[:5])
        seq_start_char = decoded_full.find(partial_seq)

        if seq_start_char == -1:
            print(f"Warning: Could not find sequence in tokenized prompt")
            print(f"  Looking for: {sequence_str[:50]}...")
            print(f"  In decoded: {decoded_full[:200]}...")
            return [None] * len(sequence_tokens)

    # Now map each sequence token to its token position
    positions = []
    current_char = seq_start_char

    for i, seq_tok in enumerate(sequence_tokens):
        # Expected position in the decoded string
        if i == 0:
            tok_start = current_char
        else:
            # Account for space separator
            tok_start = current_char + 1  # skip the space

        tok_end = tok_start + len(seq_tok)

        # Get the token index for the middle of this token
        # (using middle handles edge cases where token boundaries fall on spaces)
        tok_mid = tok_start + len(seq_tok) // 2

        if tok_mid < len(char_to_token):
            positions.append(char_to_token[tok_mid])
        else:
            positions.append(None)

        # Move to end of this token
        current_char = tok_end

    # Validate: check that we got reasonable positions
    valid_positions = [p for p in positions if p is not None]
    if valid_positions:
        # Positions should be monotonically non-decreasing
        is_monotonic = all(valid_positions[i] <= valid_positions[i+1]
                          for i in range(len(valid_positions)-1))
        if not is_monotonic:
            print(f"Warning: Token positions are not monotonic, mapping may be incorrect")

    return positions
