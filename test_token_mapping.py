"""
Tests for token_mapping.py
"""

import pytest
from transformers import AutoTokenizer

from token_mapping import find_sequence_token_positions
from counting_data import CountingSequence, create_prompt, format_chat_prompt


@pytest.fixture
def gpt2_tokenizer():
    """GPT-2 tokenizer for testing."""
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def gemma_tokenizer():
    """Gemma tokenizer for testing."""
    return AutoTokenizer.from_pretrained("google/gemma-3-12b-it")


class TestFindSequenceTokenPositions:
    def test_simple_sequence_gpt2(self, gpt2_tokenizer):
        """Test with a simple sequence in GPT-2."""
        sequence_tokens = ["A", "B", "C", "D"]
        prompt = "Count: A B C D"

        positions = find_sequence_token_positions(
            gpt2_tokenizer, prompt, sequence_tokens
        )

        # Should return 4 positions, one for each token
        assert len(positions) == 4
        # All positions should be valid (not None)
        assert all(p is not None for p in positions)
        # Positions should be monotonically increasing
        assert positions == sorted(positions)

    def test_with_target_token(self, gpt2_tokenizer):
        """Test sequence containing the target token X."""
        sequence_tokens = ["A", "X", "B", "X", "C"]
        prompt = "Count X in: A X B X C"

        positions = find_sequence_token_positions(
            gpt2_tokenizer, prompt, sequence_tokens, target_token="X"
        )

        assert len(positions) == 5
        assert all(p is not None for p in positions)
        # Positions should be monotonically non-decreasing
        for i in range(len(positions) - 1):
            assert positions[i] <= positions[i + 1]

    def test_longer_sequence(self, gpt2_tokenizer):
        """Test with a longer sequence."""
        sequence_tokens = ["A", "B", "X", "C", "D", "X", "E", "F", "X", "G"]
        prompt = f"Sequence: {' '.join(sequence_tokens)}"

        positions = find_sequence_token_positions(
            gpt2_tokenizer, prompt, sequence_tokens
        )

        assert len(positions) == 10
        valid_positions = [p for p in positions if p is not None]
        # Should have mostly valid mappings
        assert len(valid_positions) >= 8

    def test_sequence_not_found_returns_nones(self, gpt2_tokenizer):
        """Test that missing sequence returns list of Nones."""
        sequence_tokens = ["Z", "Z", "Z"]
        prompt = "This prompt has no Z tokens"

        positions = find_sequence_token_positions(
            gpt2_tokenizer, prompt, sequence_tokens
        )

        assert len(positions) == 3
        assert all(p is None for p in positions)

    def test_positions_are_integers(self, gpt2_tokenizer):
        """Test that valid positions are integers."""
        sequence_tokens = ["A", "B", "C"]
        prompt = "Test: A B C"

        positions = find_sequence_token_positions(
            gpt2_tokenizer, prompt, sequence_tokens
        )

        for p in positions:
            if p is not None:
                assert isinstance(p, int)

    def test_positions_within_token_bounds(self, gpt2_tokenizer):
        """Test that positions are within valid token range."""
        sequence_tokens = ["A", "B", "C", "D", "E"]
        prompt = "Items: A B C D E"

        positions = find_sequence_token_positions(
            gpt2_tokenizer, prompt, sequence_tokens
        )

        # Tokenize to get total number of tokens
        encoded = gpt2_tokenizer(prompt, return_tensors="pt")
        num_tokens = encoded.input_ids.shape[1]

        for p in positions:
            if p is not None:
                assert 0 <= p < num_tokens

    def test_gemma_tokenizer(self, gemma_tokenizer):
        """Test with Gemma tokenizer."""
        sequence_tokens = ["A", "X", "B", "X", "C"]
        prompt = f"Count how many X: {' '.join(sequence_tokens)}"

        positions = find_sequence_token_positions(
            gemma_tokenizer, prompt, sequence_tokens
        )

        assert len(positions) == 5
        valid_positions = [p for p in positions if p is not None]
        # Should get at least some valid positions
        assert len(valid_positions) >= 3

    def test_counting_task_prompt_format(self, gpt2_tokenizer):
        """Test with the actual counting task prompt format."""
        sequence_tokens = ["A", "X", "B", "X", "X", "C", "D"]
        sequence_str = " ".join(sequence_tokens)
        prompt = (
            f"Count how many times the token 'X' appears in this sequence:\n\n"
            f"{sequence_str}\n\n"
            f"Respond with ONLY the number."
        )

        positions = find_sequence_token_positions(
            gpt2_tokenizer, prompt, sequence_tokens, target_token="X"
        )

        assert len(positions) == 7
        # Verify we can find the sequence in this format
        valid_positions = [p for p in positions if p is not None]
        assert len(valid_positions) >= 5

    def test_with_counting_sequence_dataclass(self, gpt2_tokenizer):
        """Test with actual CountingSequence from counting_data.py."""
        tokens = ["A", "X", "B", "X", "X", "C", "D", "X"]
        example = CountingSequence(
            tokens=tokens,
            sequence=" ".join(tokens),
            true_count=4,
            target_token="X",
        )

        prompt = create_prompt(example)
        positions = find_sequence_token_positions(
            gpt2_tokenizer, prompt, example.tokens, target_token=example.target_token
        )

        assert len(positions) == len(tokens)
        valid_positions = [p for p in positions if p is not None]
        assert len(valid_positions) >= 6
        # Positions should be monotonically non-decreasing
        for i in range(len(valid_positions) - 1):
            assert valid_positions[i] <= valid_positions[i + 1]

    def test_with_counting_sequence_gemma(self, gemma_tokenizer):
        """Test CountingSequence with Gemma tokenizer and chat template."""
        tokens = ["A", "X", "B", "X", "C", "X", "D", "E", "X", "F"]
        example = CountingSequence(
            tokens=tokens,
            sequence=" ".join(tokens),
            true_count=4,
            target_token="X",
        )

        # Use the full chat-formatted prompt like activations_prototype.py does
        prompt = format_chat_prompt(example, gemma_tokenizer)
        positions = find_sequence_token_positions(
            gemma_tokenizer, prompt, example.tokens, target_token=example.target_token
        )

        assert len(positions) == len(tokens)
        valid_positions = [p for p in positions if p is not None]
        # Should map most tokens successfully
        assert len(valid_positions) >= 7
        # Verify positions are within token bounds
        encoded = gemma_tokenizer(prompt, return_tensors="pt")
        num_tokens = encoded.input_ids.shape[1]
        for p in valid_positions:
            assert 0 <= p < num_tokens

    def test_with_random_target_token(self, gemma_tokenizer):
        """Test with a non-X target token (like counting_data can generate)."""
        tokens = ["A", "B", "M", "C", "M", "D", "M", "E"]
        example = CountingSequence(
            tokens=tokens,
            sequence=" ".join(tokens),
            true_count=3,
            target_token="M",
        )

        prompt = create_prompt(example)
        positions = find_sequence_token_positions(
            gemma_tokenizer, prompt, example.tokens, target_token=example.target_token
        )

        assert len(positions) == len(tokens)
        valid_positions = [p for p in positions if p is not None]
        assert len(valid_positions) >= 6


class TestMappingCorrectness:
    """Tests that verify the mapped positions actually contain the expected tokens."""

    def _get_token_at_position(self, tokenizer, prompt: str, position: int) -> str:
        """Decode the token at a specific position."""
        encoded = tokenizer(prompt, return_tensors="pt")
        token_id = encoded.input_ids[0][position].item()
        return tokenizer.decode([token_id])

    def test_mapped_tokens_contain_sequence_chars_gpt2(self, gpt2_tokenizer):
        """Verify that decoded tokens at mapped positions contain the sequence characters."""
        tokens = ["A", "B", "C", "D", "E"]
        prompt = f"Sequence: {' '.join(tokens)}"

        positions = find_sequence_token_positions(
            gpt2_tokenizer, prompt, tokens
        )

        for i, (seq_tok, pos) in enumerate(zip(tokens, positions)):
            if pos is not None:
                decoded = self._get_token_at_position(gpt2_tokenizer, prompt, pos)
                assert seq_tok in decoded, (
                    f"Token '{seq_tok}' not found in decoded token '{decoded}' at position {pos}"
                )

    def test_mapped_tokens_contain_sequence_chars_gemma(self, gemma_tokenizer):
        """Verify that decoded tokens at mapped positions contain the sequence characters (Gemma)."""
        tokens = ["A", "X", "B", "X", "C"]
        example = CountingSequence(
            tokens=tokens,
            sequence=" ".join(tokens),
            true_count=2,
            target_token="X",
        )
        prompt = create_prompt(example)

        positions = find_sequence_token_positions(
            gemma_tokenizer, prompt, tokens, target_token="X"
        )

        for i, (seq_tok, pos) in enumerate(zip(tokens, positions)):
            if pos is not None:
                decoded = self._get_token_at_position(gemma_tokenizer, prompt, pos)
                assert seq_tok in decoded, (
                    f"Token '{seq_tok}' not found in decoded token '{decoded}' at position {pos}"
                )

    def test_mapped_tokens_with_chat_template(self, gemma_tokenizer):
        """Verify mappings are correct even with chat template formatting."""
        tokens = ["A", "X", "B", "C", "X", "D", "E", "F", "X", "G"]
        example = CountingSequence(
            tokens=tokens,
            sequence=" ".join(tokens),
            true_count=3,
            target_token="X",
        )
        prompt = format_chat_prompt(example, gemma_tokenizer)

        positions = find_sequence_token_positions(
            gemma_tokenizer, prompt, tokens, target_token="X"
        )

        correct_mappings = 0
        for i, (seq_tok, pos) in enumerate(zip(tokens, positions)):
            if pos is not None:
                decoded = self._get_token_at_position(gemma_tokenizer, prompt, pos)
                if seq_tok in decoded:
                    correct_mappings += 1

        # At least 80% of mappings should be correct
        assert correct_mappings >= len(tokens) * 0.8, (
            f"Only {correct_mappings}/{len(tokens)} mappings were correct"
        )

    def test_target_token_positions_are_correct(self, gemma_tokenizer):
        """Specifically verify that target token (X) positions are correctly mapped."""
        tokens = ["A", "X", "B", "X", "C", "X", "D"]
        example = CountingSequence(
            tokens=tokens,
            sequence=" ".join(tokens),
            true_count=3,
            target_token="X",
        )
        prompt = create_prompt(example)

        positions = find_sequence_token_positions(
            gemma_tokenizer, prompt, tokens, target_token="X"
        )

        # Check specifically the X positions (indices 1, 3, 5)
        x_indices = [i for i, t in enumerate(tokens) if t == "X"]
        for idx in x_indices:
            pos = positions[idx]
            if pos is not None:
                decoded = self._get_token_at_position(gemma_tokenizer, prompt, pos)
                assert "X" in decoded, (
                    f"Target token 'X' not found at position {pos}, got '{decoded}'"
                )

    def test_all_letters_map_correctly(self, gpt2_tokenizer):
        """Test that all uppercase letters map correctly."""
        import string
        # Use a subset of letters to keep it manageable
        tokens = list("ABCDEFGHIJ")
        prompt = f"Letters: {' '.join(tokens)}"

        positions = find_sequence_token_positions(
            gpt2_tokenizer, prompt, tokens
        )

        for seq_tok, pos in zip(tokens, positions):
            if pos is not None:
                decoded = self._get_token_at_position(gpt2_tokenizer, prompt, pos)
                assert seq_tok in decoded, (
                    f"Letter '{seq_tok}' not found in decoded token '{decoded}'"
                )

    def test_longer_sequence_mapping_accuracy(self, gemma_tokenizer):
        """Test mapping accuracy on a longer sequence typical of counting tasks."""
        # Generate a realistic sequence
        tokens = ["A", "X", "B", "C", "X", "D", "E", "X", "F", "G",
                  "H", "X", "I", "J", "K", "X", "L", "M", "N", "X"]
        example = CountingSequence(
            tokens=tokens,
            sequence=" ".join(tokens),
            true_count=6,
            target_token="X",
        )
        prompt = create_prompt(example)

        positions = find_sequence_token_positions(
            gemma_tokenizer, prompt, tokens, target_token="X"
        )

        correct = 0
        total_valid = 0
        for seq_tok, pos in zip(tokens, positions):
            if pos is not None:
                total_valid += 1
                decoded = self._get_token_at_position(gemma_tokenizer, prompt, pos)
                if seq_tok in decoded:
                    correct += 1

        # Report accuracy
        accuracy = correct / total_valid if total_valid > 0 else 0
        assert accuracy >= 0.9, (
            f"Mapping accuracy {accuracy:.1%} is below 90% threshold "
            f"({correct}/{total_valid} correct)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
