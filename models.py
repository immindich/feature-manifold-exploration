AVAILABLE_MODELS = {
    "smollm2-135m": {
        # Tiny ungated model, useful for local smoke tests on machines that
        # can't fit the Gemma models.
        "name": "SmolLM2-135M-Instruct",
        "path": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "type": "local",
    },
    "gemma-270m": {
        "name": "gemma-3-270m-it",
        "path": "google/gemma-3-270m-it",
        "type": "local",
    },
    "gemma-4-E4B": {
        # Aliased to the instruction-tuned variant (the base model has no
        # chat template, which the counting eval needs).
        "name": "gemma-4-E4B-it",
        "path": "google/gemma-4-E4B-it",
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
