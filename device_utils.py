"""Device selection helpers that work on CUDA, Apple MPS, and CPU."""

import torch


def disable_mps_allocator_warmup() -> None:
    """Skip transformers' allocator warmup on MPS.

    The warmup tries to pre-allocate the full model size as a single tensor
    so subsequent loads reuse the cached buffer. On MPS this single
    allocation exceeds the per-buffer Metal limit (~14 GiB on many Macs),
    so loading models above that size fails before any layer copy happens.
    The warmup is purely a load-speed optimization, so making it a no-op
    is safe — the only cost is slightly slower first-time loading.
    """
    try:
        import transformers.modeling_utils as _mu
    except ImportError:
        return

    original = getattr(_mu, "caching_allocator_warmup", None)
    if original is None or getattr(original, "_mps_safe", False):
        return

    def _patched(model, expanded_device_map, hf_quantizer):
        # Skip if any target device is MPS; otherwise fall back to original.
        for dev in expanded_device_map.values():
            if str(dev).startswith("mps"):
                return
        return original(model, expanded_device_map, hf_quantizer)

    _patched._mps_safe = True  # type: ignore[attr-defined]
    _mu.caching_allocator_warmup = _patched


def get_device() -> str:
    """Return the best available accelerator: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def empty_cache(device: str | None = None) -> None:
    """Free cached memory for the given accelerator if applicable."""
    if device is None:
        device = get_device()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
