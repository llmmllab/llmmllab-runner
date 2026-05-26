"""Yaml-driven GPU placement for in-process pipelines.

Each in-process pipeline (rembg, img23d, mesh2parts) used to either
default to ``cuda:0`` (which is the small 3060 on this cluster — too
small for Hunyuan3D-2.1 and a coin-flip OOM for rembg+img23d running
back-to-back), or carry its own bespoke "free-VRAM scan" code that
duplicated the same logic with slightly different filter rules.

This module centralises that — driven by per-model ``parameters`` in
.models.yaml.  The relevant fields:

  * ``main_gpu: <int>``      — explicit cuda device index.  ``-1``
    (the default for llama.cpp's main_gpu) is treated as
    "auto-pick by free VRAM".  Any non-negative integer pins to
    that exact device.

  * ``tensor_split: "0,1,1"`` — comma-separated weights, same shape
    as llama.cpp's tensor_split.  For in-process pipelines we don't
    actually shard a single layer across cards (PyTorch can't do
    that cheaply for arbitrary modules); we use this purely as a
    *filter*: only devices whose tensor_split weight is non-zero
    are candidates for ``main_gpu: -1`` auto-selection.

  * ``min_vram_gb: <float>`` — minimum *total* VRAM a candidate
    must have to be eligible.  Set this on memory-heavy pipelines
    (mesh2parts needs 20 GB+) so the auto-picker can't land on a
    GPU it'll OOM on at load time.

Resolution order, applied in :func:`pick_device`:

  1. Explicit ``main_gpu`` (>= 0) → cuda:N, no questions asked.
  2. Build candidate set from all CUDA devices that:
     - Pass the ``min_vram_gb`` floor (default 0 = no filter).
     - Have non-zero ``tensor_split`` weight (default = all eligible).
  3. Pick the candidate with the most free VRAM right now.
  4. Fall back to "cpu" if no CUDA candidate.

Returns a ``DeviceChoice`` with the picked device plus the second-most-free
candidate (for pipelines like mesh2parts that shard across two cards).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DeviceChoice:
    primary: str
    secondary: Optional[str] = None
    free_bytes_primary: int = 0
    free_bytes_secondary: int = 0


def _parse_tensor_split(spec: Optional[str], n: int) -> List[float]:
    """Parse llama.cpp-style ``"0,1,1"`` → ``[0.0, 1.0, 1.0]``.

    Pads / truncates to ``n`` entries.  Missing entries are treated
    as 1.0 (eligible) so a partial spec like ``"0"`` only excludes
    cuda:0 from the auto-picker.
    """
    if not spec:
        return [1.0] * n
    weights: List[float] = []
    for tok in spec.split(","):
        tok = tok.strip()
        try:
            weights.append(float(tok))
        except ValueError:
            weights.append(0.0)
    if len(weights) < n:
        weights.extend([1.0] * (n - len(weights)))
    return weights[:n]


def pick_device(
    *,
    main_gpu: Optional[int] = None,
    tensor_split: Optional[str] = None,
    min_vram_gb: float = 0.0,
    logger=None,
) -> DeviceChoice:
    """Return the best CUDA device(s) for a pipeline given yaml hints.

    Args mirror the yaml field names so a caller can pass
    ``**model.parameters.model_dump()`` (filtering to the relevant
    keys) when wiring this up.
    """
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return DeviceChoice(primary="cpu")

    if not torch.cuda.is_available():
        return DeviceChoice(primary="cpu")

    # Explicit pin wins.
    if main_gpu is not None and main_gpu >= 0:
        device = f"cuda:{main_gpu}"
        try:
            free, _total = torch.cuda.mem_get_info(main_gpu)
        except Exception:
            free = 0
        if logger is not None:
            logger.info(
                f"Pinned to {device} via .models.yaml main_gpu "
                f"({free / 1024**3:.1f} GB free)"
            )
        return DeviceChoice(primary=device, free_bytes_primary=free)

    # Auto-pick path.  Score = free VRAM; filter by min_vram_gb +
    # tensor_split (any device whose weight is 0.0 is excluded).
    n = torch.cuda.device_count()
    weights = _parse_tensor_split(tensor_split, n)
    min_bytes = int(min_vram_gb * (1024 ** 3))

    candidates: List[tuple[int, int]] = []  # (free_bytes, idx)
    for i in range(n):
        if weights[i] == 0.0:
            if logger is not None:
                logger.debug(
                    f"Skipping cuda:{i}: tensor_split weight is 0 in yaml"
                )
            continue
        try:
            free, total = torch.cuda.mem_get_info(i)
        except Exception:
            continue
        if total < min_bytes:
            if logger is not None:
                logger.debug(
                    f"Skipping cuda:{i}: total {total / 1024**3:.1f} GB "
                    f"< min_vram_gb {min_vram_gb}"
                )
            continue
        candidates.append((free, i))

    if not candidates:
        if logger is not None:
            logger.warning(
                f"No CUDA device meets the yaml constraints "
                f"(min_vram_gb={min_vram_gb}, tensor_split={tensor_split!r}); "
                f"falling back to CPU"
            )
        return DeviceChoice(primary="cpu")

    # Sort by free VRAM descending → primary is most free.
    candidates.sort(reverse=True)
    primary_free, primary_idx = candidates[0]
    secondary_device = None
    secondary_free = 0
    if len(candidates) > 1:
        secondary_free, secondary_idx = candidates[1]
        secondary_device = f"cuda:{secondary_idx}"

    # Pin torch's process-default cuda device to primary so submodules
    # that construct tensors with bare ``cuda`` (no explicit index)
    # land consistently.  Without this, some upstream code paths
    # spawn intermediate tensors on cuda:0 even after the model is
    # moved elsewhere — and the subsequent op trips NCCL with an
    # "unhandled system error".
    try:
        torch.cuda.set_device(primary_idx)
    except Exception:
        pass

    if logger is not None:
        msg = (
            f"Selected cuda:{primary_idx} primary "
            f"({primary_free / 1024**3:.1f} GB free)"
        )
        if secondary_device is not None:
            msg += (
                f"; cuda:{secondary_idx} secondary "
                f"({secondary_free / 1024**3:.1f} GB free)"
            )
        logger.info(msg)

    return DeviceChoice(
        primary=f"cuda:{primary_idx}",
        secondary=secondary_device,
        free_bytes_primary=primary_free,
        free_bytes_secondary=secondary_free,
    )


def device_hints_from_model(model) -> dict:
    """Extract the relevant fields from a Model into kwargs for
    :func:`pick_device`.  Tolerant of missing ``parameters``.
    """
    if model is None:
        return {}
    params = getattr(model, "parameters", None)
    if params is None:
        return {}
    out = {}
    main_gpu = getattr(params, "main_gpu", None)
    if main_gpu is not None:
        out["main_gpu"] = int(main_gpu)
    tensor_split = getattr(params, "tensor_split", None)
    if tensor_split:
        out["tensor_split"] = str(tensor_split)
    # ``min_vram_gb`` is read out of an env var fall-back inside
    # pick_device's callers (each pipeline has its own minimum); it
    # isn't currently a yaml field.  Add a yaml field for it if you
    # want per-model control later.
    return out
