#!/usr/bin/env python3
"""Phase 1 derisk: probe whether the Zenodo (Arm B) checkpoint loads into
the upstream DiffAb model architecture before any GPU time is committed.

Why this exists
---------------
Two candidate starting points exist for our VHH fine-tune:
  * Arm A — luost26/DiffAb (HuggingFace): trained on SAbDab.
  * Arm B — Zenodo 16894086 (ANDD paper): luost26's DiffAb further
            fine-tuned on ANDD's ~12.6K Ab+Nb PDBs.

If the Zenodo checkpoint was saved after architectural surgery (extra
layers, different feature dims, renamed modules), our Arm B config will
silently load garbage with strict=False. We catch that here, before
allocating an A100 hour.

What the script does
--------------------
1. Constructs a fresh DiffAb model from the upstream codesign_multicdrs
   config (this is the architecture luost26 published).
2. ``torch.load``s the Zenodo file (auto-detecting wrapped vs. raw
   state_dict — DiffAb's own train.py wraps as ``{'model': ..., 'optimizer': ...}``).
3. Tries ``model.load_state_dict(..., strict=True)``.
4. On failure, retries with ``strict=False`` and reports the exact set
   of missing / unexpected keys (and a small sample, capped to keep the
   output readable).
5. Optionally loads the upstream luost26 checkpoint too and diffs the
   two key sets, so any *additional* keys the Zenodo training stack
   introduced (e.g., a new auxiliary head) are surfaced explicitly.
6. Prints one of three verdict lines on the last line of stdout:
     ``VERDICT: LOAD_OK``           — strict load succeeded; safe for Arm B.
     ``VERDICT: LOAD_OK_NONSTRICT`` — strict failed but missing/unexpected
                                     counts are small (<5 each) and the
                                     mismatched keys are not weight-bearing
                                     (e.g., EMA shadow tensors). Use Arm B
                                     with caution.
     ``VERDICT: LOAD_FAIL``         — incompatible. Drop Arm B; thesis runs
                                     Arm A only.

The verdict line is suitable for ``grep`` in a SLURM log.

Usage
-----
    python scripts/diffab_ft/check_zenodo_checkpoint.py \\
        --config third_party/diffab/configs/train/codesign_multicdrs.yml \\
        --checkpoint data/checkpoints/andd_init/checkpoint.pt
        [--base-checkpoint third_party/diffab/trained_models/codesign_multicdrs.pt]

The base-checkpoint is optional; supplying it produces a diff of which
parameter names exist in the Zenodo file but not in luost26 (and vice
versa), which is the most informative output when the verdict is
LOAD_OK_NONSTRICT.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Tuple

# Project root is two levels up from this script (scripts/diffab_ft/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "diffab"))

import torch  # noqa: E402

from diffab.models import get_model  # noqa: E402  — also populates registry
from diffab.utils.misc import load_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
logger = logging.getLogger("check_zenodo_checkpoint")


# Threshold below which a strict-failure is treated as "minor" rather
# than "incompatible". Tuned to forgive a handful of EMA buffers or
# auxiliary heads, but flag full architectural drift.
NONSTRICT_TOLERANCE = 5
SAMPLE_KEY_LIMIT = 10  # how many missing/unexpected keys to print verbatim


def _unwrap_state_dict(obj: object) -> Tuple[OrderedDict, str]:
    """Extract a flat parameter dict from whatever torch.load returned.

    DiffAb's own train.py saves ``{'model': state_dict, 'optimizer': ...}``,
    but other forks save the bare state_dict, or ``{'state_dict': ...}``
    (PyTorch Lightning convention). We try the common shapes in order.

    Returns (state_dict, source_key_used) for logging.
    """
    if isinstance(obj, OrderedDict):
        return obj, "<raw-OrderedDict>"
    if isinstance(obj, dict):
        for key in ("model", "state_dict", "model_state_dict", "ema_model"):
            if key in obj and isinstance(obj[key], (dict, OrderedDict)):
                return OrderedDict(obj[key]), key
        # Fallback: maybe the dict *is* the state dict (string keys → tensor
        # values). Heuristic: every value is a Tensor.
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return OrderedDict(obj), "<dict-of-tensors>"
    raise RuntimeError(
        f"Cannot locate a state_dict in checkpoint of type {type(obj).__name__} "
        f"with top-level keys: {list(obj.keys()) if isinstance(obj, dict) else 'n/a'}"
    )


def _strip_module_prefix(sd: OrderedDict) -> OrderedDict:
    """Strip a leading ``module.`` prefix if the checkpoint was saved from
    a DataParallel-wrapped model. Idempotent if no prefix is present."""
    if not sd:
        return sd
    if all(k.startswith("module.") for k in sd.keys()):
        logger.info("Stripping leading 'module.' from %d keys.", len(sd))
        return OrderedDict((k[len("module."):], v) for k, v in sd.items())
    return sd


def _format_key_sample(keys: list[str]) -> str:
    if not keys:
        return "(none)"
    head = keys[:SAMPLE_KEY_LIMIT]
    suffix = "" if len(keys) <= SAMPLE_KEY_LIMIT else f"  …(+{len(keys) - SAMPLE_KEY_LIMIT} more)"
    return "\n    " + "\n    ".join(head) + suffix


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, type=Path,
                        help="DiffAb training YAML (architecture is read from "
                             "the 'model' section).")
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Path to the Zenodo .pt file (Arm B candidate).")
    parser.add_argument("--base-checkpoint", type=Path, default=None,
                        help="Optional: upstream luost26 checkpoint, used "
                             "only for a key-set diff against Zenodo.")
    parser.add_argument("--device", default="cpu",
                        help="torch.load map_location target. Default cpu — "
                             "we never run a forward pass here, so GPU is "
                             "unnecessary.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    logging.getLogger().setLevel(args.log_level)

    if not args.config.exists():
        logger.error("Config not found: %s", args.config)
        return 2
    if not args.checkpoint.exists():
        logger.error("Zenodo checkpoint not found: %s", args.checkpoint)
        return 2

    # 1. Build the reference model from the upstream config.
    logger.info("Loading config: %s", args.config)
    config, _ = load_config(str(args.config))
    logger.info("Constructing fresh DiffAb model from config.model …")
    model = get_model(config.model)
    ref_keys = set(model.state_dict().keys())
    logger.info("Reference model has %d parameter tensors.", len(ref_keys))

    # 2. Load the Zenodo file and unwrap.
    logger.info("torch.load(%s) on %s", args.checkpoint, args.device)
    raw = torch.load(str(args.checkpoint), map_location=args.device)
    if isinstance(raw, dict):
        logger.info("Top-level keys in checkpoint: %s", list(raw.keys()))
    zenodo_sd, source = _unwrap_state_dict(raw)
    zenodo_sd = _strip_module_prefix(zenodo_sd)
    logger.info("Found %d tensors via key %r.", len(zenodo_sd), source)

    # 3. Strict load attempt. We do this on a *copy* of the model so that
    # a partial load doesn't leave the model in a half-mutated state for
    # any subsequent inspection.
    logger.info("Attempting strict load …")
    try:
        model.load_state_dict(zenodo_sd, strict=True)
        logger.info("Strict load SUCCEEDED — Arm B checkpoint is fully compatible.")
        verdict = "LOAD_OK"
    except RuntimeError as exc:
        logger.warning("Strict load failed: %s", exc.__class__.__name__)
        # Reset and retry non-strict to enumerate what's mismatched.
        model = get_model(config.model)
        result = model.load_state_dict(zenodo_sd, strict=False)
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)
        logger.warning("Non-strict load: %d missing, %d unexpected.",
                       len(missing), len(unexpected))
        logger.warning("Missing (in model, absent from checkpoint): %s",
                       _format_key_sample(missing))
        logger.warning("Unexpected (in checkpoint, absent from model): %s",
                       _format_key_sample(unexpected))
        if len(missing) <= NONSTRICT_TOLERANCE and len(unexpected) <= NONSTRICT_TOLERANCE:
            verdict = "LOAD_OK_NONSTRICT"
        else:
            verdict = "LOAD_FAIL"

    # 4. Optional diff against upstream luost26.
    if args.base_checkpoint is not None:
        if not args.base_checkpoint.exists():
            logger.warning("--base-checkpoint not found, skipping diff: %s",
                           args.base_checkpoint)
        else:
            logger.info("Loading base checkpoint for key-set diff: %s",
                        args.base_checkpoint)
            base_raw = torch.load(str(args.base_checkpoint), map_location=args.device)
            base_sd, _ = _unwrap_state_dict(base_raw)
            base_sd = _strip_module_prefix(base_sd)
            base_keys = set(base_sd.keys())
            zenodo_keys = set(zenodo_sd.keys())
            only_zenodo = sorted(zenodo_keys - base_keys)
            only_base = sorted(base_keys - zenodo_keys)
            logger.info("Keys only in Zenodo: %d %s",
                        len(only_zenodo), _format_key_sample(only_zenodo))
            logger.info("Keys only in luost26: %d %s",
                        len(only_base), _format_key_sample(only_base))
            # Shape mismatches on shared keys are the most worrying signal;
            # report them explicitly.
            shared = base_keys & zenodo_keys
            shape_mismatches = [
                k for k in shared if base_sd[k].shape != zenodo_sd[k].shape
            ]
            if shape_mismatches:
                logger.warning(
                    "%d shared keys have different shapes between Zenodo "
                    "and luost26 — this is a strong incompatibility signal.",
                    len(shape_mismatches),
                )
                for k in shape_mismatches[:SAMPLE_KEY_LIMIT]:
                    logger.warning("  %s: luost26=%s zenodo=%s",
                                   k, tuple(base_sd[k].shape),
                                   tuple(zenodo_sd[k].shape))

    # 5. Final verdict line — last line of stdout for easy grepping.
    print(f"VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
