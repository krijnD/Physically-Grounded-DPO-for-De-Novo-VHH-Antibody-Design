#!/usr/bin/env python3
"""Custom thin trainer for DiffAb VHH fine-tuning.

Why not call DiffAb's ``train.py`` directly
-------------------------------------------
DiffAb's reference trainer (``third_party/diffab/train.py``) is fine for
the original SAbDab pre-training but lacks three things we need for a
thesis-grade fine-tune on 191 CDR clusters:

  1. **EMA** — Polyak-averaged shadow weights. With a small dataset the
     raw weights oscillate; the EMA copy is what we evaluate and what
     becomes ``π_ref`` for the downstream DPO phase. Implemented via
     ``torch.optim.swa_utils.AveragedModel`` + ``get_ema_multi_avg_fn``
     (PyTorch ≥ 2.0, no extra deps).
  2. **Early stopping on EMA val ELBO** — 191 clusters is small enough
     that we will overfit if we run the full ``max_iters``. We track the
     best EMA val loss and stop after ``patience`` validations without
     improvement.
  3. **W&B logging** — group-tagged so seed-stability comparisons in
     the thesis are one click in the dashboard. TensorBoard is kept as a
     fallback (compute nodes occasionally lose outbound).

Everything else is inherited from DiffAb: model construction
(``get_model``), dataset registry (``get_dataset`` — populated by
importing ``src.diffab_ft.datasets`` to register our ``vhh_andd``
adapter), transforms, optimizers/schedulers, and the
``ValidationLossTape`` averager.

Checkpoint convention
---------------------
Outputs live under ``<output_dir>/<run_name>/``:

  * ``config.yml``                — frozen copy of the input YAML.
  * ``log.txt``                   — file logger (mirrors stdout).
  * ``checkpoints/best_ema.pt``   — EMA weights at best val ELBO so far.
  * ``checkpoints/last_ema.pt``   — EMA weights at last validation.
  * ``checkpoints/iter_<N>.pt``   — top-3 by val ELBO (EMA + raw).
  * ``checkpoints/state_<N>.pt``  — optimizer/scheduler state at top-3
                                    iters (kept separate so EMA
                                    checkpoints stay light enough to
                                    download to a laptop).
  * ``wandb/``                    — local W&B cache.

Each ``.pt`` file is a dict with at minimum ``{'model', 'iteration',
'val_loss', 'config'}`` so downstream eval/DPO tooling can be agnostic
to whether it's the raw model or the EMA shadow.

Usage
-----
::

    # Smoke test (no W&B, ≤100 iters):
    python scripts/diffab_ft/train.py configs/diffab_ft/vhh_ft.yml \\
        --output-dir runs/smoke --debug

    # Full run:
    python scripts/diffab_ft/train.py configs/diffab_ft/vhh_ft.yml \\
        --output-dir runs/vhh_ft \\
        --run-name seed42 \\
        --wandb-project vhh-diffab-ft \\
        --wandb-run-name seed42 \\
        --wandb-group vhh_ft

    # Resume after pre-emption:
    python scripts/diffab_ft/train.py configs/diffab_ft/vhh_ft.yml \\
        --output-dir runs/vhh_ft \\
        --run-name seed42 \\
        --resume runs/vhh_ft/seed42/checkpoints/state_5000.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

# Fixed seed for the val DataLoader workers. Holding this constant across
# every validation pass makes the random CDR3 boundary shrink/extend in
# `random_shrink_extend` (third_party/diffab/.../mask.py) reproducible, so
# the val/loss curve, top-K, and early-stop decisions are not at the
# mercy of per-pass masking variance — important on a 22-example val set.
VAL_RNG_SEED = 12345


def _val_worker_init_fn(worker_id: int) -> None:
    """Seed worker RNGs deterministically for the val DataLoader.

    With ``persistent_workers=False`` (DataLoader default), workers are
    respawned every time we iterate ``val_loader``; this hook fires on
    each spawn, so every validation pass sees the same masking on the
    same examples.
    """
    seed = VAL_RNG_SEED + worker_id
    random.seed(seed)
    torch.manual_seed(seed)

# ── Project paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "diffab"))

# ── DiffAb (transforms registered as a side effect of import) ───────────
from diffab.datasets import get_dataset  # noqa: E402
from diffab.models import get_model  # noqa: E402
from diffab.utils.data import PaddingCollate  # noqa: E402
from diffab.utils.misc import (  # noqa: E402
    BlackHole, current_milli_time, inf_iterator, load_config, seed_all,
)
from diffab.utils.train import (  # noqa: E402
    ValidationLossTape, count_parameters, get_optimizer, get_scheduler,
    get_warmup_sched, recursive_to, sum_weighted_losses,
)

# ── Our dataset adapter (registers ``vhh_andd``) ─────────────────────────
import src.diffab_ft.datasets  # noqa: E402, F401  — registry side effect

# Optional: TF32 for A100s.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ── Helpers ──────────────────────────────────────────────────────────────
def _make_logger(name: str, log_path: Path | None) -> logging.Logger:
    """File + stdout logger. Suppresses duplicate handlers on re-runs."""
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s — %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_path is not None:
        fh = logging.FileHandler(log_path, mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def _save_ema_checkpoint(
    path: Path, ema_model: AveragedModel, iteration: int,
    val_loss: float, config: dict,
) -> None:
    """EMA-only checkpoint (lightweight; for evaluation / DPO hand-off)."""
    torch.save({
        "model": ema_model.module.state_dict(),  # unwrap AveragedModel
        "iteration": iteration,
        "val_loss": float(val_loss),
        "config": config,
        "ema": True,
    }, path)


def _save_full_state(
    path: Path, model, ema_model: AveragedModel, optimizer, scheduler,
    warmup_sched, has_warmup: bool,
    iteration: int, val_loss: float, config: dict, best_val: float,
    history: list,
) -> None:
    """Heavy state for resumption (raw + EMA + optimizer + scheduler)."""
    state = {
        "model": model.state_dict(),
        "ema_model": ema_model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "iteration": iteration,
        "val_loss": float(val_loss),
        "best_val": float(best_val),
        "history": history,  # list of (iteration, val_loss) pairs
        "config": config,
        "ema": False,
    }
    if has_warmup:
        # LambdaLR.state_dict() round-trips cleanly. Only persist when
        # warmup is enabled so resuming from a pre-warmup checkpoint
        # without a warmup config doesn't try to restore stale state.
        state["warmup_sched"] = warmup_sched.state_dict()
    torch.save(state, path)


class TopKCheckpointer:
    """Maintain the K best checkpoints on disk by ascending val_loss.

    Stores only EMA shadow weights (which is what eval and DPO hand-off
    consume). The full optimizer/scheduler state is kept elsewhere via
    ``last_state.pt`` for resume — keeping these split means scientists
    can download the small EMA files without dragging optimizer state.
    """

    def __init__(self, ckpt_dir: Path, k: int = 3):
        self.ckpt_dir = ckpt_dir
        self.k = k
        self.entries: list[tuple[float, Path]] = []  # (loss, path), worst last

    def maybe_save(
        self, iteration: int, val_loss: float, ema_model: AveragedModel,
        config: dict,
    ) -> Path | None:
        path = self.ckpt_dir / f"iter_{iteration}.pt"
        if len(self.entries) < self.k or val_loss < self.entries[-1][0]:
            _save_ema_checkpoint(path, ema_model, iteration, val_loss, config)
            self.entries.append((val_loss, path))
            self.entries.sort(key=lambda e: e[0])
            # Evict beyond K.
            while len(self.entries) > self.k:
                _, evict_path = self.entries.pop()
                if evict_path.exists():
                    evict_path.unlink()
            return path
        return None


# ── Main ─────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", type=Path,
                        help="Path to fine-tune YAML (e.g. vhh_ft.yml).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Parent dir for run outputs.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Subdirectory name. Defaults to the YAML stem.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--debug", action="store_true",
                        help="Skip W&B, repeat one batch, cap iters at 100.")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Path to a state_<N>.pt to resume.")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-mode", default="online",
                        choices=["online", "offline", "disabled"],
                        help="Set 'offline' on Snellius compute if outbound is "
                             "blocked; sync later with `wandb sync`.")
    args = parser.parse_args()

    # ── Config & run dir ────────────────────────────────────────────
    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 2
    config, config_name = load_config(str(args.config))
    seed_all(config.train.seed)

    run_name = args.run_name or config_name
    run_dir = args.output_dir / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = None if args.debug else (run_dir / "log.txt")
    logger = _make_logger("train", log_path)
    logger.info("Args: %s", vars(args))
    logger.info("Run dir: %s", run_dir)

    # Freeze a copy of the YAML inside the run dir for provenance.
    if not args.debug:
        shutil.copyfile(args.config, run_dir / args.config.name)

    # ── Data ────────────────────────────────────────────────────────
    logger.info("Loading datasets ...")
    train_dataset = get_dataset(config.dataset.train)
    val_dataset = get_dataset(config.dataset.val)
    logger.info("Train: %d | Val: %d", len(train_dataset), len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        collate_fn=PaddingCollate(),
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        collate_fn=PaddingCollate(),
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=_val_worker_init_fn,
    )
    train_iterator = inf_iterator(train_loader)
    if args.debug:
        # Pin a single batch and replay it to validate shapes/loss flow.
        debug_batch = next(train_iterator)
        train_iterator = iter(lambda: deepcopy(debug_batch), None)

    # ── Model + EMA ────────────────────────────────────────────────
    logger.info("Building model ...")
    model = get_model(config.model).to(args.device)
    logger.info("Number of parameters: %d", count_parameters(model))

    # Optional: load Arm-specific starting weights (model-only, NOT
    # optimizer — the optimizer state comes from a different distribution
    # and would mislead the early steps of fine-tuning).
    init_ckpt = config.get("init_checkpoint")
    if init_ckpt and args.resume is None:
        init_path = Path(init_ckpt)
        if not init_path.is_absolute():
            init_path = PROJECT_ROOT / init_path
        if not init_path.exists():
            logger.error("init_checkpoint not found: %s", init_path)
            return 2
        logger.info("Loading init weights from %s", init_path)
        # weights_only=False: luost26/DiffAb's checkpoint pickles its
        # original training config as an easydict.EasyDict, which the
        # torch>=2.6 default (weights_only=True) refuses. The checkpoint
        # comes from a known HF source, so opt back into the legacy
        # full-pickle path. Resume checkpoints we write ourselves (handled
        # below) only contain tensors + plain dicts — those still load
        # under the default safe path.
        ck = torch.load(str(init_path), map_location=args.device, weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        result = model.load_state_dict(sd, strict=False)
        logger.info("Init load: %d missing, %d unexpected.",
                    len(result.missing_keys), len(result.unexpected_keys))
        if result.missing_keys:
            logger.warning("Missing (sample): %s", result.missing_keys[:5])
        if result.unexpected_keys:
            logger.warning("Unexpected (sample): %s", result.unexpected_keys[:5])

    ema_decay = float(config.train.get("ema_decay", 0.999))
    ema_model = AveragedModel(
        model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay),
    )

    # ── Optimizer / scheduler ──────────────────────────────────────
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    # Optional linear LR warmup. Composes multiplicatively with the
    # plateau scheduler: ``LambdaLR`` (warmup) scales the base LR and is
    # stepped every iteration; ``ReduceLROnPlateau`` mutates the base
    # LR and is stepped per validation. Standard PyTorch idiom — they
    # don't fight because they touch different things in
    # ``param_groups``. ``BlackHole`` is the no-op fallback when no
    # warmup config is given.
    warmup_cfg = config.train.get("warmup")
    warmup_sched = get_warmup_sched(warmup_cfg, optimizer)
    has_warmup = warmup_cfg is not None
    optimizer.zero_grad()

    it_first = 1
    best_val = float("inf")
    history: list[tuple[int, float]] = []
    no_improve_count = 0

    # ── Resume (full state) ─────────────────────────────────────────
    if args.resume is not None:
        if not args.resume.exists():
            logger.error("Resume file not found: %s", args.resume)
            return 2
        logger.info("Resuming from %s", args.resume)
        ck = torch.load(str(args.resume), map_location=args.device)
        model.load_state_dict(ck["model"])
        ema_model.module.load_state_dict(ck["ema_model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        if has_warmup and "warmup_sched" in ck:
            warmup_sched.load_state_dict(ck["warmup_sched"])
        elif has_warmup:
            logger.warning(
                "Resuming with warmup enabled but checkpoint has no "
                "warmup state; warmup will restart from iter 0."
            )
        it_first = ck["iteration"] + 1
        best_val = ck.get("best_val", float("inf"))
        history = ck.get("history", [])

    # ── W&B init ────────────────────────────────────────────────────
    use_wandb = (not args.debug) and (args.wandb_project is not None) \
        and args.wandb_mode != "disabled"
    if use_wandb:
        try:
            import wandb  # noqa: WPS433  — optional dep
        except ImportError:
            logger.warning("wandb not installed; running without it.")
            use_wandb = False
            wandb = None  # noqa: F841
    if use_wandb:
        os.environ.setdefault("WANDB_DIR", str(run_dir))
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or run_name,
            group=args.wandb_group,
            dir=str(run_dir),
            mode=args.wandb_mode,
            config=dict(config),
        )
    writer = BlackHole()  # We log directly via wandb; no TensorBoard.

    # ── Training loop ───────────────────────────────────────────────
    max_iters = config.train.max_iters
    if args.debug:
        max_iters = min(max_iters, 100)
        config.train.val_freq = max(1, min(config.train.val_freq, 25))

    val_freq = int(config.train.val_freq)
    patience = int(config.train.get("early_stop_patience", 5))
    grad_clip = float(config.train.max_grad_norm)
    loss_weights = config.train.loss_weights
    topk = TopKCheckpointer(ckpt_dir, k=int(config.train.get("save_top_k", 3)))

    def _step(it: int) -> dict:
        t0 = current_milli_time()
        model.train()
        batch = recursive_to(next(train_iterator), args.device)
        loss_dict = model(batch)
        loss = sum_weighted_losses(loss_dict, loss_weights)
        loss_dict["overall"] = loss

        loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        # Warmup is per-iteration; stepping after optimizer.step() is the
        # canonical PyTorch order for LambdaLR.
        warmup_sched.step()
        optimizer.zero_grad()

        # EMA update happens AFTER the step, on the freshly updated
        # parameters. AveragedModel.update_parameters is the canonical API.
        ema_model.update_parameters(model)

        if not torch.isfinite(loss):
            logger.error("Non-finite loss at iter %d: %s", it, loss.item())
            torch.save({
                "model": model.state_dict(),
                "iteration": it,
                "config": dict(config),
            }, run_dir / f"checkpoint_nan_{it}.pt")
            raise RuntimeError(f"Non-finite loss at iter {it}")

        return {
            "iter": it,
            "loss": float(loss.item()),
            "loss_rot": float(loss_dict.get("rot", torch.tensor(0.0))),
            "loss_pos": float(loss_dict.get("pos", torch.tensor(0.0))),
            "loss_seq": float(loss_dict.get("seq", torch.tensor(0.0))),
            "grad_norm": float(grad_norm),
            "lr": optimizer.param_groups[0]["lr"],
            "ms": current_milli_time() - t0,
        }

    @torch.no_grad()
    def _validate(it: int) -> dict:
        """Run validation and return per-component EMA val losses.

        Returns the full ``{rot, pos, seq, overall}`` dict so we can log
        each component to W&B — the seed42 diagnostic specifically
        depended on which loss component moved (rot improved most, seq
        modestly, pos basically not). With only ``val/loss`` (overall)
        logged this story is invisible in the dashboard.
        """
        ema_model.eval()
        tape = ValidationLossTape()
        for batch in val_loader:
            batch = recursive_to(batch, args.device)
            loss_dict = ema_model(batch)
            loss = sum_weighted_losses(loss_dict, loss_weights)
            loss_dict["overall"] = loss
            tape.update(loss_dict, 1)
        avg_overall = tape.log(it, logger, writer, "val")
        # tape.accumulate holds summed component tensors; divide by tape
        # total to get the same per-component averages tape.log uses
        # internally.
        val_components = {
            k: float(v.item() / tape.total)
            for k, v in tape.accumulate.items()
        }
        if config.train.scheduler.type == "plateau":
            scheduler.step(avg_overall)
        else:
            scheduler.step()
        return val_components

    if has_warmup:
        logger.info(
            "Linear LR warmup enabled: 0 → optimizer.lr over %d iterations.",
            int(warmup_cfg.max_iters),
        )
    logger.info("Starting training: iters %d → %d (val every %d, patience=%d).",
                it_first, max_iters, val_freq, patience)
    t_train_start = time.time()

    try:
        for it in range(it_first, max_iters + 1):
            stats = _step(it)
            if it % 25 == 0 or it == it_first:
                logger.info(
                    "iter %5d | loss %.4f (rot %.3f pos %.3f seq %.3f) "
                    "| grad %.2f | lr %.2e | %dms",
                    it, stats["loss"], stats["loss_rot"], stats["loss_pos"],
                    stats["loss_seq"], stats["grad_norm"], stats["lr"],
                    stats["ms"],
                )
            if use_wandb:
                wandb.log({f"train/{k}": v for k, v in stats.items()
                           if k != "iter"}, step=it)

            if it % val_freq == 0 or it == max_iters:
                val_components = _validate(it)
                val_loss = val_components["overall"]
                history.append((it, val_loss))
                logger.info("iter %d | EMA val loss %.4f (best %.4f)",
                            it, val_loss, best_val)
                if use_wandb:
                    # ``val/loss`` mirrors the seed42 metric name for
                    # cross-run comparison; per-component scalars give
                    # the diagnostic breakdown (rot/pos/seq).
                    payload = {
                        f"val/loss_{k}": v
                        for k, v in val_components.items()
                        if k != "overall"
                    }
                    payload["val/loss"] = val_loss
                    payload["val/best"] = best_val
                    wandb.log(payload, step=it)

                # Top-K save.
                topk.maybe_save(it, val_loss, ema_model, dict(config))

                # Best / last EMA.
                _save_ema_checkpoint(
                    ckpt_dir / "last_ema.pt", ema_model, it, val_loss,
                    dict(config),
                )
                if val_loss < best_val:
                    best_val = val_loss
                    no_improve_count = 0
                    _save_ema_checkpoint(
                        ckpt_dir / "best_ema.pt", ema_model, it, val_loss,
                        dict(config),
                    )
                    logger.info("New best EMA val loss: %.4f", best_val)
                else:
                    no_improve_count += 1
                    logger.info("No improvement for %d/%d validations.",
                                no_improve_count, patience)

                # Full-state save (resume target).
                _save_full_state(
                    ckpt_dir / f"state_{it}.pt",
                    model, ema_model, optimizer, scheduler,
                    warmup_sched, has_warmup,
                    it, val_loss, dict(config), best_val, history,
                )
                # Keep only the latest full-state to bound disk usage.
                for old in ckpt_dir.glob("state_*.pt"):
                    if old.name != f"state_{it}.pt":
                        old.unlink(missing_ok=True)

                if no_improve_count >= patience:
                    logger.info(
                        "Early stop: %d validations without improvement "
                        "(best EMA val loss = %.4f at history step %d).",
                        patience, best_val,
                        max(history, key=lambda h: -h[1])[0],
                    )
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user; checkpointing then exiting.")
        _save_full_state(
            ckpt_dir / "state_interrupted.pt",
            model, ema_model, optimizer, scheduler,
            warmup_sched, has_warmup,
            it, history[-1][1] if history else float("nan"),
            dict(config), best_val, history,
        )

    t_total = time.time() - t_train_start
    logger.info("=" * 60)
    logger.info("Training finished in %.1f min.", t_total / 60)
    logger.info("Best EMA val loss: %.4f", best_val)
    logger.info("Best checkpoint:   %s", ckpt_dir / "best_ema.pt")

    if use_wandb:
        wandb.summary["best_val"] = best_val
        wandb.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
