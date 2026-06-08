#!/usr/bin/env python3
"""Diffusion-DPO trainer for DiffAb on the Pareto-pair training data.

What this does
--------------
Loads two copies of a DiffAb fine-tune (π_θ trainable, π_ref frozen) and
fine-tunes π_θ on the AbDPO direct-energy preference loss (Zhou et al.
2024, Eq. 8) over a parquet of (winner = GT, loser = AAPR sample) pairs
emitted by ``scripts/dpo/select_pareto_pairs.py``.

Scaffolding is forked from ``scripts/diffab_ft/train.py`` so the EMA
shadow weights, early-stopping on the best EMA val signal, top-K
checkpointing, optional warmup, and W&B logging behave identically —
only the loss and the data loader change.

Why two models
--------------
DPO needs the *ratio* of π_θ(y_w|x)/π_ref(y_w|x) vs π_θ(y_l|x)/π_ref(y_l|x).
For diffusion models that becomes a difference of ELBO terms — see
``src.dpo.loss.abdpo_loss``. π_ref must be frozen end-to-end (no EMA on
it during DPO); AbDPO §3.3 explicitly assumes a stable reference.

Checkpoint convention
---------------------
Same as the fine-tune trainer:

  * ``runs/dpo/<run>/checkpoints/best_ema.pt``  — EMA(π_θ) at best val
                                                  DPO loss so far.
  * ``runs/dpo/<run>/checkpoints/last_ema.pt``  — EMA(π_θ) at last val.
  * ``runs/dpo/<run>/checkpoints/iter_<N>.pt``  — top-K by val DPO loss.
  * ``runs/dpo/<run>/checkpoints/state_<N>.pt`` — full resume state.

The EMA checkpoints are the deliverables that downstream evaluation
(NbBench, RAbD, design-mode AAR/RMSD) consumes.

Usage
-----
::

    # Smoke test (no W&B, single repeated batch, ≤100 iters):
    python scripts/dpo/train_dpo.py configs/dpo/vhh_dpo.yml \
        --output-dir runs/dpo --debug

    # Canary run on Snellius:
    python scripts/dpo/train_dpo.py configs/dpo/vhh_dpo.yml \
        --output-dir runs/dpo \
        --run-name   dpo_seed42_jfix_canary \
        --num-workers 4 \
        --wandb-project vhh-dpo \
        --wandb-run-name dpo_seed42_jfix_canary \
        --wandb-group   vhh_dpo

    # Resume after pre-emption:
    python scripts/dpo/train_dpo.py configs/dpo/vhh_dpo.yml \
        --output-dir runs/dpo \
        --run-name   dpo_seed42_jfix_canary \
        --resume runs/dpo/dpo_seed42_jfix_canary/checkpoints/state_2000.pt
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

# Fixed seed for the val DataLoader workers — matches the fine-tune
# trainer's convention so per-validation-pass masking variance doesn't
# bleed into the early-stop signal on a small val pool.
VAL_RNG_SEED = 12345


def _val_worker_init_fn(worker_id: int) -> None:
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
from diffab.utils.misc import (  # noqa: E402
    BlackHole, current_milli_time, inf_iterator, load_config, seed_all,
)
from diffab.utils.train import (  # noqa: E402
    count_parameters, get_optimizer, get_scheduler, get_warmup_sched,
    recursive_to,
)
from diffab.utils.transforms import get_transform  # noqa: E402

# ── Our dataset adapter (registers ``vhh_andd``) + DPO modules ─────────
import src.diffab_ft.datasets  # noqa: E402, F401  — registry side effect
from src.dpo.dataset import PairCollate, PairDataset  # noqa: E402
from src.dpo.loss import (  # noqa: E402
    abdpo_loss,
    check_pair_alignment,
    forward_pair_with_shared_noise,
)
from src.dpo.loss_ipo import ipo_loss  # noqa: E402

# Optional: TF32 for A100s.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ── Helpers (forked from scripts/diffab_ft/train.py) ────────────────────
def _make_logger(name: str, log_path: Path | None) -> logging.Logger:
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
    torch.save({
        "model": ema_model.module.state_dict(),
        "iteration": iteration,
        "val_loss": float(val_loss),
        "config": config,
        "ema": True,
    }, path)


def _save_full_state(
    path: Path, model, ema_model: AveragedModel, optimizer, scheduler,
    iteration: int, val_loss: float, config: dict, best_val: float,
    history: list,
) -> None:
    state = {
        "model": model.state_dict(),
        "ema_model": ema_model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "iteration": iteration,
        "val_loss": float(val_loss),
        "best_val": float(best_val),
        "history": history,
        "config": config,
        "ema": False,
    }
    torch.save(state, path)


class TopKCheckpointer:
    """Maintain the K best EMA checkpoints by ascending val DPO loss."""

    def __init__(self, ckpt_dir: Path, k: int = 3):
        self.ckpt_dir = ckpt_dir
        self.k = k
        self.entries: list[tuple[float, Path]] = []

    def maybe_save(
        self, iteration: int, val_loss: float, ema_model: AveragedModel,
        config: dict,
    ) -> Path | None:
        path = self.ckpt_dir / f"iter_{iteration}.pt"
        if len(self.entries) < self.k or val_loss < self.entries[-1][0]:
            _save_ema_checkpoint(path, ema_model, iteration, val_loss, config)
            self.entries.append((val_loss, path))
            self.entries.sort(key=lambda e: e[0])
            while len(self.entries) > self.k:
                _, evict_path = self.entries.pop()
                if evict_path.exists():
                    evict_path.unlink()
            return path
        return None


# ── π_ref loader ────────────────────────────────────────────────────────
def _load_pi_ref_into(model: torch.nn.Module, ckpt_path: Path,
                      device: str, logger: logging.Logger) -> None:
    """Load π_ref weights into a target model in-place.

    Uses ``weights_only=False`` because the upstream luost26/DiffAb
    checkpoints (and our derived best_ema.pt) pickle their training
    config as EasyDict, which the post-torch-2.6 safe path rejects.
    Our fine-tune-emitted checkpoints come from a trusted source.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"π_ref checkpoint not found: {ckpt_path}")
    ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    result = model.load_state_dict(sd, strict=False)
    if result.missing_keys:
        logger.warning(
            "π_ref load: %d missing keys (sample: %s)",
            len(result.missing_keys), result.missing_keys[:5],
        )
    if result.unexpected_keys:
        logger.warning(
            "π_ref load: %d unexpected keys (sample: %s)",
            len(result.unexpected_keys), result.unexpected_keys[:5],
        )
    if not result.missing_keys and not result.unexpected_keys:
        logger.info("π_ref load: all keys matched.")


# ── Main ─────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", type=Path,
                        help="Path to DPO YAML (e.g. vhh_dpo.yml).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Parent dir for run outputs.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Subdirectory name. Defaults to the YAML stem.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers. PairDataset's per-item "
                             "loser parsing is CPU-bound (~50ms) so a few "
                             "workers help, but the DPO step itself is "
                             "GPU-bound (~2s/step). 4-8 is the sweet spot.")
    parser.add_argument("--debug", action="store_true",
                        help="Skip W&B, repeat one batch, cap iters at 100.")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Path to a state_<N>.pt to resume.")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-mode", default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--pi-ref-override", type=Path, default=None,
                        help="Override the π_ref path in the config (useful "
                             "for ablating different references against the "
                             "same pair pool).")
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
    logger = _make_logger("train_dpo", log_path)
    logger.info("Args: %s", vars(args))
    logger.info("Run dir: %s", run_dir)

    # Cross-check that the YAML's dpo block is sane.
    if "dpo" not in config:
        logger.error("Config missing 'dpo' block — this is a DPO trainer.")
        return 2
    if "pair_parquet" not in config.dpo:
        logger.error("Config missing dpo.pair_parquet.")
        return 2
    pi_ref_path = (
        args.pi_ref_override
        if args.pi_ref_override is not None
        else Path(config.dpo.pi_ref_checkpoint)
    )
    if not pi_ref_path.is_absolute():
        pi_ref_path = PROJECT_ROOT / pi_ref_path

    # T must agree between the model config and the dpo block — the
    # loss multiplies by T, so a silent mismatch would scale the
    # gradient by a constant factor and confuse hyperparameter sweeps.
    if (
        int(config.dpo.get("num_timesteps", config.model.diffusion.num_steps))
        != int(config.model.diffusion.num_steps)
    ):
        logger.error(
            "Inconsistent T: dpo.num_timesteps=%d vs model.diffusion.num_steps=%d",
            int(config.dpo.num_timesteps),
            int(config.model.diffusion.num_steps),
        )
        return 2
    T = int(config.model.diffusion.num_steps)

    if not args.debug:
        shutil.copyfile(args.config, run_dir / args.config.name)

    # ── Base dataset (provides LMDB + manifest for the winner side) ─
    logger.info("Constructing base VHHANDDDataset (winner-side source)...")
    base_dataset = get_dataset(config.dataset.train)
    transform = get_transform(config.dataset.train.transform)

    # ── Pair datasets (train / val) ─────────────────────────────────
    logger.info("Building DPO pair datasets...")
    _pair_kwargs = dict(
        pairs_parquet=config.dpo.pair_parquet,
        base_dataset=base_dataset,
        transform=transform,
        val_split_seed=int(config.dpo.get("val_split_seed", 42)),
        val_gt_holdout=int(config.dpo.get("val_gt_holdout", 3)),
        val_gt_ids=config.dpo.get("val_gt_ids", None),
        heavy_max_resseq=int(
            config.dataset.train.get("heavy_max_resseq", 150)
        ),
        pair_seed_offset=int(config.dpo.get("pair_seed_offset", 0)),
        drop_misaligned=bool(config.dpo.get("drop_misaligned", True)),
    )
    train_pairs = PairDataset(split="train", **_pair_kwargs)
    val_pairs = PairDataset(split="val", **_pair_kwargs)
    logger.info(
        "Train pairs: %d  |  Val pairs: %d", len(train_pairs), len(val_pairs),
    )

    collate = PairCollate(eight=False)
    train_loader = DataLoader(
        train_pairs,
        batch_size=config.train.batch_size,
        collate_fn=collate,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_pairs,
        batch_size=config.train.batch_size,
        collate_fn=collate,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=_val_worker_init_fn,
    )
    train_iterator = inf_iterator(train_loader)
    if args.debug:
        debug_batch = next(train_iterator)
        train_iterator = iter(lambda: deepcopy(debug_batch), None)

    # ── Models (π_θ trainable + π_ref frozen) ───────────────────────
    logger.info("Building π_θ ...")
    model_theta = get_model(config.model).to(args.device)
    logger.info("π_θ parameter count: %d", count_parameters(model_theta))

    logger.info("Building π_ref (frozen) ...")
    model_ref = get_model(config.model).to(args.device)

    logger.info("Loading π_ref weights from %s", pi_ref_path)
    _load_pi_ref_into(model_ref, pi_ref_path, args.device, logger)
    # Initialise π_θ from the SAME checkpoint — see locked decision #2
    # in docs/dpo_training_context.md. Iter-0 δ ≈ 0 → loss ≈ -log σ(0)
    # = log 2 ≈ 0.693, which is the sanity check for "DPO is wired up".
    logger.info("Initialising π_θ from π_ref ...")
    _load_pi_ref_into(model_theta, pi_ref_path, args.device, logger)

    # Freeze π_ref end-to-end: eval mode (deactivates dropout) +
    # requires_grad_(False) (no autograd allocation) + caller uses
    # torch.no_grad in the forward (no graph at all).
    model_ref.eval()
    for p in model_ref.parameters():
        p.requires_grad_(False)

    ema_decay = float(config.train.get("ema_decay", 0.999))
    ema_theta = AveragedModel(
        model_theta, multi_avg_fn=get_ema_multi_avg_fn(ema_decay),
    )

    # ── Optimizer / scheduler ──────────────────────────────────────
    optimizer = get_optimizer(config.train.optimizer, model_theta)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
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
        logger.info("Resuming π_θ + EMA + optimizer from %s", args.resume)
        ck = torch.load(str(args.resume), map_location=args.device,
                        weights_only=False)
        model_theta.load_state_dict(ck["model"])
        ema_theta.module.load_state_dict(ck["ema_model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        if has_warmup and "warmup_sched" in ck:
            warmup_sched.load_state_dict(ck["warmup_sched"])
        it_first = ck["iteration"] + 1
        best_val = ck.get("best_val", float("inf"))
        history = ck.get("history", [])

    # ── W&B init ────────────────────────────────────────────────────
    use_wandb = (not args.debug) and (args.wandb_project is not None) \
        and args.wandb_mode != "disabled"
    wandb = None
    if use_wandb:
        try:
            import wandb as _wandb  # noqa: WPS433
            wandb = _wandb
        except ImportError:
            logger.warning("wandb not installed; running without it.")
            use_wandb = False
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
    writer = BlackHole()  # noqa: F841

    # ── Training loop ───────────────────────────────────────────────
    max_iters = int(config.train.max_iters)
    if args.debug:
        max_iters = min(max_iters, 100)
        config.train.val_freq = max(1, min(int(config.train.val_freq), 25))

    val_freq = int(config.train.val_freq)
    patience = int(config.train.get("early_stop_patience", 10))
    grad_clip = float(config.train.max_grad_norm)
    loss_weights = dict(config.train.loss_weights)
    beta_dpo = float(config.dpo.beta_dpo)
    aggregation = str(config.dpo.get("aggregation", "residue"))
    # Brief 18: objective dispatch — "dpo" (default; abdpo_loss) or
    # "ipo" (loss_ipo.ipo_loss). Both take the same parameter signature
    # so the dispatch is one line at each loss-call site (one in the
    # main forward path and one in the per-channel grad audit).
    objective = str(config.dpo.get("objective", "dpo")).lower()
    if objective not in ("dpo", "ipo"):
        raise ValueError(
            f"config.dpo.objective must be 'dpo' or 'ipo', got {objective!r}"
        )
    _loss_fn = ipo_loss if objective == "ipo" else abdpo_loss
    # eval_mode_theta: put π_θ into .eval() during DPO training. The
    # default is True because keeping π_θ in .train() (dropout active)
    # while π_ref is in .eval() (dropout off) creates an asymmetric
    # baseline at iter 1 — observed: L_l_θ - L_l_ref ≈ +2.3 even with
    # identical weights, because the AAPR inputs are OOD for the model
    # and dropout amplifies the response. This produces a fake
    # +margin that the DPO loss chases instead of learning real
    # preferences. Setting π_θ to .eval() eliminates the asymmetry at
    # root: at iter 1 we get clean L_w_θ == L_w_ref and L_l_θ ==
    # L_l_ref (up to floating-point noise). We lose dropout's
    # regularization, but for DPO (a small perturbation off a
    # converged model on ~1300 pairs) this is the right trade.
    eval_mode_theta = bool(config.dpo.get("eval_mode_theta", True))
    topk = TopKCheckpointer(ckpt_dir, k=int(config.train.get("save_top_k", 3)))

    # Per-channel gradient-norm imbalance threshold. Brief 17 §11
    # (orchestrator note 2): if any structural channel's grad norm
    # exceeds STRUCT_VS_SEQ_GRAD_WARN × the seq channel's norm at an
    # audit iter, that's a destabilization signal — log a WARN line so
    # the operator can decide whether to kill the run. We don't auto-
    # terminate; the existing NaN-loss guard at line ~528 covers true
    # failure. Threshold 10× per the orchestrator spec.
    STRUCT_VS_SEQ_GRAD_WARN = 10.0
    PER_CHANNEL_AUDIT_KEYS = ("rot", "pos", "seq")

    def _audit_per_channel_grad_norms(
        pair_losses_,
        mask_,
    ) -> dict[str, float]:
        """Three separate backward passes (one per channel) to measure
        the gradient each channel would contribute at unit weight.

        Restores ``model_theta``'s grads to zero before returning, so
        the caller can do a fresh ``loss.backward()`` on the main
        composite without contamination. Uses ``retain_graph=True``
        on each backward so the next channel and the main backward
        can still traverse the same forward graph.

        Returns ``{"rot": …, "pos": …, "seq": …}`` (Python floats).
        """
        norms: dict[str, float] = {}
        for ch in PER_CHANNEL_AUDIT_KEYS:
            model_theta.zero_grad(set_to_none=True)
            ch_weights = {k: 0.0 for k in PER_CHANNEL_AUDIT_KEYS}
            ch_weights[ch] = 1.0
            ch_loss, _ = _loss_fn(
                pair_losses_,
                mask=mask_,
                loss_weights=ch_weights,
                beta_dpo=beta_dpo,
                num_timesteps=T,
                aggregation=aggregation,
            )
            ch_loss.backward(retain_graph=True)
            sq = 0.0
            for p in model_theta.parameters():
                if p.grad is not None:
                    sq += float((p.grad.detach() ** 2).sum().item())
            norms[ch] = float(sq ** 0.5)
        model_theta.zero_grad(set_to_none=True)
        return norms

    def _run_dpo_step(
        batch_dict: dict,
        *,
        training: bool,
        audit_per_channel_grads: bool = False,
    ) -> dict:
        """Single DPO loss evaluation. Returns logging-friendly dict.

        When ``audit_per_channel_grads`` is True (training only), runs
        three extra single-channel backward passes BEFORE the caller's
        main backward, captures their grad norms, and emits them under
        ``grad_norm_{rot,pos,seq}`` keys. Grads are zeroed before the
        function returns so the caller's main ``loss.backward()`` runs
        on a clean grad state. See ``_audit_per_channel_grad_norms``.
        """
        batch_w = recursive_to(batch_dict["winner"], args.device)
        batch_l = recursive_to(batch_dict["loser"], args.device)

        # Pair-alignment cross-check. Drop the batch on mismatch
        # rather than letting δ become meaningless. With the
        # PairDataset's per-pair RNG pinning this should never fire,
        # but we want a loud signal if it does.
        if not check_pair_alignment(batch_w, batch_l):
            raise RuntimeError(
                "Batch-level generate_flag mismatch slipped past PairDataset. "
                "Check the masking transforms and per-pair RNG pinning."
            )

        # Shared timestep per batch (sampled per-pair; AbDPO §3.1).
        B = batch_w["aa"].size(0)
        t = torch.randint(
            1, T + 1, (B,), dtype=torch.long, device=args.device,
        )

        if training:
            pair_losses = forward_pair_with_shared_noise(
                model_theta, model_ref, batch_w, batch_l, t,
                device=torch.device(args.device),
            )
        else:
            # EMA val: swap π_θ for the EMA shadow. AveragedModel does
            # *not* auto-forward attribute access, so we unwrap to
            # ``.module`` — our forward helper needs ``.cfg``,
            # ``.encode``, and ``.diffusion`` directly on the model.
            with torch.no_grad():
                pair_losses = forward_pair_with_shared_noise(
                    ema_theta.module, model_ref, batch_w, batch_l, t,
                    device=torch.device(args.device),
                )

        loss, diag = _loss_fn(
            pair_losses,
            mask=batch_w["generate_flag"],
            loss_weights=loss_weights,
            beta_dpo=beta_dpo,
            num_timesteps=T,
            aggregation=aggregation,
        )

        diag_out = {k: float(v.item()) for k, v in diag.items()}
        diag_out["loss"] = float(loss.item())

        if audit_per_channel_grads and training:
            # Audit BEFORE the caller's main backward so the autograd
            # graph rooted at pair_losses is still alive (we pass
            # retain_graph=True internally to keep it that way).
            per_channel = _audit_per_channel_grad_norms(
                pair_losses, batch_w["generate_flag"],
            )
            for ch, n in per_channel.items():
                diag_out[f"grad_norm_{ch}"] = n
            seq_n = per_channel["seq"]
            if seq_n > 1e-9:
                ratios = {
                    f"grad_ratio_{ch}_over_seq": per_channel[ch] / seq_n
                    for ch in PER_CHANNEL_AUDIT_KEYS
                }
                for k, v in ratios.items():
                    diag_out[k] = float(v)
                worst_struct = max(
                    ratios["grad_ratio_rot_over_seq"],
                    ratios["grad_ratio_pos_over_seq"],
                )
                if worst_struct > STRUCT_VS_SEQ_GRAD_WARN:
                    logger.warning(
                        "GRAD-IMBALANCE: structural channel exceeds %.1f× seq "
                        "(rot=%.3f pos=%.3f seq=%.3f → max ratio %.2f). Brief 17 "
                        "§11 destabilization warning — consider killing run if "
                        "this persists across multiple audits.",
                        STRUCT_VS_SEQ_GRAD_WARN,
                        per_channel["rot"], per_channel["pos"], seq_n,
                        worst_struct,
                    )

        return loss, diag_out

    def _step(it: int) -> dict:
        t0 = current_milli_time()
        # eval_mode_theta=True keeps π_θ in eval() during the train
        # step so dropout doesn't introduce stochasticity that π_ref
        # (frozen + eval) doesn't see — see comment at the flag def.
        # Parameter updates still happen normally; only the forward-pass
        # noise from dropout is suppressed. EMA tracks parameter values,
        # so this doesn't affect ema_theta's updates.
        if eval_mode_theta:
            model_theta.eval()
        else:
            model_theta.train()
        batch_dict = next(train_iterator)

        # Brief 17 §11: audit per-channel grad norms on the same
        # cadence as validation (every val_freq iters). Catches the
        # rot/pos destabilization risk early without paying the audit
        # cost on every iter.
        audit_grads = (it % val_freq == 0) or (it == it_first)
        loss, diag = _run_dpo_step(
            batch_dict, training=True, audit_per_channel_grads=audit_grads,
        )

        if not torch.isfinite(loss):
            logger.error("Non-finite DPO loss at iter %d: %s", it, loss.item())
            torch.save({
                "model": model_theta.state_dict(),
                "iteration": it,
                "config": dict(config),
            }, run_dir / f"checkpoint_nan_{it}.pt")
            raise RuntimeError(f"Non-finite DPO loss at iter {it}")

        loss.backward()
        grad_norm = clip_grad_norm_(model_theta.parameters(), grad_clip)
        optimizer.step()
        warmup_sched.step()
        optimizer.zero_grad()
        ema_theta.update_parameters(model_theta)

        out = {
            "iter": it,
            "loss": diag["loss"],
            "margin": diag["margin_mean"],
            "accuracy": diag["accuracy"],
            "L_w_theta": diag["L_w_theta"],
            "L_l_theta": diag["L_l_theta"],
            "L_w_ref": diag["L_w_ref"],
            "L_l_ref": diag["L_l_ref"],
            "delta_mean": diag["delta_mean"],
            "grad_norm": float(grad_norm),
            "lr": optimizer.param_groups[0]["lr"],
            "ms": current_milli_time() - t0,
        }
        # Forward the per-channel audit keys when this iter ran the
        # audit (set only on the val_freq cadence — see _step's
        # audit_grads flag). Brief 17 §11 logs them to W&B for the
        # operator to monitor rot/pos vs seq balance.
        # Brief 18 §7: also forward IPO-specific diagnostics
        # (tau_target, margin_distance_from_tau_mean, converged_pair_fraction)
        # so the operator can watch IPO convergence on W&B.
        IPO_EXTRA_KEYS = (
            "tau_target",
            "margin_distance_from_tau_mean",
            "converged_pair_fraction",
            "margin_per_residue_mean",
            "mask_count_mean",
        )
        for k, v in diag.items():
            if (
                k.startswith("grad_norm_")
                or k.startswith("grad_ratio_")
                or k in IPO_EXTRA_KEYS
            ):
                out[k] = float(v)
        return out

    @torch.no_grad()
    def _validate(it: int) -> dict:
        """Run val pass under EMA(π_θ). Average diagnostics across val pairs.

        Note: the DPO loss is per-batch noisy on small val sets; the
        early-stop decision is on val/loss (= mean over val batches of
        the DPO loss) which still has the right monotone trend signal.
        """
        ema_theta.eval()
        accum: dict[str, float] = {}
        n_batches = 0
        for batch_dict in val_loader:
            _, diag = _run_dpo_step(batch_dict, training=False)
            for k, v in diag.items():
                accum[k] = accum.get(k, 0.0) + v
            n_batches += 1
        if n_batches == 0:
            logger.warning(
                "Validation produced 0 batches — val pool too small for the "
                "configured batch_size? Skipping val pass."
            )
            return {}
        averaged = {k: v / n_batches for k, v in accum.items()}
        avg_loss = averaged["loss"]
        if config.train.scheduler.type == "plateau":
            scheduler.step(avg_loss)
        else:
            scheduler.step()
        return averaged

    if has_warmup:
        logger.info(
            "Linear LR warmup enabled: 0 → optimizer.lr over %d iterations.",
            int(warmup_cfg.max_iters),
        )
    logger.info(
        "Starting DPO training: objective=%s  |  iters %d → %d  |  "
        "val every %d  |  β=%.3f  |  T=%d  |  aggregation=%s  |  "
        "patience=%d  |  grad_clip=%.2f  |  π_θ mode=%s",
        objective.upper(), it_first, max_iters, val_freq, beta_dpo, T,
        aggregation, patience, grad_clip,
        "eval" if eval_mode_theta else "train",
    )
    if objective == "ipo":
        logger.info(
            "IPO regression target τ = 1/(2β) = %.3f (m near 0 → loss "
            "near τ² = %.3f; m near τ → loss near 0).",
            1.0 / (2.0 * beta_dpo), (1.0 / (2.0 * beta_dpo)) ** 2,
        )
    t_train_start = time.time()

    try:
        for it in range(it_first, max_iters + 1):
            stats = _step(it)
            if it % 25 == 0 or it == it_first:
                logger.info(
                    "iter %5d | dpo %.4f | margin %+ .3f | acc %.2f | "
                    "L_w_θ %.3f L_l_θ %.3f L_w_r %.3f L_l_r %.3f | "
                    "grad %.2f | lr %.2e | %dms",
                    it, stats["loss"], stats["margin"], stats["accuracy"],
                    stats["L_w_theta"], stats["L_l_theta"],
                    stats["L_w_ref"], stats["L_l_ref"],
                    stats["grad_norm"], stats["lr"], stats["ms"],
                )
            if use_wandb:
                wandb.log(
                    {f"train/{k}": v for k, v in stats.items() if k != "iter"},
                    step=it,
                )

            if it % val_freq == 0 or it == max_iters:
                val_metrics = _validate(it)
                if not val_metrics:
                    continue
                val_loss = val_metrics["loss"]
                history.append((it, val_loss))
                logger.info(
                    "iter %d | val dpo %.4f (best %.4f) | val margin %+ .3f | "
                    "val acc %.2f",
                    it, val_loss, best_val,
                    val_metrics.get("margin_mean", float("nan")),
                    val_metrics.get("accuracy", float("nan")),
                )
                if use_wandb:
                    payload = {f"val/{k}": v for k, v in val_metrics.items()}
                    payload["val/best"] = best_val
                    wandb.log(payload, step=it)

                topk.maybe_save(it, val_loss, ema_theta, dict(config))

                _save_ema_checkpoint(
                    ckpt_dir / "last_ema.pt", ema_theta, it, val_loss,
                    dict(config),
                )
                if val_loss < best_val:
                    best_val = val_loss
                    no_improve_count = 0
                    _save_ema_checkpoint(
                        ckpt_dir / "best_ema.pt", ema_theta, it, val_loss,
                        dict(config),
                    )
                    logger.info("New best val DPO loss: %.4f", best_val)
                else:
                    no_improve_count += 1
                    logger.info(
                        "No val improvement for %d/%d validations.",
                        no_improve_count, patience,
                    )

                _save_full_state(
                    ckpt_dir / f"state_{it}.pt",
                    model_theta, ema_theta, optimizer, scheduler,
                    it, val_loss, dict(config), best_val, history,
                )
                for old in ckpt_dir.glob("state_*.pt"):
                    if old.name != f"state_{it}.pt":
                        old.unlink(missing_ok=True)

                if no_improve_count >= patience:
                    best_iter = (
                        min(history, key=lambda h: h[1])[0]
                        if history else it
                    )
                    logger.info(
                        "Early stop: %d validations without improvement "
                        "(best val DPO loss = %.4f at iter %d).",
                        patience, best_val, best_iter,
                    )
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user; checkpointing then exiting.")
        _save_full_state(
            ckpt_dir / "state_interrupted.pt",
            model_theta, ema_theta, optimizer, scheduler,
            it, history[-1][1] if history else float("nan"),
            dict(config), best_val, history,
        )

    t_total = time.time() - t_train_start
    logger.info("=" * 60)
    logger.info("DPO training finished in %.1f min.", t_total / 60)
    logger.info("Best val DPO loss: %.4f", best_val)
    logger.info("Best checkpoint:   %s", ckpt_dir / "best_ema.pt")

    if use_wandb:
        wandb.summary["best_val"] = best_val
        wandb.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
