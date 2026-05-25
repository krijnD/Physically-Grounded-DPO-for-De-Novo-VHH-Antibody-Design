"""Pair-aware dataset/collate for Diffusion-DPO training on DiffAb.

The pair-selection script (``scripts/dpo/select_pareto_pairs.py``) emits
a parquet whose rows are ``(gt_complex_id, winner_pdb_path,
loser_pdb_path, …)``. This module turns those rows into DiffAb-format
batches that the AbDPO loss in :mod:`src.dpo.loss` can consume.

Key invariants this dataset enforces — without these, the per-residue
δ in AbDPO Eq. 8 is not well-defined:

1. **Aligned ``generate_flag``** across (winner, loser) within a pair.
   We apply *the same* ``MaskMultipleCDRs``/``random_shrink_extend``
   pipeline to both sides under a pinned RNG. Since AAPR keeps the
   framework + antigen verbatim and only re-samples CDR residues at the
   same positions, this produces byte-identical masks.

2. **Aligned tensor shapes** across (winner, loser). DiffAb's parser
   on the loser PDB uses the *winner's* ``H_chain`` + ``ag_chains`` IDs
   and the J-anchor-fixed ``heavy_max_resseq`` cap, so chain counts and
   residue counts match the winner's LMDB-cached structure.

3. **GT-stratified train/val split.** With ~25 GTs and ~192 pairs,
   random per-pair splitting would leak (multiple pairs share a GT). We
   hold out N GTs entirely and put all their pairs in the val set. The
   holdout is deterministic given ``val_split_seed``.

Anything that violates 1 or 2 is logged and the pair is *dropped* — see
the design doc's "Edge case" note in
``docs/dpo_training_context.md`` § "Multi-CDR scope handling".
"""

from __future__ import annotations

import copy
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

# DiffAb internals — third_party/diffab must be on sys.path before import.
from diffab.utils.data import PaddingCollate

# Our J-anchor-fixed parser (mirrors DiffAb's preprocess + raises the
# heavy-chain max_resseq from 113 → 150). Used for the loser side since
# AAPR PDBs are parsed on the fly rather than pre-cached.
from src.diffab_ft.datasets.vhh_andd import (
    DEFAULT_HEAVY_MAX_RESSEQ,
    _preprocess_vhh_structure,
)

logger = logging.getLogger(__name__)


# ── Pair sample type ─────────────────────────────────────────────────────
@dataclass
class PairSample:
    """One pair before collation. Stored as a dict to play nicely with
    PyTorch's default collator paths; this dataclass is just for docs."""
    winner: dict
    loser: dict
    pair_id: str
    gt_id: str


# ── Dataset ──────────────────────────────────────────────────────────────
class PairDataset(Dataset):
    """Yields aligned (winner, loser) DiffAb-format pairs.

    Parameters
    ----------
    pairs_parquet
        Path to the parquet emitted by ``select_pareto_pairs.py``. Must
        carry at least ``gt_complex_id``, ``loser_pdb_path``,
        ``winner_pdb_path``, ``pair_id``.
    base_dataset
        An already-constructed :class:`src.diffab_ft.datasets.vhh_andd
        .VHHANDDDataset` whose LMDB covers every GT referenced by the
        pairs parquet. We use it for two things: (a) winner-side
        structure loading via ``get_structure`` (no re-parse), and (b)
        per-GT manifest entry lookup so the loser can be parsed under
        the matching ``H_chain``/``ag_chains``.
    transform
        Composed transform (``MaskMultipleCDRs → MergeChains →
        PatchAroundAnchor``) — typically constructed by DiffAb's
        ``get_transform`` from the YAML.
    split
        ``"train"`` or ``"val"``. GT-stratified split.
    val_split_seed
        Seed for the GT shuffle used to pick the held-out val GTs.
        Identical across train/val constructions so the partition is
        coherent.
    val_gt_holdout
        Number of GTs to hold out entirely for val. Default 3 — with
        the 192-pair canary's ~25 GTs and ~7.7 pairs/GT, this yields
        roughly 23 val pairs / 169 train pairs (~12% val), which
        matches the design doc's "5-fold CV reasonable for 58 pairs"
        guidance scaled up. **Ignored if ``val_gt_ids`` is given.**
    val_gt_ids
        Optional explicit list of bare-PDB GT IDs to use as the val
        set. Overrides ``val_gt_holdout``/``val_split_seed`` — useful
        when you want a non-random, semantically-meaningful val pool
        (e.g., for a combined train+val AAPR run, use the original
        fine-tune val-split GTs as DPO val so ``L_w_ref > 0`` and
        the val DPO loss has the symmetric structure DPO assumes,
        making early-stop decisions cleaner). Unknown IDs are
        ignored with a warning; if no IDs survive the filter, we
        fall back to the random ``val_gt_holdout`` path.
    heavy_max_resseq
        J-anchor cap forwarded to the loser parser. Default 150 (the
        post-J-anchor-fix value).
    pair_seed_offset
        Added to the per-pair RNG seed so different epochs/runs can
        sample different CDR-subset choices. Set deterministically per
        epoch by the trainer if desired.
    drop_misaligned
        If True, log and skip pairs whose post-transform generate_flag
        differs between winner and loser. False = raise.
    """

    def __init__(
        self,
        pairs_parquet: str | Path,
        base_dataset,
        transform,
        *,
        split: str = "train",
        val_split_seed: int = 42,
        val_gt_holdout: int = 3,
        val_gt_ids: Optional[Sequence[str]] = None,
        heavy_max_resseq: int = DEFAULT_HEAVY_MAX_RESSEQ,
        pair_seed_offset: int = 0,
        drop_misaligned: bool = True,
    ):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        self.transform = transform
        self.base_dataset = base_dataset
        self.heavy_max_resseq = int(heavy_max_resseq)
        self.pair_seed_offset = int(pair_seed_offset)
        self.drop_misaligned = bool(drop_misaligned)

        df = pd.read_parquet(pairs_parquet)
        if not len(df):
            raise RuntimeError(f"Empty pairs parquet: {pairs_parquet}")
        for required in ("gt_complex_id", "loser_pdb_path", "pair_id"):
            if required not in df.columns:
                raise KeyError(
                    f"pairs parquet missing required column {required!r}: "
                    f"{pairs_parquet}"
                )

        # Build the bare-PDB → manifest entry lookup. AAPR's gt_complex_id
        # is the bare PDB code (see _normalize_complex_id in the
        # pair-selection script); the base dataset uses '{pdb}_{H}'.
        live_ids = set(base_dataset.db_ids or [])
        self._pdb_to_entry: dict[str, dict] = {}
        for entry in base_dataset.sabdab_entries:
            if entry["id"] not in live_ids:
                continue  # silently skip entries that didn't make it into LMDB
            self._pdb_to_entry[entry["pdbcode"]] = entry

        # Drop pairs whose GT isn't in the LMDB — defensive. With the
        # post-j-fix dataset every test-split GT *should* be present,
        # but we guard against operator error (wrong split, wrong LMDB).
        before = len(df)
        df = df[df["gt_complex_id"].isin(self._pdb_to_entry)].reset_index(drop=True)
        n_dropped_no_gt = before - len(df)
        if n_dropped_no_gt:
            logger.warning(
                "PairDataset: dropped %d pairs whose GT is not in the base "
                "dataset's LMDB. Check that base_dataset.split covers the "
                "pairs parquet's GTs.", n_dropped_no_gt,
            )

        # GT-stratified train/val split — explicit list takes priority,
        # otherwise random holdout.
        all_gts = sorted(df["gt_complex_id"].unique().tolist())
        val_gts: set[str]
        split_mode: str
        if val_gt_ids:
            requested = set(map(str, val_gt_ids))
            present = set(all_gts)
            val_gts = requested & present
            missing = requested - present
            if missing:
                logger.warning(
                    "PairDataset: %d val_gt_ids not present in the pairs "
                    "parquet (sample: %s). Using %d that are present.",
                    len(missing), sorted(missing)[:5], len(val_gts),
                )
            if not val_gts:
                logger.warning(
                    "PairDataset: val_gt_ids matched 0 GTs; falling back "
                    "to random holdout of %d GTs (seed=%d).",
                    val_gt_holdout, val_split_seed,
                )
                rng = random.Random(val_split_seed)
                shuffled = list(all_gts)
                rng.shuffle(shuffled)
                val_gts = set(shuffled[: int(val_gt_holdout)])
                split_mode = f"random (fallback, holdout={val_gt_holdout})"
            else:
                split_mode = f"explicit (val_gt_ids n={len(val_gts)})"
        else:
            rng = random.Random(val_split_seed)
            shuffled = list(all_gts)
            rng.shuffle(shuffled)
            val_gts = set(shuffled[: int(val_gt_holdout)])
            split_mode = f"random (holdout={val_gt_holdout}, seed={val_split_seed})"

        if split == "val":
            df = df[df["gt_complex_id"].isin(val_gts)].reset_index(drop=True)
        else:
            df = df[~df["gt_complex_id"].isin(val_gts)].reset_index(drop=True)

        if not len(df):
            raise RuntimeError(
                f"PairDataset[{split}] empty after GT-holdout split "
                f"(val_gt_holdout={val_gt_holdout}, val_split_seed={val_split_seed}). "
                f"Consider lowering val_gt_holdout."
            )

        self.pairs_df = df
        logger.info(
            "PairDataset[%s]: %d pairs across %d GTs (split_mode=%s).",
            split, len(df), df["gt_complex_id"].nunique(), split_mode,
        )
        # Cache of {pair_idx: (winner_data, loser_data)} — populated on
        # first access if caching is enabled. Disabled by default: with
        # transforms doing random masking, caching breaks epoch-to-epoch
        # variation. Re-parsing the loser PDB is ~50 ms — negligible
        # next to the 2 s DPO step time.
        # (No persistent cache; this attribute reserved for future use.)

    # ── PyTorch hooks ────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.pairs_df)

    def __getitem__(self, idx: int) -> dict:
        row = self.pairs_df.iloc[idx]
        gt_complex_id = str(row["gt_complex_id"])
        pair_id = str(row["pair_id"])
        loser_pdb_path = str(row["loser_pdb_path"])

        entry = self._pdb_to_entry[gt_complex_id]

        # Winner: pull from LMDB and deep-copy so transforms don't mutate
        # the cached version (transforms set generate_flag in-place).
        winner_raw = copy.deepcopy(self.base_dataset.get_structure(entry["id"]))

        # Loser: parse on the fly using the SAME entry (so H_chain etc.
        # match the winner). _preprocess_vhh_structure expects a "task"
        # dict; we use a synthetic id so error messages identify the
        # loser side cleanly.
        loser_raw = self._parse_loser(loser_pdb_path, entry)
        if loser_raw is None:
            raise RuntimeError(
                f"Failed to parse loser PDB at idx={idx} pair={pair_id}: "
                f"{loser_pdb_path}"
            )

        # Align the pair so the masking pipeline produces byte-identical
        # generate_flag / patch_idx between winner and loser. Three
        # corrections, all on the loser (winner is canonical):
        #
        # (a) cdr_flag — AAPR PDBs are sequentially numbered, GTs are
        #     Chothia-numbered; `_label_heavy_chain_cdr` keys CDR labels
        #     off resseq against Chothia ranges, so the same physical
        #     CDR gets shifted cdr_flag indices. Copy winner's cdr_flag
        #     onto the loser to relabel consistently.
        # (b) Framework positions — `PatchAroundAnchor` ranks residues
        #     by `cdist(pos_alpha, anchor_points)` and picks the top-128
        #     closest. Even sub-Å perturbations in framework positions
        #     (from AAPR rounding, write/read precision) shift the
        #     dist_anchor ranking and break topk ties differently,
        #     yielding different patches on the same physical structure.
        #     AAPR is *intended* to carry the framework verbatim from
        #     the GT, so byte-copying the winner's framework positions
        #     onto the loser is a no-op in the ideal case and a
        #     precision-correction in the realistic case.
        # (c) Antigen positions — same reasoning as (b); AAPR carries
        #     antigen verbatim.
        #
        # CRITICAL: we do NOT copy `aa` or the CDR-region positions.
        # Those are the per-residue content the DPO loss is supposed
        # to discriminate between. The copy is restricted to:
        #   * cdr_flag (relabeling, no structural change)
        #   * pos_heavyatom / mask_heavyatom at framework positions
        #     (where cdr_flag == 0)
        #   * antigen pos_heavyatom / mask_heavyatom (whole antigen)
        #
        # If the heavy-chain residue counts disagree (AAPR dropped/
        # added a residue), copying is unsafe and we let the alignment
        # guard below skip the pair.
        if winner_raw.get("heavy") is not None and loser_raw.get("heavy") is not None:
            w_heavy = winner_raw["heavy"]
            l_heavy = loser_raw["heavy"]
            if w_heavy["aa"].size(0) == l_heavy["aa"].size(0):
                # (a) cdr_flag transfer
                l_heavy["cdr_flag"] = w_heavy["cdr_flag"].clone()
                for k in ("H1_seq", "H2_seq", "H3_seq"):
                    if k in w_heavy:
                        l_heavy[k] = w_heavy[k]
                # (b) framework position transfer — clone first so we
                # don't mutate the freshly-parsed loser tensor in place
                # (paranoid: it isn't shared, but the cost is trivial).
                fw_mask = (w_heavy["cdr_flag"] == 0)
                l_heavy["pos_heavyatom"] = l_heavy["pos_heavyatom"].clone()
                l_heavy["mask_heavyatom"] = l_heavy["mask_heavyatom"].clone()
                l_heavy["pos_heavyatom"][fw_mask] = (
                    w_heavy["pos_heavyatom"][fw_mask].clone()
                )
                l_heavy["mask_heavyatom"][fw_mask] = (
                    w_heavy["mask_heavyatom"][fw_mask].clone()
                )
        # (c) antigen transfer
        if winner_raw.get("antigen") is not None and loser_raw.get("antigen") is not None:
            w_ag = winner_raw["antigen"]
            l_ag = loser_raw["antigen"]
            if w_ag["aa"].size(0) == l_ag["aa"].size(0):
                l_ag["pos_heavyatom"] = w_ag["pos_heavyatom"].clone()
                l_ag["mask_heavyatom"] = w_ag["mask_heavyatom"].clone()

        # Apply identical transforms with pinned RNG so generate_flag
        # and patch indexing align across the pair. Seed = stable hash
        # over (pair_id, offset) — same for winner and loser, varies
        # across pairs/epochs.
        pair_seed = self._pair_seed(pair_id)
        winner_data = self._apply_transforms(winner_raw, pair_seed)
        loser_data = self._apply_transforms(loser_raw, pair_seed)

        if not torch.equal(
            winner_data["generate_flag"], loser_data["generate_flag"]
        ):
            w_heavy_len = (
                winner_raw["heavy"]["aa"].size(0)
                if winner_raw.get("heavy") is not None else -1
            )
            l_heavy_len = (
                loser_raw["heavy"]["aa"].size(0)
                if loser_raw.get("heavy") is not None else -1
            )
            msg = (
                f"PairDataset: generate_flag mismatch for pair {pair_id} "
                f"(winner sum={int(winner_data['generate_flag'].sum())} "
                f"len={winner_data['generate_flag'].size(0)}, "
                f"loser sum={int(loser_data['generate_flag'].sum())} "
                f"len={loser_data['generate_flag'].size(0)}; "
                f"heavy_len pre-transform winner={w_heavy_len} "
                f"loser={l_heavy_len}). With cdr_flag + framework + "
                f"antigen position transfers in place, this should "
                f"only fire on residue-count mismatch (AAPR dropped/"
                f"added a residue)."
            )
            if self.drop_misaligned:
                logger.warning(msg)
                return self.__getitem__((idx + 1) % len(self))
            raise RuntimeError(msg)

        return {
            "winner": winner_data,
            "loser": loser_data,
            "pair_id": pair_id,
            "gt_id": gt_complex_id,
        }

    # ── Helpers ─────────────────────────────────────────────────────
    def _parse_loser(self, pdb_path: str, gt_entry: dict) -> Optional[dict]:
        """Parse an AAPR-sampled PDB using the GT's chain IDs.

        Critical: we pass the *GT entry* (with its H_chain, ag_chains)
        as the parsing key. AAPR carries the antigen + framework chains
        over verbatim, so the chain IDs match. Using the GT's entry
        also guarantees the same CDR-labeling protocol is applied
        (chothia ranges via ``_label_heavy_chain_cdr`` inside
        ``_preprocess_vhh_structure``).
        """
        task = {
            "id": f"loser__{Path(pdb_path).stem}",
            "entry": gt_entry,
            "pdb_path": pdb_path,
        }
        return _preprocess_vhh_structure(task, self.heavy_max_resseq)

    def _apply_transforms(self, structure: dict, seed: int) -> dict:
        """Run the masking pipeline under a pinned RNG.

        Pinning ``random.seed`` covers ``MaskMultipleCDRs.mask_for_one_chain_``
        (which uses ``random.randint``/``random.shuffle``/``random.choice``)
        and ``random_shrink_extend``. We also pin ``torch.manual_seed``
        defensively for any ``torch.*`` random op a downstream transform
        might add.
        """
        random.seed(seed)
        torch.manual_seed(seed)
        # `transform` is a torchvision.Compose; calling it returns a
        # new dict (transforms mutate in-place but the pipeline already
        # deep-modifies the structure). The deep-copy at the call site
        # protects the LMDB-backed winner.
        return self.transform(structure)

    def _pair_seed(self, pair_id: str) -> int:
        """Stable per-pair seed, shifted by offset.

        Python's built-in ``hash`` is salted per-interpreter in Python
        3.3+, so we use a manual deterministic hash. We want the same
        seed for both halves of a pair (call site enforces this by
        re-using the seed), and varied across pairs.
        """
        # zlib.adler32 is fast, deterministic, fits in 32 bits.
        import zlib
        h = zlib.adler32(pair_id.encode("utf-8"))
        return (h ^ self.pair_seed_offset) & 0x7FFFFFFF


# ── Collate ──────────────────────────────────────────────────────────────
class PairCollate:
    """Pad winner and loser to a common max length per *pair batch*.

    DiffAb's :class:`PaddingCollate` pads to the max within a list. For
    DPO we need every winner *and* every loser in a batch to land on
    the same padded length so the per-residue tensors line up across
    (winner, loser, batch). We achieve this by:

      1. Computing ``max_len`` over the union of winners + losers in
         the batch.
      2. Padding each side independently to that single ``max_len``
         using DiffAb's per-key pad values.

    We default ``eight=False`` because the multiple-of-8 rounding
    (originally an FP16 nicety) would silently grow the padded length
    differently if winner-max and loser-max round into different buckets.
    """

    def __init__(self, eight: bool = False):
        # Re-use DiffAb's PaddingCollate for its pad-value table and
        # _pad_last/_get_pad_mask helpers; we just override the
        # max-length computation.
        self._base = PaddingCollate(eight=eight)
        self.eight = bool(eight)

    def __call__(self, batch: list[dict]) -> dict:
        winners = [b["winner"] for b in batch]
        losers = [b["loser"] for b in batch]
        max_len = max(
            max(d["aa"].size(0) for d in winners),
            max(d["aa"].size(0) for d in losers),
        )
        if self.eight:
            max_len = math.ceil(max_len / 8) * 8

        return {
            "winner": self._collate_side(winners, max_len),
            "loser": self._collate_side(losers, max_len),
            "pair_id": [b["pair_id"] for b in batch],
            "gt_id": [b["gt_id"] for b in batch],
        }

    def _collate_side(self, data_list: list[dict], max_length: int) -> dict:
        keys = PaddingCollate._get_common_keys(data_list)
        padded = []
        for data in data_list:
            d = {
                k: (
                    PaddingCollate._pad_last(
                        v, max_length, value=self._base._get_pad_value(k),
                    )
                    if k not in self._base.no_padding
                    else v
                )
                for k, v in data.items()
                if k in keys
            }
            d["mask"] = PaddingCollate._get_pad_mask(
                data[self._base.length_ref_key].size(0), max_length,
            )
            padded.append(d)
        return default_collate(padded)


# ── Convenience: build a PairDataset from a YAML-style config ──────────
def build_pair_dataset_from_config(
    cfg,
    base_dataset,
    transform,
    *,
    split: str,
) -> PairDataset:
    """Wrapper that pulls DPO-relevant fields from an EasyDict cfg."""
    dpo = cfg.dpo
    return PairDataset(
        pairs_parquet=dpo.pair_parquet,
        base_dataset=base_dataset,
        transform=transform,
        split=split,
        val_split_seed=int(dpo.get("val_split_seed", 42)),
        val_gt_holdout=int(dpo.get("val_gt_holdout", 3)),
        val_gt_ids=dpo.get("val_gt_ids", None),
        heavy_max_resseq=int(
            cfg.dataset.train.get("heavy_max_resseq", DEFAULT_HEAVY_MAX_RESSEQ)
        ),
        pair_seed_offset=int(dpo.get("pair_seed_offset", 0)),
        drop_misaligned=bool(dpo.get("drop_misaligned", True)),
    )


# ── Helper to recover the masking pipeline as Python objects ────────────
def build_pair_transforms(transform_cfg) -> any:
    """Build the same Compose that DiffAb's get_transform would.

    Exists separately because for DPO we may want a slightly different
    pipeline (e.g., fixed CDR selection) without re-running the dataset
    YAML through the full ``get_transform`` codepath.
    """
    from diffab.utils.transforms import get_transform
    return get_transform(transform_cfg)
