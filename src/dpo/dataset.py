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
from diffab.utils.protein.constants import BBHeavyAtom, Fragment
from diffab.utils.transforms._base import _mask_select_data
from diffab.utils.transforms.patch import PatchAroundAnchor

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

        # Winner-source routing summary — counts how many pairs will pull
        # winners from the LMDB (floor: winner = GT crystal) vs from disk
        # (decoy: winner_provenance set, winner_pdb_path consumed). If
        # this prints `disk=0` on a pool that was supposed to have decoys
        # in it, you have a manifest-vs-loader drift bug like the one
        # diagnosed on 2026-06-08 (see brief 17 deliverable).
        n_disk_winners = 0
        if "winner_provenance" in df.columns:
            n_disk_winners = int(
                df["winner_provenance"].fillna("").astype(str).str.strip().ne("").sum()
            )
        n_lmdb_winners = len(df) - n_disk_winners
        logger.info(
            "PairDataset[%s]: %d pairs across %d GTs (split_mode=%s) | "
            "winner sources: lmdb=%d disk=%d",
            split, len(df), df["gt_complex_id"].nunique(), split_mode,
            n_lmdb_winners, n_disk_winners,
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

        # GT scaffold — the canonical Chothia-numbered structure from
        # the LMDB. Used as the alignment reference for cdr_flag,
        # framework positions, and antigen positions on BOTH sides
        # whenever the parsed side is sequentially-numbered (AAPR
        # losers, decoy winners). In floor mode the winner IS this
        # object, so the existing transfer semantics are preserved.
        gt_raw = copy.deepcopy(self.base_dataset.get_structure(entry["id"]))

        winner_source, winner_pdb_path = self._resolve_winner_source(row)
        if winner_source == "lmdb":
            # Floor pair: winner is the GT crystal. Same object as the
            # scaffold; downstream transforms mutate it in place, but
            # the scaffold-transfer step below short-circuits when
            # winner is gt_raw, so no aliasing harm.
            winner_raw = gt_raw
        else:
            # Decoy / non-GT winner: parse from disk under the GT's
            # chain IDs, same way losers are parsed. The scaffold-
            # transfer step below realigns it to gt_raw.
            winner_raw = self._parse_pdb(
                winner_pdb_path, entry, side_tag=f"winner__{row.get('winner_provenance', 'disk')}",
            )
            if winner_raw is None:
                raise RuntimeError(
                    f"Failed to parse winner PDB at idx={idx} pair={pair_id}: "
                    f"{winner_pdb_path}"
                )

        # Loser: parse on the fly using the SAME entry (so H_chain etc.
        # match the GT scaffold). _preprocess_vhh_structure expects a
        # "task" dict; we use a synthetic id so error messages identify
        # the loser side cleanly.
        loser_raw = self._parse_pdb(loser_pdb_path, entry, side_tag="loser")
        if loser_raw is None:
            raise RuntimeError(
                f"Failed to parse loser PDB at idx={idx} pair={pair_id}: "
                f"{loser_pdb_path}"
            )

        # Align both sides to the GT scaffold so the masking pipeline
        # produces byte-identical generate_flag / patch_idx across the
        # pair. Three corrections, applied to any disk-parsed side:
        #
        # (a) cdr_flag — AAPR / decoy PDBs are sequentially numbered,
        #     GTs are Chothia-numbered; `_label_heavy_chain_cdr` keys
        #     CDR labels off resseq against Chothia ranges, so the same
        #     physical CDR gets shifted cdr_flag indices. Copy the GT's
        #     cdr_flag onto the side to relabel consistently.
        # (b) Framework positions — `PatchAroundAnchor` ranks residues
        #     by `cdist(pos_alpha, anchor_points)` and picks the top-128
        #     closest. Even sub-Å perturbations in framework positions
        #     (from AAPR rounding, decoy reconstruction, write/read
        #     precision) shift the dist_anchor ranking and break topk
        #     ties differently, yielding different patches on the same
        #     physical structure. AAPR / decoys are *intended* to carry
        #     the framework verbatim from the GT, so byte-copying the
        #     scaffold's framework positions is a no-op in the ideal
        #     case and a precision-correction in the realistic case.
        # (c) Antigen positions — same reasoning as (b); AAPR / decoys
        #     carry the antigen verbatim.
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
        if winner_raw is not gt_raw:
            self._align_to_gt_scaffold(winner_raw, gt_raw)
        self._align_to_gt_scaffold(loser_raw, gt_raw)

        # Apply masking + merge + patch with shared patch_mask across
        # the pair. Naively running self.transform on each side breaks
        # alignment at the patch step: PatchAroundAnchor ranks residues
        # by `cdist(positions, anchor_points)` and picks the top-128,
        # but the CDR positions necessarily differ between winner (GT)
        # and loser (AAPR) — that's the DPO signal we want to keep —
        # so the top-128 boundary resolves differently and the
        # patch_mask ends up with different residue subsets (typically
        # ±1 residue). The aligned variant below computes patch_mask
        # on the winner side only and applies it to both, preserving
        # per-residue alignment for the AbDPO loss while keeping the
        # loser's actual aa/pos at CDR positions intact.
        pair_seed = self._pair_seed(pair_id)
        winner_data, loser_data = self._apply_pair_transforms(
            winner_raw, loser_raw, pair_seed,
        )

        if winner_data is None or loser_data is None:
            # _apply_pair_transforms returned a skip signal — post-merge
            # tensor lengths disagreed (heavy or antigen residue counts
            # differed between winner and loser even after our position
            # transfers tried to align them). Move to the next pair.
            msg = (
                f"PairDataset: pre-patch length mismatch for pair {pair_id} "
                f"(skipping). Likely cause: AAPR PDB has a different number "
                f"of heavy or antigen residues than the GT."
            )
            if self.drop_misaligned:
                logger.warning(msg)
                return self.__getitem__((idx + 1) % len(self))
            raise RuntimeError(msg)

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
    @staticmethod
    def _resolve_winner_source(row) -> tuple[str, Optional[str]]:
        """Route winner loading: LMDB (floor) vs disk (decoy / non-GT).

        Returns ``("lmdb", None)`` when the pair does not carry a
        ``winner_provenance`` sentinel (i.e., the winner is the GT
        crystal pulled from the LMDB by ``gt_complex_id``); returns
        ``("disk", winner_pdb_path)`` when ``winner_provenance`` is
        present and non-empty, signalling the winner is a PDB on disk
        that should be parsed the same way losers are.

        Pure function of one parquet row; isolated here so the unit
        test in ``scripts/test_pair_dataset_winner_routing.py`` can
        exercise the routing logic without instantiating a
        ``PairDataset``. Defends against the 2026-06-08 bug where the
        winner_pdb_path swap was a manifest no-op because nothing
        consumed the column.
        """
        provenance = ""
        if hasattr(row, "get"):
            raw = row.get("winner_provenance", "")
        else:
            raw = row["winner_provenance"] if "winner_provenance" in row else ""
        # pandas converts Python None → NaN when materialising a Series
        # with mixed/object dtype, so check both None and NaN before
        # stringifying (str(float('nan')) == 'nan', which would survive
        # the .strip() check and silently route to disk).
        if raw is None or (isinstance(raw, float) and raw != raw):
            raw = ""
        try:
            provenance = str(raw).strip()
        except Exception:  # noqa: BLE001
            provenance = ""
        # Also treat literal 'nan' / 'none' (case-insensitive) as blank
        # — defensive against parquet round-trips that stringify NaN.
        if provenance.lower() in ("", "nan", "none"):
            return "lmdb", None
        return "disk", str(row["winner_pdb_path"])

    @staticmethod
    def _align_to_gt_scaffold(side_raw: dict, gt_raw: dict) -> None:
        """Copy cdr_flag / framework / antigen from gt_raw onto side_raw.

        In-place. No-op when residue counts disagree (the alignment
        guard in __getitem__ will drop the pair). See the comment block
        in ``__getitem__`` for the full rationale.
        """
        if side_raw.get("heavy") is not None and gt_raw.get("heavy") is not None:
            gt_heavy = gt_raw["heavy"]
            side_heavy = side_raw["heavy"]
            if gt_heavy["aa"].size(0) == side_heavy["aa"].size(0):
                side_heavy["cdr_flag"] = gt_heavy["cdr_flag"].clone()
                for k in ("H1_seq", "H2_seq", "H3_seq"):
                    if k in gt_heavy:
                        side_heavy[k] = gt_heavy[k]
                fw_mask = (gt_heavy["cdr_flag"] == 0)
                side_heavy["pos_heavyatom"] = side_heavy["pos_heavyatom"].clone()
                side_heavy["mask_heavyatom"] = side_heavy["mask_heavyatom"].clone()
                side_heavy["pos_heavyatom"][fw_mask] = (
                    gt_heavy["pos_heavyatom"][fw_mask].clone()
                )
                side_heavy["mask_heavyatom"][fw_mask] = (
                    gt_heavy["mask_heavyatom"][fw_mask].clone()
                )
        if side_raw.get("antigen") is not None and gt_raw.get("antigen") is not None:
            gt_ag = gt_raw["antigen"]
            side_ag = side_raw["antigen"]
            if gt_ag["aa"].size(0) == side_ag["aa"].size(0):
                side_ag["pos_heavyatom"] = gt_ag["pos_heavyatom"].clone()
                side_ag["mask_heavyatom"] = gt_ag["mask_heavyatom"].clone()

    def _parse_pdb(
        self, pdb_path: str, gt_entry: dict, *, side_tag: str = "pdb",
    ) -> Optional[dict]:
        """Parse a PDB on disk under the GT's chain IDs.

        Used for both losers (AAPR-sampled) and decoy winners. Passing
        the *GT entry* (with its H_chain, ag_chains) ensures the chain
        IDs and CDR-labeling protocol match across all three sides
        (GT scaffold, winner, loser). ``side_tag`` only appears in
        error messages — it doesn't influence parsing.
        """
        task = {
            "id": f"{side_tag}__{Path(pdb_path).stem}",
            "entry": gt_entry,
            "pdb_path": pdb_path,
        }
        return _preprocess_vhh_structure(task, self.heavy_max_resseq)

    def _apply_transforms(self, structure: dict, seed: int) -> dict:
        """Run the masking pipeline under a pinned RNG.

        Kept for any single-side use (e.g., test utilities). The pair
        pipeline uses :meth:`_apply_pair_transforms` instead because it
        needs a single shared ``patch_mask`` across (winner, loser).
        """
        random.seed(seed)
        torch.manual_seed(seed)
        return self.transform(structure)

    def _apply_pair_transforms(
        self, winner_raw: dict, loser_raw: dict, seed: int,
    ) -> tuple[dict, dict]:
        """Apply masking + merge + patch to a pair with shared patch_mask.

        Steps:
          1. Decompose ``self.transform`` (Compose) into the pre-patch
             transforms (MaskMultipleCDRs, MergeChains) and the patch
             transform (PatchAroundAnchor).
          2. Run pre-patch transforms on both sides under the same
             seeded RNG. With cdr_flag and framework positions already
             aligned, ``MaskMultipleCDRs`` produces the same
             generate_flag + anchor_flag on both, and ``MergeChains``
             is deterministic, so the post-merge tensors have the same
             shape and the same flags. They differ only in aa codes at
             CDR positions (and pos_heavyatom at CDR positions, since
             those are intentionally not copied).
          3. Compute the patch on the *winner* using PatchAroundAnchor's
             own algorithm. Capture ``patch_mask`` and ``origin``.
          4. Apply the same ``patch_mask`` and centering origin to the
             loser. The loser keeps its own aa codes and CDR positions
             (the DPO signal); only the selection-and-centering of
             residues is borrowed from the winner.

        If the pre-patch merged tensors have different lengths (which
        means the heavy or antigen residue counts disagreed and the
        position-transfer at __getitem__ couldn't fix it), return
        ``(None, None)`` to signal an unrecoverable misalignment; the
        caller skips the pair.
        """
        # Locate the patch transform inside the Compose. We assume the
        # canonical order [MaskMultipleCDRs, MergeChains,
        # PatchAroundAnchor] — assert it so a config drift fails
        # loudly rather than silently producing misaligned pairs.
        compose_transforms = list(self.transform.transforms)
        patch_t = None
        pre_patch_ts = []
        for t in compose_transforms:
            if isinstance(t, PatchAroundAnchor):
                patch_t = t
            else:
                if patch_t is not None:
                    raise RuntimeError(
                        "PairDataset expects PatchAroundAnchor to be the "
                        "LAST transform in the pipeline; got transforms "
                        f"after it: {type(t).__name__}."
                    )
                pre_patch_ts.append(t)
        if patch_t is None:
            raise RuntimeError(
                "PairDataset expects a PatchAroundAnchor transform in the "
                "pipeline; found none. Update configs/dpo/vhh_dpo.yml's "
                "dataset.train.transform to include 'patch_around_anchor'."
            )

        # Pre-patch transforms on both, same RNG seed → same masks.
        random.seed(seed); torch.manual_seed(seed)
        winner_merged = winner_raw
        for t in pre_patch_ts:
            winner_merged = t(winner_merged)

        random.seed(seed); torch.manual_seed(seed)
        loser_merged = loser_raw
        for t in pre_patch_ts:
            loser_merged = t(loser_merged)

        if winner_merged["aa"].size(0) != loser_merged["aa"].size(0):
            # Heavy/antigen residue-count mismatch survived the
            # position-transfer at __getitem__. Cannot patch with a
            # shared mask. Signal skip.
            return None, None

        # Custom patch computation — mirrors PatchAroundAnchor.__call__
        # but exposes (patch_mask, origin) so we can apply them to the
        # loser as well. We use the WINNER's positions to compute
        # dist_anchor, so the topk selection is the same for both.
        patch_mask, origin = self._compute_patch_on_winner(
            winner_merged, patch_t,
        )

        winner_data = self._apply_patch(winner_merged, patch_mask, origin)
        loser_data = self._apply_patch(loser_merged, patch_mask, origin)
        return winner_data, loser_data

    @staticmethod
    def _compute_patch_on_winner(
        data: dict, patch_t: PatchAroundAnchor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mirror of PatchAroundAnchor's mask-construction logic.

        Returns ``(patch_mask, origin)``. The mask is True at residues
        that survive the patch; the origin is the centering point
        (anchor mean, or the antibody centroid in the no-anchor case).
        """
        anchor_flag = data["anchor_flag"]
        anchor_points = data["pos_heavyatom"][anchor_flag, BBHeavyAtom.CA]
        antigen_mask = (data["fragment_type"] == Fragment.Antigen)
        antibody_mask = torch.logical_not(antigen_mask)

        if anchor_flag.sum().item() == 0:
            # Full-antibody-Fv design (no anchor → no antigen-side
            # selection). Patch is the whole antibody; origin is the
            # antibody Cα centroid.
            patch_mask = antibody_mask.clone()
            origin = data["pos_heavyatom"][antibody_mask, BBHeavyAtom.CA].mean(dim=0)
            return patch_mask, origin

        pos_alpha = data["pos_heavyatom"][:, BBHeavyAtom.CA]
        dist_anchor = torch.cdist(pos_alpha, anchor_points).min(dim=1)[0]

        initial_patch_idx = torch.topk(
            dist_anchor,
            k=min(patch_t.initial_patch_size, dist_anchor.size(0)),
            largest=False,
        )[1]
        dist_anchor_antigen = dist_anchor.masked_fill(
            mask=antibody_mask, value=float("+inf"),
        )
        antigen_patch_idx = torch.topk(
            dist_anchor_antigen,
            k=min(patch_t.antigen_size, antigen_mask.sum().item()),
            largest=False, sorted=True,
        )[1]

        patch_mask = torch.logical_or(
            data["generate_flag"], data["anchor_flag"],
        ).clone()  # clone so we don't mutate the data dict's tensor
        patch_mask[initial_patch_idx] = True
        patch_mask[antigen_patch_idx] = True

        origin = anchor_points.mean(dim=0)
        return patch_mask, origin

    @staticmethod
    def _apply_patch(
        data: dict, patch_mask: torch.Tensor, origin: torch.Tensor,
    ) -> dict:
        """Apply a precomputed patch_mask + centering to a merged dict.

        Matches PatchAroundAnchor's output schema (sets ``origin`` and
        ``patch_idx`` on the returned dict). Note we re-compute
        ``patch_idx`` from ``patch_mask`` rather than borrowing it from
        the winner — they're equivalent since patch_mask is shared,
        but constructing it here keeps the dict self-consistent.
        """
        patch_idx = torch.arange(0, patch_mask.shape[0])[patch_mask]
        data_patch = _mask_select_data(data, patch_mask)

        origin_reshaped = origin.reshape(1, 1, 3)
        data_patch["pos_heavyatom"] = (
            data_patch["pos_heavyatom"] - origin_reshaped
        )
        data_patch["pos_heavyatom"] = (
            data_patch["pos_heavyatom"]
            * data_patch["mask_heavyatom"][:, :, None]
        )
        data_patch["origin"] = origin.reshape(3)
        data_patch["patch_idx"] = patch_idx
        return data_patch

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
