"""Masking strategies for the AAPR loop.

Implements four strategies divided into two phases:

Phase 1 — Candidate Generation (positive DPO examples):
  PARATOPE:     Structure-based interface masking with anchor/FR2 protection.
  CDR_FOCUSED:  Sequence-only CDR masking (fallback when no structure).

Phase 2 — Hard Negative Mining (negative DPO examples):
  FR2_REVERSION:    Mask FR2 region to induce VH-like hydrophobic reversion.
  UNANCHORED_CLASH: Mask full paratope without anchors to induce steric clashes.

References:
- Paratope masking: Leem et al. (2022), Mitchell & Colwell (2018)
- CDR-focused / PARA: Gao et al. (2023)
- Anchor protection: GeoGAD (Tan et al., 2024), AbFlex (Ruffolo et al., 2024)
- FR2 hallmark VHH vs VH: Desmyter et al. (2015)
- Hard negative mining: Dignum (2026) pipeline proposal
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto

from src.masking.kabat_mapper import KabatMapping, PDBResidueMap

logger = logging.getLogger(__name__)


class MaskStrategy(Enum):
    """The four masking strategies for the AAPR loop."""

    PARATOPE = auto()
    CDR_FOCUSED = auto()
    FR2_REVERSION = auto()
    UNANCHORED_CLASH = auto()


@dataclass
class MaskResult:
    """Output of a masking strategy.

    Attributes:
        masked_positions: Set of 0-based indices in ``raw_sequence``
                          to mask.
        strategy: Which strategy produced this mask.
        contiguous_spans: Positions grouped into ``(start_inclusive,
                          end_exclusive)`` tuples for IgLM infilling.
                          Each span becomes one IgLM call.
        metadata: Strategy-specific metadata for logging and thesis
                  documentation (e.g. number of paratope residues,
                  anchor positions subtracted, citations).
    """

    masked_positions: set[int]
    strategy: MaskStrategy
    contiguous_spans: list[tuple[int, int]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# ─── Shared Helper ────────────────────────────────────────────────────


def _compute_contiguous_spans(
    positions: set[int],
) -> list[tuple[int, int]]:
    """Group a set of positions into contiguous ``(start, end)`` spans.

    Each span is ``[start_inclusive, end_exclusive)``.  Adjacent
    positions (diff == 1) are merged.

    Example::

        >>> _compute_contiguous_spans({3, 4, 5, 10, 11, 20})
        [(3, 6), (10, 12), (20, 21)]
    """
    if not positions:
        return []

    sorted_pos = sorted(positions)
    spans: list[tuple[int, int]] = []
    span_start = sorted_pos[0]
    prev = sorted_pos[0]

    for pos in sorted_pos[1:]:
        if pos == prev + 1:
            prev = pos
        else:
            spans.append((span_start, prev + 1))
            span_start = pos
            prev = pos

    spans.append((span_start, prev + 1))
    return spans


# ─── Strategy 1: Paratope Mask ────────────────────────────────────────


def paratope_mask(
    kabat_mapping: KabatMapping,
    pdb_map: PDBResidueMap,
    paratope_resseqs: set[int],
) -> MaskResult:
    """Interface-biased mask with anchor and FR2 protection.

    Masks all paratope residues (identified by NeighborSearch), then
    subtracts anchor residues flanking each CDR and FR2 hallmark
    positions.

    This is the primary strategy for generating positive binding
    candidates in the AAPR loop.

    Args:
        kabat_mapping: Full Kabat mapping with CDR, anchor, and FR2
                       indices precomputed.
        pdb_map: PDB resseq-to-sequence-index mapping.
        paratope_resseqs: Set of PDB resseq integers for paratope
                          residues (from ``detect_paratope_residues``).

    Returns:
        MaskResult with paratope positions minus anchors and FR2.
    """
    # Convert paratope PDB resseqs to sequence indices.
    paratope_seq_indices: set[int] = set()
    for resseq in paratope_resseqs:
        if resseq in pdb_map.pdb_resid_to_seq_idx:
            paratope_seq_indices.add(pdb_map.pdb_resid_to_seq_idx[resseq])

    # Subtract protected positions.
    masked = (
        paratope_seq_indices
        - kabat_mapping.anchor_indices
        - kabat_mapping.fr2_hallmark_indices
    )

    spans = _compute_contiguous_spans(masked)

    logger.info(
        "Paratope mask: %d paratope residues -> %d after anchor/FR2 protection -> %d spans",
        len(paratope_seq_indices),
        len(masked),
        len(spans),
    )

    return MaskResult(
        masked_positions=masked,
        strategy=MaskStrategy.PARATOPE,
        contiguous_spans=spans,
        metadata={
            "total_paratope_residues": len(paratope_seq_indices),
            "anchors_subtracted": len(
                paratope_seq_indices & kabat_mapping.anchor_indices
            ),
            "fr2_subtracted": len(
                paratope_seq_indices & kabat_mapping.fr2_hallmark_indices
            ),
        },
    )


# ─── Strategy 2: CDR-Focused Mask ────────────────────────────────────


def cdr_focused_mask(kabat_mapping: KabatMapping) -> MaskResult:
    """CDR-focused mask with anchor and FR2 protection.

    Masks all CDR positions (CDR1 + CDR2 + CDR3) using the Kabat
    boundaries from ANARCI's HMM alignment.  Subtracts anchor
    residues and FR2 hallmarks.

    Used as a fallback when no complex PDB structure is available
    for paratope detection.

    Args:
        kabat_mapping: Full Kabat mapping with CDR, anchor, and FR2
                       indices precomputed.

    Returns:
        MaskResult with CDR positions minus anchors and FR2.
    """
    # Start with all CDR positions.
    masked = (
        kabat_mapping.cdr_indices.copy()
        - kabat_mapping.anchor_indices
        - kabat_mapping.fr2_hallmark_indices
    )

    spans = _compute_contiguous_spans(masked)

    logger.info(
        "CDR-focused mask: %d CDR residues -> %d after anchor/FR2 protection -> %d spans",
        len(kabat_mapping.cdr_indices),
        len(masked),
        len(spans),
    )

    return MaskResult(
        masked_positions=masked,
        strategy=MaskStrategy.CDR_FOCUSED,
        contiguous_spans=spans,
        metadata={
            "total_cdr_residues": len(kabat_mapping.cdr_indices),
            "anchors_subtracted": len(
                kabat_mapping.cdr_indices & kabat_mapping.anchor_indices
            ),
        },
    )


# ─── Strategy 3: FR2 Reversion Mask ──────────────────────────────────


def fr2_reversion_mask(kabat_mapping: KabatMapping) -> MaskResult:
    """FR2 region mask for biological hard negative mining.

    Masks the entire FR2 region (from ``chain.regions["FR2"]``) as one
    contiguous span.  This includes the hallmark positions (37, 44, 45,
    47) plus surrounding FR2 residues.

    No anchor or FR2 protection — the goal is to let IgLM's
    VH-dominated training data naturally insert conventional VH-like
    hydrophobic residues, producing aggregation-prone sequences that
    the Biology Judge should reject via SAP analysis.

    Args:
        kabat_mapping: Full Kabat mapping (uses ``fr2_region_indices``).

    Returns:
        MaskResult covering the full FR2 region.
    """
    masked = kabat_mapping.fr2_region_indices.copy()
    spans = _compute_contiguous_spans(masked)

    logger.info(
        "FR2 reversion mask: %d positions (full FR2 region) -> %d spans",
        len(masked),
        len(spans),
    )

    return MaskResult(
        masked_positions=masked,
        strategy=MaskStrategy.FR2_REVERSION,
        contiguous_spans=spans,
        metadata={
            "fr2_region_size": len(masked),
            "hallmark_positions_included": len(
                masked & kabat_mapping.fr2_hallmark_indices
            ),
        },
    )


# ─── Strategy 4: Unanchored Clash Mask ────────────────────────────────


def unanchored_clash_mask(
    kabat_mapping: KabatMapping,
    pdb_map: PDBResidueMap,
    paratope_resseqs: set[int],
) -> MaskResult:
    """Full paratope mask WITHOUT anchor or FR2 protection.

    Masks the entire paratope (same residues as Strategy 1) PLUS
    includes the anchor residues that Strategy 1 protects.  FR2
    hallmarks are also NOT protected.

    Without geometric anchor constraints, IgLM generates CDR loops
    that protrude into the antigen surface, causing steric clashes.
    The Physics Judge should reject these via E_Rep > 5.0 REU.

    Args:
        kabat_mapping: Full Kabat mapping with CDR and anchor indices.
        pdb_map: PDB resseq-to-sequence-index mapping.
        paratope_resseqs: Set of PDB resseq integers for paratope
                          residues (from ``detect_paratope_residues``).

    Returns:
        MaskResult with full paratope plus anchors (no protection).
    """
    # Convert paratope PDB resseqs to sequence indices.
    paratope_seq_indices: set[int] = set()
    for resseq in paratope_resseqs:
        if resseq in pdb_map.pdb_resid_to_seq_idx:
            paratope_seq_indices.add(pdb_map.pdb_resid_to_seq_idx[resseq])

    # Include anchors adjacent to paratope CDR positions (no protection).
    masked = paratope_seq_indices | (
        kabat_mapping.anchor_indices & _get_adjacent_anchors(
            paratope_seq_indices, kabat_mapping.anchor_indices
        )
    )

    spans = _compute_contiguous_spans(masked)

    logger.info(
        "Unanchored clash mask: %d paratope + anchors -> %d total -> %d spans",
        len(paratope_seq_indices),
        len(masked),
        len(spans),
    )

    return MaskResult(
        masked_positions=masked,
        strategy=MaskStrategy.UNANCHORED_CLASH,
        contiguous_spans=spans,
        metadata={
            "total_paratope_residues": len(paratope_seq_indices),
            "anchors_included": len(masked - paratope_seq_indices),
        },
    )


def _get_adjacent_anchors(
    paratope_indices: set[int],
    all_anchor_indices: set[int],
) -> set[int]:
    """Find anchor residues adjacent to any paratope residue.

    Returns anchors that are within 5 sequence positions of a paratope
    residue, ensuring we include the relevant flanking anchors for CDR
    loops that contain paratope residues, but do not include anchors
    from distant, non-paratope CDR regions.
    """
    adjacent: set[int] = set()
    for anchor_idx in all_anchor_indices:
        for para_idx in paratope_indices:
            if abs(anchor_idx - para_idx) <= 5:
                adjacent.add(anchor_idx)
                break
    return adjacent
