"""MaskingEngine: top-level orchestrator for sequence masking.

Coordinates Kabat mapping, paratope detection, and strategy dispatch
to produce masked positions for IgLM span infilling.

Usage::

    engine = MaskingEngine()

    # Strategy 1 — structure-based paratope masking (primary):
    result = engine.mask(candidate, MaskStrategy.PARATOPE)

    # Strategy 2 — sequence-only CDR masking (fallback):
    result = engine.mask(candidate, MaskStrategy.CDR_FOCUSED)

    # Strategy 3 — FR2 reversion (biological hard negatives):
    result = engine.mask(candidate, MaskStrategy.FR2_REVERSION)

    # Strategy 4 — unanchored clash (physical hard negatives):
    result = engine.mask(candidate, MaskStrategy.UNANCHORED_CLASH)

    # Access results:
    print(result.masked_positions)   # {3, 4, 5, 28, 29, ...}
    print(result.contiguous_spans)   # [(3, 6), (28, 30), ...]
"""

import logging
from pathlib import Path

from src.common.candidate import NanobodyCandidate
from src.common.config import Config
from src.common.pdb_utils import load_structure
from src.masking.kabat_mapper import (
    KabatMapping,
    PDBResidueMap,
    build_kabat_mapping,
    build_pdb_to_sequence_map,
)
from src.masking.paratope_detector import detect_paratope_residues
from src.masking.strategies import (
    MaskResult,
    MaskStrategy,
    cdr_focused_mask,
    fr2_reversion_mask,
    paratope_mask,
    unanchored_clash_mask,
)

logger = logging.getLogger(__name__)


class MaskingEngine:
    """Top-level masking interface for the AAPR loop.

    Encapsulates Kabat mapping, paratope detection, and strategy
    dispatch behind a single :meth:`mask` method.

    Args:
        distance_cutoff: Paratope detection distance in Angstroms.
        anchor_flank: Number of anchor residues flanking each CDR.
    """

    def __init__(
        self,
        distance_cutoff: float = Config.PARATOPE_DISTANCE_CUTOFF,
        anchor_flank: int = Config.ANCHOR_FLANK_SIZE,
    ):
        self._distance_cutoff = distance_cutoff
        self._anchor_flank = anchor_flank
        # Cache KabatMapping per sequence (deterministic).
        self._kabat_cache: dict[str, KabatMapping] = {}

    def _get_kabat_mapping(self, sequence: str) -> KabatMapping:
        """Get or compute the Kabat mapping for a sequence."""
        if sequence not in self._kabat_cache:
            self._kabat_cache[sequence] = build_kabat_mapping(
                sequence, anchor_flank=self._anchor_flank
            )
        return self._kabat_cache[sequence]

    def _load_structure_data(
        self,
        candidate: NanobodyCandidate,
        kabat_mapping: KabatMapping,
    ) -> tuple[PDBResidueMap, set[int]]:
        """Load PDB structure, detect paratope, and build PDB map.

        Returns:
            Tuple of (PDBResidueMap, paratope_resseqs).

        Raises:
            ValueError: If required structural data is missing.
        """
        complex_pdb = candidate.complex_pdb_path
        nb_chain = candidate.nanobody_chain_id or "H"
        ag_chains = candidate.antigen_chain_ids

        if not complex_pdb or not Path(complex_pdb).exists():
            raise ValueError(
                f"Candidate {candidate.candidate_id}: complex PDB not available "
                f"at {complex_pdb!r}"
            )
        if not ag_chains:
            raise ValueError(
                f"Candidate {candidate.candidate_id}: antigen_chain_ids not set"
            )

        structure = load_structure(complex_pdb, candidate.candidate_id)

        pdb_map = build_pdb_to_sequence_map(
            structure, nb_chain, candidate.raw_sequence
        )
        paratope_resseqs = detect_paratope_residues(
            structure, nb_chain, ag_chains, self._distance_cutoff
        )

        return pdb_map, paratope_resseqs

    def mask(
        self,
        candidate: NanobodyCandidate,
        strategy: MaskStrategy,
    ) -> MaskResult:
        """Apply a masking strategy to a candidate.

        For structure-based strategies (``PARATOPE``, ``UNANCHORED_CLASH``),
        loads the complex PDB and runs paratope detection.  Falls back to
        ``CDR_FOCUSED`` if ``PARATOPE`` is requested but no structure is
        available.

        Args:
            candidate: Must have ``raw_sequence`` populated.  For
                structure-based strategies, must also have
                ``complex_pdb_path``, ``nanobody_chain_id``, and
                ``antigen_chain_ids``.
            strategy: Which masking strategy to apply.

        Returns:
            MaskResult with the set of positions to mask and
            IgLM-ready contiguous spans.

        Raises:
            ValueError: If ``UNANCHORED_CLASH`` is requested without
                structural data (no fallback available).
        """
        kabat_mapping = self._get_kabat_mapping(candidate.raw_sequence)

        # Sequence-only strategies — no structure needed.
        if strategy == MaskStrategy.CDR_FOCUSED:
            return cdr_focused_mask(kabat_mapping)

        if strategy == MaskStrategy.FR2_REVERSION:
            return fr2_reversion_mask(kabat_mapping)

        # Structure-based strategies — need complex PDB.
        try:
            pdb_map, paratope_resseqs = self._load_structure_data(
                candidate, kabat_mapping
            )
        except ValueError as exc:
            if strategy == MaskStrategy.PARATOPE:
                logger.warning(
                    "Candidate %s: %s — falling back to CDR_FOCUSED",
                    candidate.candidate_id,
                    exc,
                )
                result = cdr_focused_mask(kabat_mapping)
                result.metadata["fallback_reason"] = str(exc)
                return result
            # UNANCHORED_CLASH has no fallback.
            raise

        if strategy == MaskStrategy.PARATOPE:
            return paratope_mask(kabat_mapping, pdb_map, paratope_resseqs)

        if strategy == MaskStrategy.UNANCHORED_CLASH:
            return unanchored_clash_mask(
                kabat_mapping, pdb_map, paratope_resseqs
            )

        raise ValueError(f"Unknown strategy: {strategy}")
