"""Full Kabat numbering and PDB-to-sequence alignment for masking.

Bridges three coordinate systems:
- Sequence index (0-based position in raw_sequence) — what IgLM needs
- Kabat position (e.g. "26", "52A") — what abnumber provides
- PDB residue ID (resseq integer) — what Biopython structures use

Uses ANARCI (via abnumber) for Kabat alignment, following the same
pattern as biology_judge/sequence_filter.py.

References:
- Kabat numbering: Wu & Kabat (1970)
- ANARCI HMMs: Dunbar & Deane (2016)
- Anchor residue protection: GeoGAD (Tan et al., 2024), AbFlex (Ruffolo et al., 2024)
- FR2 hallmark positions: Mitchell & Colwell (2018)
"""

import logging
from dataclasses import dataclass, field

from abnumber import Chain
from Bio.PDB.Polypeptide import PPBuilder, three_to_one
from Bio.PDB.Structure import Structure

from src.common.config import Config

logger = logging.getLogger(__name__)

CDR_REGION_NAMES = ("CDR1", "CDR2", "CDR3")


@dataclass
class KabatMapping:
    """Complete Kabat numbering for a VHH sequence.

    Attributes:
        kabat_to_seq_idx: Maps Kabat position string (e.g. "26", "52A")
                          to 0-based index in raw_sequence.
        seq_idx_to_kabat: Inverse mapping.
        cdr_indices: 0-based sequence indices that fall within CDR regions.
        anchor_indices: 0-based indices for anchor residues flanking
                        each CDR boundary.
        fr2_hallmark_indices: 0-based indices for FR2 hallmark positions
                              (Kabat 37, 44, 45, 47) only.
        fr2_region_indices: 0-based indices for the full FR2 region
                            (from chain.regions["FR2"]).
        chain: The abnumber Chain object (for downstream region access).
    """

    kabat_to_seq_idx: dict[str, int]
    seq_idx_to_kabat: dict[int, str]
    cdr_indices: set[int]
    anchor_indices: set[int]
    fr2_hallmark_indices: set[int]
    fr2_region_indices: set[int]
    chain: Chain


@dataclass
class PDBResidueMap:
    """Maps PDB residue sequence numbers to 0-based sequence indices.

    Attributes:
        pdb_resid_to_seq_idx: Maps PDB resseq integer to the
                              corresponding 0-based index in raw_sequence.
    """

    pdb_resid_to_seq_idx: dict[int, int] = field(default_factory=dict)


def build_kabat_mapping(
    sequence: str,
    anchor_flank: int = Config.ANCHOR_FLANK_SIZE,
    fr2_positions: list[str] = Config.FR2_HALLMARK_POSITIONS,
) -> KabatMapping:
    """Build the complete Kabat mapping for a VHH sequence.

    Uses abnumber.Chain to perform the ANARCI HMM alignment, then
    identifies CDR regions, anchor residues, and FR2 hallmarks.

    Anchor residues are the ``anchor_flank`` positions immediately
    before the first CDR residue and immediately after the last CDR
    residue in each loop, as ordered by Kabat numbering.  This handles
    insertion codes (52A, 52B, etc.) correctly.

    Args:
        sequence: Raw amino acid sequence string.
        anchor_flank: Number of flanking residues to protect on each
                      side of each CDR boundary.
        fr2_positions: Kabat position strings for FR2 hallmark residues.

    Returns:
        A fully populated KabatMapping.

    Raises:
        abnumber.exceptions.ChainParseError: If ANARCI cannot parse
            the sequence as an immunoglobulin.
    """
    chain = Chain(sequence, scheme="kabat", assign_germline=False)

    # Build the full position-to-index mapping by iterating in order.
    # abnumber.Chain.positions is an OrderedDict[Position, str].
    kabat_to_seq_idx: dict[str, int] = {}
    seq_idx_to_kabat: dict[int, str] = {}
    all_positions_ordered: list[str] = []

    seq_idx = 0
    for position, _aa in chain:
        pos_str = str(position)
        kabat_to_seq_idx[pos_str] = seq_idx
        seq_idx_to_kabat[seq_idx] = pos_str
        all_positions_ordered.append(pos_str)
        seq_idx += 1

    # Identify CDR indices from chain.regions.
    cdr_indices: set[int] = set()
    cdr_position_sets: dict[str, set[str]] = {}

    for region_name, region_positions in chain.regions.items():
        if region_name in CDR_REGION_NAMES:
            pos_strings = {str(p) for p in region_positions}
            cdr_position_sets[region_name] = pos_strings
            for pos_str in pos_strings:
                if pos_str in kabat_to_seq_idx:
                    cdr_indices.add(kabat_to_seq_idx[pos_str])

    # Identify anchor residues: the N positions immediately before
    # and after each CDR region in the ordered position list.
    anchor_indices: set[int] = set()

    for region_name in CDR_REGION_NAMES:
        if region_name not in cdr_position_sets:
            continue
        cdr_pos_set = cdr_position_sets[region_name]

        # Find the indices in all_positions_ordered that belong to this CDR.
        cdr_ordered_indices = [
            i
            for i, pos in enumerate(all_positions_ordered)
            if pos in cdr_pos_set
        ]
        if not cdr_ordered_indices:
            continue

        first_cdr_idx = cdr_ordered_indices[0]
        last_cdr_idx = cdr_ordered_indices[-1]

        # N-terminal anchors: positions immediately before the CDR.
        for offset in range(1, anchor_flank + 1):
            anchor_pos_idx = first_cdr_idx - offset
            if anchor_pos_idx >= 0:
                anchor_kabat = all_positions_ordered[anchor_pos_idx]
                if anchor_kabat in kabat_to_seq_idx:
                    anchor_indices.add(kabat_to_seq_idx[anchor_kabat])

        # C-terminal anchors: positions immediately after the CDR.
        for offset in range(1, anchor_flank + 1):
            anchor_pos_idx = last_cdr_idx + offset
            if anchor_pos_idx < len(all_positions_ordered):
                anchor_kabat = all_positions_ordered[anchor_pos_idx]
                if anchor_kabat in kabat_to_seq_idx:
                    anchor_indices.add(kabat_to_seq_idx[anchor_kabat])

    # FR2 hallmark positions.
    fr2_hallmark_indices: set[int] = set()
    for pos_str in fr2_positions:
        if pos_str in kabat_to_seq_idx:
            fr2_hallmark_indices.add(kabat_to_seq_idx[pos_str])
        else:
            logger.debug(
                "FR2 hallmark position %s not found in Kabat alignment", pos_str
            )

    # Full FR2 region (from chain.regions).
    fr2_region_indices: set[int] = set()
    fr2_region = chain.regions.get("FR2")
    if fr2_region is not None:
        for position in fr2_region:
            pos_str = str(position)
            if pos_str in kabat_to_seq_idx:
                fr2_region_indices.add(kabat_to_seq_idx[pos_str])

    logger.info(
        "Kabat mapping: %d positions, CDR=%d, anchors=%d, FR2_hallmarks=%d, FR2_region=%d",
        len(kabat_to_seq_idx),
        len(cdr_indices),
        len(anchor_indices),
        len(fr2_hallmark_indices),
        len(fr2_region_indices),
    )

    return KabatMapping(
        kabat_to_seq_idx=kabat_to_seq_idx,
        seq_idx_to_kabat=seq_idx_to_kabat,
        cdr_indices=cdr_indices,
        anchor_indices=anchor_indices,
        fr2_hallmark_indices=fr2_hallmark_indices,
        fr2_region_indices=fr2_region_indices,
        chain=chain,
    )


def build_pdb_to_sequence_map(
    structure: Structure,
    chain_id: str,
    raw_sequence: str,
) -> PDBResidueMap:
    """Map PDB residue IDs to positions in the raw amino acid sequence.

    Extracts the chain's sequence from the PDB via PPBuilder, aligns
    it to ``raw_sequence``, and maps each PDB residue's resseq to a
    0-based index in ``raw_sequence``.

    Follows the pattern from ``sabdab_loader.py:extract_chain_sequence``.

    Args:
        structure: Biopython Structure (parsed complex PDB).
        chain_id: Chain letter of the nanobody (e.g. ``"H"``).
        raw_sequence: The ground-truth amino acid sequence.

    Returns:
        A PDBResidueMap with the resseq-to-index mapping.  Missing
        residues (gaps in PDB density) will simply be absent.
    """
    model = structure[0]
    try:
        chain = model[chain_id]
    except KeyError:
        logger.warning("Chain %s not found in structure", chain_id)
        return PDBResidueMap()

    # Extract standard residues and their resseq numbers in order.
    residues_with_ids: list[tuple[int, str]] = []
    for residue in chain.get_residues():
        # Skip hetero atoms (water, ligands, etc.).
        hetflag = residue.id[0]
        if hetflag.strip():
            continue
        resname = residue.get_resname()
        try:
            one_letter = three_to_one(resname)
        except KeyError:
            continue
        resseq = residue.id[1]
        residues_with_ids.append((resseq, one_letter))

    if not residues_with_ids:
        logger.warning("No standard residues found for chain %s", chain_id)
        return PDBResidueMap()

    # Build the PDB chain sequence for alignment.
    pdb_sequence = "".join(aa for _, aa in residues_with_ids)

    # Find the PDB sequence within the raw sequence.
    # For ground-truth complexes the PDB chain should be a substring.
    offset = raw_sequence.find(pdb_sequence)

    if offset == -1:
        # Try allowing for minor mismatches at the termini by checking
        # if the core (middle 80%) aligns.
        core_start = len(pdb_sequence) // 10
        core_end = len(pdb_sequence) - core_start
        core = pdb_sequence[core_start:core_end]
        core_offset = raw_sequence.find(core)

        if core_offset != -1:
            offset = core_offset - core_start
            logger.debug(
                "PDB chain %s: core alignment at offset %d (terminal mismatch)",
                chain_id,
                offset,
            )
        else:
            logger.warning(
                "PDB chain %s sequence does not align with raw_sequence "
                "(PDB: %.30s..., raw: %.30s...)",
                chain_id,
                pdb_sequence,
                raw_sequence,
            )
            return PDBResidueMap()

    # Build the mapping.
    pdb_resid_to_seq_idx: dict[int, int] = {}
    for pdb_idx, (resseq, _aa) in enumerate(residues_with_ids):
        seq_idx = offset + pdb_idx
        if 0 <= seq_idx < len(raw_sequence):
            pdb_resid_to_seq_idx[resseq] = seq_idx

    logger.info(
        "PDB-to-sequence map: chain %s, %d residues mapped (offset=%d)",
        chain_id,
        len(pdb_resid_to_seq_idx),
        offset,
    )

    return PDBResidueMap(pdb_resid_to_seq_idx=pdb_resid_to_seq_idx)
