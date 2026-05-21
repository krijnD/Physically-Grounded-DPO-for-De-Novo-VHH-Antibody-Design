"""Direct biophysics scoring of a provided PDB via ``theraprofnano``.

Bypasses TNP's NanoBodyBuilder2 (NB2) re-folding step. Used by the AAPR
pipeline so that PSH / PPC / PNC / CDR3-Compactness are computed on the
DiffAb-generated structure rather than on a re-fold of the sequence —
which is what the TNP CLI would do.

The numerical values are produced by the same Python functions TNP's CLI
calls after folding, so they match TNP's clinical-threshold semantics
(Gordon et al. 2026). The geometry being scored is the only difference:
this module scores the input PDB directly, not an NB2 re-fold of the
sequence.

Public API:
    score_pdb(...) -> TNPResult
"""

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .pdb_utils import extract_vhh_monomer, renumber_to_imgt

logger = logging.getLogger(__name__)


@dataclass
class TNPResult:
    """Parsed biophysics metrics for a single nanobody structure.

    Shape matches the dataclass the legacy ``tnp_runner`` returned so
    pipeline glue (e.g. ``src/pipeline.py``) does not change.
    """

    psh: float
    ppc: float
    pnc: float
    compactness: float
    cdr_length: int
    cdr3_length: int
    pdb_path: Optional[str] = None
    flags: dict[str, str] | None = None


def _import_theraprofnano():
    """Lazy-import theraprofnano so unrelated tests don't pull DSSP/PSA."""
    from theraprofnano.CDR_Profiler.CDR_Assigner import main as region_and_aa_dicts
    from theraprofnano.CDR_Profiler.CDR3_Conf_Assigner import main_compactness
    from theraprofnano.Hydrophobicity_and_Charge_Profiler.Hydrophobicity_and_Charge_Assigner import (
        CreateAnnotation,
    )
    return region_and_aa_dicts, main_compactness, CreateAnnotation


def score_pdb(
    complex_pdb_path: str | Path,
    nanobody_chain_id: str,
    candidate_id: str,
    sequence: str,
    output_dir: str | Path,
    numbering_scheme: str = "imgt",
    hydrophobicity_index: int = 0,
    pH: float = 7.4,
) -> TNPResult:
    """Score a complex PDB's VHH chain with TNP's biophysics metrics.

    Pipeline:
      1. Extract the VHH chain from ``complex_pdb_path`` into a clean
         monomer PDB (no hydrogens, no HETATM, chain id renamed to "H"
         because TNP's CreateAnnotation hardcodes "H").
      2. Renumber the monomer's residues to the IMGT scheme via
         ``abnumber``. TNP's metric functions look up residues by IMGT
         number; the source crystal numbering DiffAb inherits does not
         align with IMGT and would leave ``parse_nb``'s CDR3 anchor
         list empty.
      3. Compute CDR lengths (IMGT) from the *sequence* via TNP's
         ``CDR_Assigner.main`` — same source TNP CLI uses.
      4. Compute rho via TNP's ``main_compactness`` on the renumbered
         monomer PDB. compactness = imgt_cdr3_length / rho  (TNP's
         corrected formula, per "UPDATED (corrected) 23.08.25" in
         bin/TNP).
      5. Compute PSH / PPC / PNC via TNP's ``CreateAnnotation`` on the
         same renumbered monomer.
      6. Return the metrics packaged as a ``TNPResult``. ``pdb_path``
         points at the renumbered monomer so downstream judges
         (Biology) can re-use the same coordinates.

    Args:
        complex_pdb_path: VHH+antigen complex PDB (DiffAb output or
            crystal). Only the nanobody chain is scored.
        nanobody_chain_id: Chain id of the VHH in the source PDB.
        candidate_id: Used to name the extracted monomer file.
        sequence: VHH amino-acid sequence. Used to derive IMGT CDR
            lengths the same way TNP's CLI does.
        output_dir: Where the extracted monomer PDB is written.
            ``output_dir / "{candidate_id}.pdb"``.
        numbering_scheme: "imgt" or "chothia". Defaults to "imgt"
            (matches TNP CLI).
        hydrophobicity_index: 0=Kyte-Doolittle, 1=Wimley-White,
            2=Hessa, 3=Eisenberg-McLachlan, 4=Black-Mould.
            0 is TNP CLI's default.
        pH: pH for charge calculations. TNP CLI uses 7.4.

    Returns:
        TNPResult with all six metrics + the extracted monomer path.

    Raises:
        Any exception from theraprofnano is propagated. Callers in the
        pipeline are expected to catch and translate to a
        ``skipped_no_tnp`` verdict.
    """
    region_and_aa_dicts, main_compactness, CreateAnnotation = _import_theraprofnano()

    output_dir = Path(output_dir)
    raw_monomer_path = output_dir / f"{candidate_id}.raw.pdb"
    monomer_path = output_dir / f"{candidate_id}.pdb"
    extract_vhh_monomer(
        complex_pdb_path=complex_pdb_path,
        source_chain_id=nanobody_chain_id,
        output_path=raw_monomer_path,
        target_chain_id="H",
    )
    renumber_to_imgt(raw_monomer_path, monomer_path)

    # ── CDR lengths (sequence-derived, IMGT) ──
    # region_and_aa_dicts writes intermediate files to a temp dir we
    # discard. Signature: main(name, sequence, chain, output_dest, ncpu, verbose).
    with tempfile.TemporaryDirectory() as cdr_tmp:
        _, length_dict = region_and_aa_dicts(
            candidate_id, sequence, "H", cdr_tmp, ncpu=1, verbose=False,
        )

    imgt_lengths = length_dict[numbering_scheme]
    cdr3_length = int(imgt_lengths["cdrh3"])
    cdr_length_total = int(sum(imgt_lengths.values()))

    # ── Compactness ──
    rho = main_compactness(str(monomer_path), numbering_scheme, verbose=False)
    if rho is None or rho <= 0:
        raise RuntimeError(
            f"main_compactness returned {rho!r} for {monomer_path} — "
            f"anchor residues likely missing in the structure."
        )
    compactness = cdr3_length / rho

    # ── Surface metrics (PSH / PPC / PNC) ──
    # CreateAnnotation returns {hydrophobicity_index: {Patch_*_CDR: float}}.
    # The `chains` argument is accepted but ignored by the function body
    # (which hardcodes 'H'); we pass "IG" to match TNP CLI defaults.
    annotation = CreateAnnotation(
        hydrophobicity_index, pH, str(monomer_path), "IG",
        numbering_scheme, verbose=False,
    )
    metrics = annotation[hydrophobicity_index]

    return TNPResult(
        psh=float(metrics["Patch_Hydrophob_CDR"]),
        ppc=float(metrics["Patch_Pos_Charge_CDR"]),
        pnc=float(metrics["Patch_Neg_Charge_CDR"]),
        compactness=float(compactness),
        cdr_length=cdr_length_total,
        cdr3_length=cdr3_length,
        pdb_path=str(monomer_path),
    )
