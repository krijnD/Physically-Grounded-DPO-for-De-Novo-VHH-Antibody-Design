"""PDB utilities for biophysics scoring on DiffAb-generated structures.

Two helpers compose into the preprocessing that ``tnp_direct.score_pdb``
runs before calling ``theraprofnano``:

  * ``extract_vhh_monomer`` — single chain, ATOM only, no hydrogens,
    chain relabeled to "H".
  * ``renumber_to_imgt`` — rewrite ATOM residue numbers / insertion
    codes to the IMGT scheme via ``abnumber`` (drops residues outside
    the FV region).

Three preprocessing details matter for downstream TNP compatibility:

1. ``CreateAnnotation`` in TNP's Hydrophobicity_and_Charge_Assigner
   hardcodes the chain label to "H" (``nb_structure = {'H': PDBchain(...,'H')}``),
   so the extracted monomer must use chain id "H".
2. TNP's CLI normally strips hydrogens after NanoBodyBuilder2 folding via
   its ``pdb_remove_hydrogens.py`` script. The metric functions therefore
   assume a hydrogen-free structure with no HETATM, and a single altloc
   per atom — we strip those here.
3. ``parse_nb`` (compactness) and ``is_CDR`` (charge/hydrophobicity)
   both look up residues by their PDB residue number under the IMGT
   scheme. DiffAb-generated PDBs carry the source crystal's original
   numbering, which doesn't line up with IMGT — so we renumber.
"""

import logging
import warnings
from pathlib import Path

from Bio import BiopythonWarning
from Bio.PDB import PDBParser

logger = logging.getLogger(__name__)

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def extract_vhh_monomer(
    complex_pdb_path: str | Path,
    source_chain_id: str,
    output_path: str | Path,
    target_chain_id: str = "H",
) -> Path:
    """Write a single-chain VHH PDB extracted from a complex PDB.

    Implemented as a column-precise text rewrite — Biopython's
    PDBIO+Select doesn't play well with in-place chain renames, so we
    avoid the tree entirely. Reads the source PDB line-by-line, keeps
    only ATOM records whose chain id matches ``source_chain_id``,
    drops hydrogens / HETATM / non-A altlocs, and rewrites the chain id
    column to ``target_chain_id`` (defaults to "H" because TNP's
    ``CreateAnnotation`` hardcodes "H").

    Args:
        complex_pdb_path: Source PDB (typically a DiffAb-generated
            VHH+antigen complex, or a crystal complex).
        source_chain_id: Chain id of the VHH in the source PDB.
        output_path: Destination PDB path. Parent dir is created.
        target_chain_id: Chain id to assign in the output PDB.

    Returns:
        ``output_path`` as a Path.

    Raises:
        ValueError: if no ATOM records for ``source_chain_id`` are found.
    """
    complex_pdb_path = Path(complex_pdb_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(target_chain_id) != 1:
        raise ValueError(f"target_chain_id must be a single char, got {target_chain_id!r}")

    n_written = 0
    with open(complex_pdb_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            if not line.startswith("ATOM"):
                # Drop everything that isn't ATOM: HEADER/TITLE we don't
                # need, HETATM (waters/ligands) we never want, TER lines
                # that reference dropped residues, MODEL/ENDMDL boundaries
                # that confuse a single-chain output.
                continue
            if len(line) < 22:
                continue
            chain_col = line[21]
            if chain_col != source_chain_id:
                continue
            # Drop hydrogens. PDB element column is cols 77-78 (0-idx 76:78).
            # If absent or whitespace, infer from atom-name col 13-14
            # (0-idx 12:14): hydrogens start with " H" or "1H"/"2H".
            element = line[76:78].strip() if len(line) >= 78 else ""
            atom_name = line[12:16].strip()
            if element == "H" or (not element and atom_name.startswith("H")):
                continue
            # Drop non-blank, non-A altlocs (col 17, 0-idx 16).
            altloc = line[16] if len(line) > 16 else " "
            if altloc not in (" ", "A"):
                continue
            # Rewrite chain id column to target.
            fout.write(line[:21] + target_chain_id + line[22:])
            n_written += 1
        fout.write("END\n")

    if n_written == 0:
        raise ValueError(
            f"No ATOM records for chain {source_chain_id!r} in {complex_pdb_path}. "
            f"Check that the chain id is correct."
        )
    return output_path


def renumber_to_imgt(monomer_pdb: str | Path, output_pdb: str | Path) -> Path:
    """Rewrite ``monomer_pdb`` with IMGT residue numbers via ``abnumber``.

    Residues outside the antibody FV region (signal peptide, C-terminal
    tags) are dropped because TNP's metric functions iterate over all
    chain residues and would otherwise misclassify them as framework.

    The output keeps only ATOM records for residues that mapped to an
    IMGT position; all other lines (HEADER, TER, END, etc.) are passed
    through unchanged so the file remains a valid PDB.
    """
    from abnumber import Chain as AbChain
    from abnumber.exceptions import ChainParseError

    monomer_pdb = Path(monomer_pdb)
    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Extract ordered AA sequence from the monomer's ATOM records ──
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BiopythonWarning)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("monomer", str(monomer_pdb))

    chain = next(iter(next(structure.get_models()).get_chains()))
    ordered_residues = [
        res for res in chain
        if res.id[0] == " " and res.get_resname() in _THREE_TO_ONE
    ]
    if not ordered_residues:
        raise ValueError(f"No standard residues found in {monomer_pdb}")

    seq = "".join(_THREE_TO_ONE[res.get_resname()] for res in ordered_residues)

    # ── 2. Get IMGT numbering for the sequence ──
    try:
        ab = AbChain(seq, scheme="imgt", assign_germline=False)
    except ChainParseError as e:
        raise ValueError(f"abnumber failed to parse {monomer_pdb}: {e}") from e

    fv = ab.seq
    fv_start = seq.find(fv)
    if fv_start < 0:
        raise ValueError(
            f"abnumber FV region ({fv[:30]}…) not found in input sequence "
            f"({seq[:30]}…) for {monomer_pdb}"
        )

    positions = list(ab.positions.keys())
    if len(positions) != len(fv):
        raise ValueError(
            f"abnumber returned {len(positions)} positions for an "
            f"FV of length {len(fv)} in {monomer_pdb}"
        )

    # ── 3. Build (orig_resseq, orig_icode) → (imgt_num, imgt_icode) map ──
    remap: dict[tuple[int, str], tuple[int, str]] = {}
    for i, pos in enumerate(positions):
        res = ordered_residues[fv_start + i]
        orig_resseq = res.id[1]
        orig_icode = res.id[2] if res.id[2] != " " else " "
        new_resseq = pos.number
        new_icode = pos.letter if pos.letter else " "
        remap[(orig_resseq, orig_icode)] = (new_resseq, new_icode)

    # ── 4. Rewrite the PDB file, column-precise on ATOM records ──
    # PDB ATOM/HETATM record layout (1-indexed): cols 23-26 = resseq (i4),
    # col 27 = icode (a1). In Python 0-indexed slicing that's [22:26] and
    # [26:27].
    with open(monomer_pdb) as fin, open(output_pdb, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    resseq = int(line[22:26])
                except ValueError:
                    continue
                icode = line[26] if len(line) > 26 else " "
                new = remap.get((resseq, icode))
                if new is None:
                    continue  # outside FV — drop
                new_resseq, new_icode = new
                fout.write(
                    line[:22]
                    + f"{new_resseq:>4d}"
                    + new_icode
                    + line[27:]
                )
            elif line.startswith("TER"):
                # Drop TER lines — they reference residues that may have
                # been dropped or renumbered, and TNP doesn't need them.
                continue
            else:
                fout.write(line)

    return output_pdb


# Expected number of heavy atoms per standard amino acid type. A residue
# with fewer atoms than its entry here is missing side-chain coordinates
# and needs packing. Includes CB for everything except GLY.
_EXPECTED_HEAVY_ATOMS: dict[str, int] = {
    "ALA": 5,  "ARG": 11, "ASN": 8,  "ASP": 8,  "CYS": 6,
    "GLN": 9,  "GLU": 9,  "GLY": 4,  "HIS": 10, "ILE": 8,
    "LEU": 8,  "LYS": 9,  "MET": 8,  "PHE": 11, "PRO": 7,
    "SER": 6,  "THR": 7,  "TRP": 14, "TYR": 12, "VAL": 7,
}


def _find_incomplete_residues(
    pdb_path: Path,
    chain_id: str | None = None,
) -> set[tuple[str, int, str, str]]:
    """Return ``{(chain, resseq, icode, resname)}`` for residues with
    fewer heavy atoms than ``_EXPECTED_HEAVY_ATOMS`` says they should.

    If ``chain_id`` is given, only consider residues in that chain.
    Hydrogens and non-standard residues are ignored.
    """
    counts: dict[tuple[str, int, str], tuple[str, int]] = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain = line[21]
            if chain_id is not None and chain != chain_id:
                continue
            element = line[76:78].strip() if len(line) >= 78 else ""
            atom_name = line[12:16].strip()
            if element == "H" or (not element and atom_name.startswith("H")):
                continue
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            icode = line[26] if len(line) > 26 else " "
            resname = line[17:20].strip()
            key = (chain, resseq, icode)
            prior_name, prior_n = counts.get(key, (resname, 0))
            counts[key] = (prior_name, prior_n + 1)

    incomplete: set[tuple[str, int, str, str]] = set()
    for (chain, resseq, icode), (resname, n) in counts.items():
        expected = _EXPECTED_HEAVY_ATOMS.get(resname)
        if expected is None:
            continue
        if n < expected:
            incomplete.add((chain, resseq, icode, resname))
    return incomplete


def pack_missing_sidechains(
    input_pdb: str | Path,
    output_pdb: str | Path,
    chain_id: str | None = None,
    shell_radius: float = 6.0,
) -> Path:
    """Place side chains for residues whose heavy-atom record is incomplete.

    DiffAb writes only backbone (N, CA, C, O) atoms for the residues it
    regenerated. This function fills those in using PyRosetta's standard
    ``PackRotamersMover`` over the Dunbrack 2010 rotamer library — a
    deterministic post-process that places each missing side chain at
    its most plausible orientation given the surrounding backbone.

    Scope is kept deliberately tight:

      * Backbone atoms are **never moved** (no FastRelax, no minimization).
      * Only residues identified as incomplete are flagged for repack.
      * A spherical shell of ``shell_radius`` Å around the incomplete
        residues is included for local context (so the placed rotamers
        respect their immediate neighbors), but those shell residues
        themselves keep their original rotamers — they're held fixed
        and only used for clash evaluation.
      * If ``chain_id`` is given, repacking is also restricted to that
        chain, so antigen rotamers are never touched even if a CDR
        residue happens to sit within ``shell_radius`` of one.

    If no incomplete residues are found, the function simply copies
    ``input_pdb`` to ``output_pdb`` and returns.

    Args:
        input_pdb: Source PDB (e.g. DiffAb's sample output).
        output_pdb: Destination PDB.
        chain_id: If set, restrict repack to this chain. For AAPR
            samples, pass the VHH chain id so the antigen is preserved.
        shell_radius: Distance (Å) around incomplete residues from
            which neighbors are loaded into the packer as fixed
            context. 6 Å roughly captures the first contact shell.

    Returns:
        ``output_pdb`` as a Path.
    """
    input_pdb = Path(input_pdb)
    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    incomplete = _find_incomplete_residues(input_pdb, chain_id=chain_id)
    if not incomplete:
        import shutil
        shutil.copy(input_pdb, output_pdb)
        logger.debug("No incomplete residues in %s — copied as-is.", input_pdb)
        return output_pdb

    from src.physics_judge.rosetta_scorer import _ensure_init
    _ensure_init()
    import pyrosetta
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.pack.task.operation import (
        InitializeFromCommandline,
        OperateOnResidueSubset,
        PreventRepackingRLT,
        RestrictToRepacking,
    )
    from pyrosetta.rosetta.core.select.residue_selector import (
        ChainSelector,
        NeighborhoodResidueSelector,
        NotResidueSelector,
        ResidueIndexSelector,
    )
    from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover

    pose = pyrosetta.pose_from_pdb(str(input_pdb))
    pdb_info = pose.pdb_info()

    # Map (chain, resseq, icode) → pose index for the incomplete residues.
    incomplete_lookup = {(c, r, i): rn for (c, r, i, rn) in incomplete}
    pose_indices: list[int] = []
    for i in range(1, pose.total_residue() + 1):
        chain = pdb_info.chain(i)
        resseq = pdb_info.number(i)
        icode = pdb_info.icode(i) or " "
        if (chain, resseq, icode) in incomplete_lookup:
            pose_indices.append(i)

    if not pose_indices:
        logger.warning(
            "Found %d incomplete residues in %s but could not map any "
            "to pose indices — copying input as-is.",
            len(incomplete), input_pdb,
        )
        import shutil
        shutil.copy(input_pdb, output_pdb)
        return output_pdb

    indices_str = ",".join(str(i) for i in pose_indices)
    focus_selector = ResidueIndexSelector(indices_str)
    # The neighborhood selector by default returns focus + neighbors. We
    # want focus packable, neighbors packable too (so rotamers relax
    # locally), and everything else frozen. ``include_focus_in_subset``
    # defaults True in newer PyRosetta — keep it that way.
    repack_selector = NeighborhoodResidueSelector(focus_selector, shell_radius)
    if chain_id is not None:
        # AND with chain restriction so the shell can't pull in antigen.
        from pyrosetta.rosetta.core.select.residue_selector import AndResidueSelector
        repack_selector = AndResidueSelector(repack_selector, ChainSelector(chain_id))

    no_repack_selector = NotResidueSelector(repack_selector)

    sfxn = pyrosetta.create_score_function("ref2015")
    tf = TaskFactory()
    tf.push_back(InitializeFromCommandline())
    tf.push_back(RestrictToRepacking())  # never design — only repack
    tf.push_back(OperateOnResidueSubset(PreventRepackingRLT(), no_repack_selector))
    task = tf.create_task_and_apply_taskoperations(pose)

    pre = sfxn(pose)
    PackRotamersMover(sfxn, task).apply(pose)
    post = sfxn(pose)

    logger.info(
        "pack_missing_sidechains(%s): %d incomplete residues in chain(s) %s; "
        "ref2015 %.1f → %.1f REU (Δ %+.1f).",
        input_pdb.name, len(pose_indices),
        chain_id or "any", pre, post, post - pre,
    )

    pose.dump_pdb(str(output_pdb))
    return output_pdb
