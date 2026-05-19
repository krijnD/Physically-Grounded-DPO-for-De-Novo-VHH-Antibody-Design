"""SAbDab nanobody dataset loader.

Reads the SAbDab nanobody summary TSV and the corresponding PDB files
downloaded from RCSB, producing pipeline-ready input dicts that include
the nanobody sequence, complex PDB path, and chain identifiers.

Usage:
    entries = load_sabdab_entries(
        summary_tsv="sabdab_nano_summary.tsv",
        pdb_dir="filtered_vhh_pdbs",
    )
    # entries is a list of dicts ready for run_pipeline() or direct judge use
"""

import logging
from pathlib import Path

import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder

logger = logging.getLogger(__name__)

# Three-letter → one-letter amino acid mapping (standard 20)
_THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def extract_chain_sequence(pdb_path: str, chain_id: str) -> str | None:
    """Extract the amino acid sequence of a specific chain from a PDB file.

    Uses Biopython's PPBuilder to handle chain breaks and non-standard
    residues gracefully.

    Args:
        pdb_path: Path to the PDB file.
        chain_id: Chain letter to extract (e.g. ``"H"``).

    Returns:
        One-letter amino acid sequence string, or ``None`` if the chain
        is not found or has no standard residues.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("tmp", pdb_path)
    model = structure[0]

    try:
        chain = model[chain_id]
    except KeyError:
        logger.warning("Chain %s not found in %s", chain_id, pdb_path)
        return None

    ppb = PPBuilder()
    polypeptides = ppb.build_peptides(chain)

    if not polypeptides:
        logger.warning(
            "No polypeptides found for chain %s in %s", chain_id, pdb_path
        )
        return None

    # Concatenate all polypeptide fragments (handles chain breaks)
    sequence = "".join(str(pp.get_sequence()) for pp in polypeptides)

    if not sequence:
        return None

    return sequence


def load_sabdab_entries(
    summary_tsv: str,
    pdb_dir: str,
    date_cutoff: str = "2023-01-01",
    resolution_max: float = 2.5,
) -> list[dict]:
    """Load and filter SAbDab entries into pipeline-ready input dicts.

    Applies the same filters as ``data scripts/fetch_nano.py``:
      - Date >= date_cutoff (post-IgLM training data)
      - Resolution <= resolution_max (required for reliable Rosetta scoring)

    For each passing entry, extracts the nanobody sequence from the
    PDB file's H-chain.

    Args:
        summary_tsv: Path to ``sabdab_nano_summary.tsv``.
        pdb_dir: Directory containing downloaded PDB files (from RCSB).
        date_cutoff: Minimum release date (ISO format).
        resolution_max: Maximum X-ray resolution in Angstroms.

    Returns:
        List of dicts with keys:
          - ``candidate_id``: PDB ID
          - ``raw_sequence``: nanobody amino acid sequence
          - ``pdb_filepath``: path to the complex PDB (used by Biology Judge)
          - ``complex_pdb_path``: same path (used by Physics Judge)
          - ``nanobody_chain_id``: H-chain letter from SAbDab
          - ``antigen_chain_ids``: antigen chain letter(s) from SAbDab
    """
    pdb_dir = Path(pdb_dir)

    # Read and filter the TSV
    df = pd.read_csv(summary_tsv, sep="\t")
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y", errors="coerce")
    df["resolution"] = pd.to_numeric(df["resolution"], errors="coerce")

    filtered = df[
        (df["date"] >= date_cutoff)
        & (df["resolution"] <= resolution_max)
    ].copy()

    logger.info(
        "SAbDab: %d/%d entries pass filters (date >= %s, resolution <= %.1f)",
        len(filtered),
        len(df),
        date_cutoff,
        resolution_max,
    )

    # Deduplicate by PDB ID (multiple rows per PDB if multiple antigen chains)
    filtered = filtered.drop_duplicates(subset="pdb", keep="first")

    entries: list[dict] = []
    for _, row in filtered.iterrows():
        pdb_id = str(row["pdb"]).lower()
        pdb_path = pdb_dir / f"{pdb_id}.pdb"

        if not pdb_path.exists():
            logger.debug("PDB file not found for %s, skipping.", pdb_id)
            continue

        hchain = str(row.get("Hchain", "H")).strip()
        antigen_chain = str(row.get("antigen_chain", "")).strip()

        # Handle missing/nan antigen chain
        if antigen_chain in ("", "nan", "None"):
            antigen_chain = None

        # Extract nanobody sequence from the H-chain
        sequence = extract_chain_sequence(str(pdb_path), hchain)
        if sequence is None:
            logger.warning(
                "Could not extract sequence for %s chain %s, skipping.",
                pdb_id,
                hchain,
            )
            continue

        entries.append(
            {
                "candidate_id": pdb_id,
                "raw_sequence": sequence,
                "pdb_filepath": str(pdb_path),
                "complex_pdb_path": str(pdb_path),
                "nanobody_chain_id": hchain,
                "antigen_chain_ids": antigen_chain,
            }
        )

    logger.info("Loaded %d SAbDab entries with valid sequences.", len(entries))
    return entries


def load_andd_entries(csv_path: str, pdb_dir: str) -> list[dict]:
    """Load pipeline-ready entries from the ANDD CSV metadata file.

    The ANDD CSV (``ANDD_VHH_with_structure.csv``) contains one row per
    mutation variant per PDB structure.  This function deduplicates to one
    entry per PDB file, reads chain IDs and sequences from the CSV, and
    matches them to the PDB files in ``pdb_dir``.

    Chain IDs come from ``H_Chain Auth Asym ID`` and ``Ag_Auth Asym ID``
    (first value when comma-separated).  The nanobody sequence is taken
    directly from ``Ab/Nano H_Chain AA`` — no PDB parsing needed for the
    sequence.  If a chain ID from the CSV is not found in the actual PDB
    file (e.g. the file has been renumbered), the function logs a warning
    and skips that entry.

    Args:
        csv_path: Path to ``ANDD_VHH_with_structure.csv``.
        pdb_dir: Directory containing the PDB files (filenames should match
            ``PDB_ID`` case-insensitively, e.g. ``7b2m.pdb`` or ``7B2M.pdb``).

    Returns:
        List of dicts compatible with the test pipeline (same keys as
        :func:`load_sabdab_entries`).
    """
    pdb_dir = Path(pdb_dir)
    df = pd.read_csv(csv_path)

    # Keep first occurrence of each PDB_ID (wildtype / most complete row)
    df = df.drop_duplicates(subset="PDB_ID", keep="first")
    logger.info(
        "ANDD CSV: %d unique PDB entries after deduplication.", len(df)
    )

    # Build a case-insensitive lookup of available PDB files
    pdb_lookup: dict[str, Path] = {}
    for p in pdb_dir.glob("*.pdb"):
        pdb_lookup[p.stem.lower()] = p

    def _first_chain(cell) -> str | None:
        """Return the first chain letter from a comma-separated cell."""
        if pd.isna(cell) or str(cell).strip() in ("", "nan", "\\"):
            return None
        return str(cell).split(",")[0].strip()

    entries: list[dict] = []
    skipped_no_file = 0
    skipped_no_seq = 0
    skipped_no_chain = 0

    for _, row in df.iterrows():
        pdb_id = str(row["PDB_ID"]).strip()
        pdb_path = pdb_lookup.get(pdb_id.lower())

        if pdb_path is None:
            skipped_no_file += 1
            logger.debug("No PDB file for %s, skipping.", pdb_id)
            continue

        # Sequence — prefer CSV value, fall back to PDB extraction
        sequence = str(row.get("Ab/Nano H_Chain AA", "")).strip()
        if sequence in ("", "nan", "\\"):
            sequence = None

        nb_chain = _first_chain(row.get("H_Chain Auth Asym ID"))
        ag_chain = _first_chain(row.get("Ag_Auth Asym ID"))

        if nb_chain is None:
            skipped_no_chain += 1
            logger.warning("No nanobody chain ID for %s, skipping.", pdb_id)
            continue

        # If sequence not in CSV, extract from PDB
        if sequence is None:
            sequence = extract_chain_sequence(str(pdb_path), nb_chain)
            if sequence is None:
                skipped_no_seq += 1
                logger.warning(
                    "Could not extract sequence for %s chain %s, skipping.",
                    pdb_id, nb_chain,
                )
                continue

        entries.append(
            {
                "candidate_id": pdb_id.lower(),
                "raw_sequence": sequence,
                "pdb_filepath": str(pdb_path),
                "complex_pdb_path": str(pdb_path) if ag_chain else None,
                "nanobody_chain_id": nb_chain,
                "antigen_chain_ids": ag_chain,
            }
        )

    logger.info(
        "ANDD: loaded %d entries (skipped: %d no file, %d no chain, %d no seq).",
        len(entries), skipped_no_file, skipped_no_chain, skipped_no_seq,
    )
    return entries


def load_aapr_entries(csv_path: str) -> list[dict]:
    """Load pipeline-ready entries from an AAPR candidate manifest CSV.

    The AAPR sampler (``scripts/aapr/sample_candidates.py``) emits one row
    per (GT, sample) candidate with the per-row ``complex_pdb_path``
    already pointing at the generated PDB. Unlike SAbDab/ANDD loading,
    there is no directory search and no sequence-from-PDB extraction —
    the manifest carries everything the judges need.

    Critically, this loader preserves ``gt_complex_id`` and ``sample_idx``
    from the manifest so the scored parquet retains the grouping needed
    by downstream Pareto pair selection.

    Manifest schema (per docs/aapr_generation_context.md §6):
      - ``candidate_id``: per-sample primary key (e.g. ``7f5g_B_s0003``)
      - ``gt_complex_id``: parent GT id (e.g. ``7f5g_B``)
      - ``sample_idx``: 0..K-1 replicate index
      - ``raw_sequence``: reconstructed VHH sequence
      - ``complex_pdb_path``: path to the AAPR-generated PDB
      - ``nanobody_chain_id``: chain letter for the VHH
      - ``antigen_chain_ids``: comma-joined antigen chain letter(s)
      - (plus provenance: ``mask_strategy``, ``cdrs_masked``,
        ``temperature``, ``checkpoint_id``, ``seed``)

    Args:
        csv_path: Path to the AAPR manifest CSV.

    Returns:
        List of dicts with the keys shared by :func:`load_andd_entries`
        plus ``gt_complex_id`` and ``sample_idx`` for downstream grouping.
        Rows whose ``complex_pdb_path`` is missing on disk are skipped
        with a warning.
    """
    df = pd.read_csv(csv_path)
    logger.info("AAPR manifest: %d rows in %s", len(df), csv_path)

    required = {
        "candidate_id", "gt_complex_id", "sample_idx", "raw_sequence",
        "complex_pdb_path", "nanobody_chain_id", "antigen_chain_ids",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"AAPR manifest {csv_path!r} is missing required columns: "
            f"{sorted(missing)}. Expected schema from "
            "docs/aapr_generation_context.md §6."
        )

    entries: list[dict] = []
    skipped_no_file = 0
    for _, row in df.iterrows():
        pdb_path = Path(str(row["complex_pdb_path"]))
        if not pdb_path.exists():
            skipped_no_file += 1
            logger.debug("PDB not on disk for %s: %s", row["candidate_id"], pdb_path)
            continue

        ag_chains = row.get("antigen_chain_ids")
        if pd.isna(ag_chains) or str(ag_chains).strip() in ("", "nan"):
            ag_chains = None
        else:
            ag_chains = str(ag_chains).strip()

        entries.append(
            {
                "candidate_id":      str(row["candidate_id"]),
                "raw_sequence":      str(row["raw_sequence"]),
                "pdb_filepath":      str(pdb_path),
                "complex_pdb_path":  str(pdb_path) if ag_chains else None,
                "nanobody_chain_id": str(row["nanobody_chain_id"]),
                "antigen_chain_ids": ag_chains,
                "gt_complex_id":     str(row["gt_complex_id"]),
                "sample_idx":        int(row["sample_idx"]),
            }
        )

    logger.info(
        "AAPR: loaded %d entries (skipped %d missing PDB files).",
        len(entries), skipped_no_file,
    )
    return entries


def load_pdb_entries(
    pdb_dir: str,
    chain_id: str = "A",
    antigen_chain_id: str | None = None,
) -> list[dict]:
    """Load pipeline-ready entries directly from a directory of PDB files.

    Use this when there is no SAbDab TSV — e.g. for IgLM-generated or other
    curated VHH structures.  If ``antigen_chain_id`` is provided the Physics
    Judge will run; otherwise it is skipped.

    Args:
        pdb_dir: Directory containing ``.pdb`` files.
        chain_id: Chain letter that holds the nanobody sequence (default ``"A"``).
        antigen_chain_id: Chain letter(s) for the antigen (e.g. ``"B"``).
            Set to ``None`` for nanobody-only structures.

    Returns:
        List of dicts compatible with the test pipeline (same keys as
        :func:`load_sabdab_entries`).
    """
    pdb_dir = Path(pdb_dir)
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    logger.info(
        "Found %d PDB files in %s (nanobody chain='%s', antigen chain='%s')",
        len(pdb_files), pdb_dir, chain_id, antigen_chain_id or "none",
    )

    entries: list[dict] = []
    for pdb_path in pdb_files:
        candidate_id = pdb_path.stem
        sequence = extract_chain_sequence(str(pdb_path), chain_id)
        if sequence is None:
            logger.warning(
                "Could not extract sequence for %s chain %s, skipping.",
                candidate_id, chain_id,
            )
            continue
        entries.append(
            {
                "candidate_id": candidate_id,
                "raw_sequence": sequence,
                "pdb_filepath": str(pdb_path),
                "complex_pdb_path": str(pdb_path) if antigen_chain_id else None,
                "nanobody_chain_id": chain_id,
                "antigen_chain_ids": antigen_chain_id,
            }
        )

    logger.info("Loaded %d entries with valid sequences.", len(entries))
    return entries
