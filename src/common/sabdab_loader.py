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
