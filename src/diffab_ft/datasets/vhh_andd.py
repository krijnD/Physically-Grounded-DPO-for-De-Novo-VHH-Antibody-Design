"""DiffAb-compatible dataset adapter for our curated VHH+antigen set.

Why a subclass instead of a fork
--------------------------------
DiffAb's :class:`SAbDabDataset` does four things in its ``__init__``:
  1. ``_load_sabdab_entries`` — parse the summary TSV, filter by
     resolution and antigen type.
  2. ``_load_structures``     — preprocess Biopython structures into
     an LMDB cache (CDR labels, heavy-atom tensors).
  3. ``_load_clusters``       — run MMseqs2 on CDR-H3 sequences at
     **50%% identity** and write ``cluster_result_cluster.tsv``.
  4. ``_load_split``          — split based on hard-coded
     ``TEST_ANTIGENS`` and the just-built CDR-H3 clusters.

Steps 2 (LMDB caching) and the heavy-atom parsing inside step 1 are
exactly what we want and contain non-trivial logic we shouldn't
duplicate. Steps 3 and 4 are the *wrong policy* for our data:

  * The 50% CDR-H3-only clustering would silently override the 70%
    concatenated-CDR clustering we already built for thesis-level rigor
    (see :mod:`scripts.diffab_ft.cluster_split`).
  * 64% of our data is cryo-EM with ``resolution=NOT``; the upstream
    ``RESOLUTION_THRESHOLD = 4.0`` filter (line 277 of sabdab.py) would
    drop them all because the predicate is ``resolution is not None and
    resolution <= 4.0``. We've already manually curated for resolution
    in ``curate_andd.py``, so we trust our own filter.
  * The upstream ``TEST_ANTIGENS`` list is SAbDab-specific (SARS-CoV-2,
    HIV gp160, …) — we want our cluster-level splits.

So we subclass and override exactly those four hooks. Everything else
(LMDB parsing, transforms, ``__getitem__``, the chain-merge logic in
:func:`preprocess_sabdab_structure`) is inherited unchanged.

Splits supported
----------------
The config's ``dataset.<split>.split`` field accepts:

* ``"train"``, ``"val"``, ``"test"`` — read directly from
  ``cluster_splits.json``.
* ``"test_antigen_disjoint"`` — derived: take ``test`` entries and
  filter to those whose antigen cluster (50% identity, computed by
  ``cluster_split.py --audit-antigens``) is **not** present in
  ``train``. This is the strict held-out test set used to detect
  antigen-side leakage from the primary CDR-only split. Construction is
  reproducible from the artifacts in ``cluster_splits.json``'s sibling
  ``antigen_cluster_cluster.tsv``.

Config schema
-------------
::

    dataset:
      train:
        type: vhh_andd
        manifest_path: data/datasets/diffab_manifest.tsv
        pdb_dir:       /projects/.../VHH_structures_post_diffab
        processed_dir: data/processed/arm_a   # arm-specific to avoid LMDB clashes
        splits_path:   data/datasets/clustering/cluster_splits.json
        split:         train
        # split_seed unused (splits already materialized in JSON), but
        # accepted for parity with upstream.
        transform: [...]

Registers under ``vhh_andd`` so ``get_dataset({type: vhh_andd, ...})``
constructs an instance.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Iterable

import joblib
import lmdb
import pandas as pd
from Bio import PDB
from Bio.PDB import PDBExceptions
from tqdm.auto import tqdm

# DiffAb imports — these require third_party/diffab on sys.path. The
# training entrypoint (scripts/diffab_ft/train.py) handles that; for
# unit-test imports the user must add it themselves.
from diffab.datasets._base import register_dataset
from diffab.datasets.sabdab import (
    ALLOWED_AG_TYPES,
    SAbDabDataset,
    _label_heavy_chain_cdr,
    nan_to_empty_string,
    nan_to_none,
    parse_sabdab_resolution,
    split_sabdab_delimited_str,
)
from diffab.utils.protein import parsers

logger = logging.getLogger(__name__)


VALID_SPLITS = ("train", "val", "test", "test_antigen_disjoint")


def _entry_id(pdb: str, h_chain: str) -> str:
    """ID format used by ``scripts/diffab_ft/cluster_split.py``.

    Lowercase PDB + underscore + (case-preserved) H-chain auth-asym ID.
    Must match exactly so the JSON splits/cluster-assignments index
    against entries we load here.
    """
    return f"{str(pdb).strip().lower()}_{str(h_chain).strip()}"


# Default cap for the heavy-chain parser. The upstream DiffAb default of
# 113 was designed for Chothia-numbered SAbDab antibody PDBs (where
# Chothia 113 = last residue of the canonical J-anchor ``WGQGTQVTVSS``).
# Our VHH PDBs come straight from RCSB and many use sequential or
# author numbering — for those, the 113-cap silently truncates the
# J-anchor (sometimes mid-CDR3), which causes downstream NanoBodyBuilder2
# folding to refuse the resulting sequences. Raising the cap to 150
# covers any reasonable VHH length while remaining well below the
# next-chain boundary in multi-chain PDBs (chains are passed in
# separately, so this just gates resseq within a single chain).
# Investigation in docs/aapr_masking_research_context.md.
DEFAULT_HEAVY_MAX_RESSEQ = 150


def _preprocess_vhh_structure(task, heavy_max_resseq: int = DEFAULT_HEAVY_MAX_RESSEQ):
    """VHH-aware reimplementation of DiffAb's ``preprocess_sabdab_structure``.

    Identical to the upstream function except:
      * The heavy-chain ``max_resseq`` is configurable (default
        ``DEFAULT_HEAVY_MAX_RESSEQ``) instead of hardcoded to 113.
      * The light-chain branch is preserved for parity but is dead code
        for our VHH dataset (``L_chain`` is always ``None``).

    Returns the same dict shape as the upstream:
    ``{id, heavy, heavy_seqmap, light, light_seqmap, antigen, antigen_seqmap}``
    or ``None`` if parsing failed (matches upstream error semantics so
    ``_load_structures``' filter logic keeps working).
    """
    entry = task["entry"]
    pdb_path = task["pdb_path"]

    parser = PDB.PDBParser(QUIET=True)
    model = parser.get_structure(id, pdb_path)[0]

    parsed = {
        "id": entry["id"],
        "heavy": None,
        "heavy_seqmap": None,
        "light": None,
        "light_seqmap": None,
        "antigen": None,
        "antigen_seqmap": None,
    }
    try:
        if entry["H_chain"] is not None:
            (
                parsed["heavy"],
                parsed["heavy_seqmap"],
            ) = _label_heavy_chain_cdr(*parsers.parse_biopython_structure(
                model[entry["H_chain"]],
                max_resseq=heavy_max_resseq,
            ))

        # Dead branch for VHH (L_chain always None), kept for parity.
        if entry["L_chain"] is not None:  # pragma: no cover
            from diffab.datasets.sabdab import _label_light_chain_cdr
            (
                parsed["light"],
                parsed["light_seqmap"],
            ) = _label_light_chain_cdr(*parsers.parse_biopython_structure(
                model[entry["L_chain"]],
                max_resseq=106,
            ))

        if parsed["heavy"] is None and parsed["light"] is None:
            raise ValueError("Neither valid H-chain or L-chain is found.")

        if len(entry["ag_chains"]) > 0:
            chains = [model[c] for c in entry["ag_chains"]]
            (
                parsed["antigen"],
                parsed["antigen_seqmap"],
            ) = parsers.parse_biopython_structure(chains)

    except (
        PDBExceptions.PDBConstructionException,
        parsers.ParsingException,
        KeyError,
        ValueError,
    ) as e:
        logging.warning("[%s] %s: %s", task["id"], e.__class__.__name__, str(e))
        return None

    return parsed


class VHHANDDDataset(SAbDabDataset):
    """SAbDabDataset subclass that consumes our curated manifest + splits.

    Overrides:
      * ``_load_sabdab_entries`` (relax resolution filter, align entry IDs)
      * ``_load_clusters``        (read from JSON, no MMseqs2)
      * ``_load_split``           (JSON splits + antigen-disjoint variant)
      * ``_preprocess_structures`` (raise heavy-chain ``max_resseq`` cap so
        the J-anchor isn't silently truncated on sequentially-numbered
        VHH PDBs — see ``DEFAULT_HEAVY_MAX_RESSEQ`` above)

    ``__getitem__``, ``get_structure``, ``_load_structures`` (the LMDB
    plumbing on top of ``_preprocess_structures``), and the heavy-atom
    parsing in :func:`preprocess_sabdab_structure` (mirrored locally as
    :func:`_preprocess_vhh_structure`) are inherited / borrowed as-is.
    """

    def __init__(
        self,
        manifest_path: str,
        pdb_dir: str,
        processed_dir: str,
        splits_path: str,
        split: str = "train",
        split_seed: int = 42,
        transform=None,
        reset: bool = False,
        heavy_max_resseq: int = DEFAULT_HEAVY_MAX_RESSEQ,
    ):
        # Set extras BEFORE calling super().__init__: the parent ctor
        # immediately calls our overridden hooks, which reference these.
        self.heavy_max_resseq = int(heavy_max_resseq)
        self.splits_path = Path(splits_path)
        if not self.splits_path.exists():
            raise FileNotFoundError(f"splits JSON not found: {self.splits_path}")
        with open(self.splits_path) as f:
            self._splits_data = json.load(f)
        # Sanity checks on the JSON shape we wrote in cluster_split.py.
        for required_key in ("splits", "cluster_assignments"):
            if required_key not in self._splits_data:
                raise KeyError(
                    f"splits JSON missing required key {required_key!r}: "
                    f"{self.splits_path}"
                )

        super().__init__(
            summary_path=manifest_path,
            chothia_dir=pdb_dir,
            processed_dir=processed_dir,
            split=split,
            split_seed=split_seed,
            transform=transform,
            reset=reset,
        )

    # ── Override 1: relax filters, align entry IDs ──────────────────────
    def _load_sabdab_entries(self):
        """Parse our DiffAb-format manifest TSV.

        Identical to the parent except:
          * Resolution filter is *removed*. Our 64% cryo-EM rows arrive
            with ``resolution = "NOT"`` (which ``parse_sabdab_resolution``
            normalizes to ``None``). The parent would drop those because
            its predicate is ``resolution is not None and resolution <= 4.0``.
          * The entry ID is ``"{pdb}_{H}"`` (matching what
            ``cluster_split.py`` wrote into ``cluster_splits.json``)
            rather than the parent's ``"{pdb}_{H}_{L}_{Ag}"``.
          * The ``ALLOWED_AG_TYPES`` filter is preserved as a defensive
            check; our ``prepare_manifest.py`` already conforms.
        """
        df = pd.read_csv(self.summary_path, sep="\t")
        entries_all = []
        n_dropped_ag_type = 0

        for _, row in tqdm(
            df.iterrows(),
            dynamic_ncols=True,
            desc="Loading VHH entries",
            total=len(df),
        ):
            pdbcode = str(row["pdb"]).strip().lower()
            h_chain = nan_to_none(row["Hchain"])
            l_chain = nan_to_none(row["Lchain"])  # always None for VHH
            ag_chains = split_sabdab_delimited_str(
                nan_to_empty_string(row["antigen_chain"])
            )
            ag_type = nan_to_none(row["antigen_type"])
            resolution = parse_sabdab_resolution(row["resolution"])

            # Date is mandatory for the parent; ours always has one
            # (prepare_manifest.py reads it from the PDB header).
            try:
                date = datetime.datetime.strptime(row["date"], "%m/%d/%y")
            except (ValueError, TypeError):
                # Fall back to a sentinel — DiffAb's training loop never
                # uses .date for sampling, only for logging.
                date = datetime.datetime(1900, 1, 1)

            entry = {
                "id": _entry_id(pdbcode, h_chain or ""),
                "pdbcode": pdbcode,
                "H_chain": h_chain,
                "L_chain": l_chain,
                "ag_chains": ag_chains,
                "ag_type": ag_type,
                "ag_name": nan_to_none(row.get("antigen_name")),
                "date": date,
                "resolution": resolution,
                "method": row.get("method"),
                "scfv": row.get("scfv"),
            }

            # Antigen-type filter only — drop the resolution gate.
            if entry["ag_type"] in ALLOWED_AG_TYPES or entry["ag_type"] is None:
                entries_all.append(entry)
            else:
                n_dropped_ag_type += 1

        logger.info(
            "Loaded %d VHH entries from %s (dropped %d for antigen_type).",
            len(entries_all), self.summary_path, n_dropped_ag_type,
        )
        self.sabdab_entries = entries_all

    # ── Override 2: read clusters from JSON, no MMseqs2 ────────────────
    def _load_clusters(self, reset):
        """Populate ``self.clusters`` and ``self.id_to_cluster`` directly
        from ``cluster_splits.json`` — bypassing the parent's
        CDR-H3-only MMseqs2 step. The ``reset`` flag is accepted for
        signature parity but ignored (the JSON is the source of truth)."""
        cluster_assignments: dict[str, str] = self._splits_data["cluster_assignments"]
        clusters: dict[str, list[str]] = {}
        for member, rep in cluster_assignments.items():
            clusters.setdefault(rep, []).append(member)

        self.clusters = clusters
        self.id_to_cluster = dict(cluster_assignments)
        logger.info(
            "Loaded %d clusters spanning %d members from %s.",
            len(clusters), len(cluster_assignments), self.splits_path,
        )

    # ── Override 3: split lookup, with antigen-disjoint variant ────────
    def _load_split(self, split, split_seed):
        """Resolve split name → list of entry IDs.

        ``split_seed`` is unused (splits are already materialized) but
        accepted for parity with the parent.
        """
        if split not in VALID_SPLITS:
            raise ValueError(
                f"split must be one of {VALID_SPLITS}, got {split!r}"
            )

        # ``self.db_ids`` is populated by super()._load_structures, which
        # runs before _load_split. Some manifest entries can fail in
        # preprocess_sabdab_structure (CDR-H3 too long, missing chain),
        # so we filter the JSON's split lists down to what's actually
        # in LMDB.
        live_ids: set[str] = set(self.db_ids or [])

        if split == "test_antigen_disjoint":
            ids_in_split = self._compute_antigen_disjoint_test(live_ids)
        else:
            json_ids: Iterable[str] = self._splits_data["splits"].get(split, [])
            ids_in_split = [i for i in json_ids if i in live_ids]
            n_dropped = len(list(json_ids)) - len(ids_in_split)
            if n_dropped:
                logger.info(
                    "Split %r: %d entries from JSON, %d after LMDB filter "
                    "(dropped %d that failed structure preprocessing).",
                    split, len(self._splits_data["splits"].get(split, [])),
                    len(ids_in_split), n_dropped,
                )

        if not ids_in_split:
            raise RuntimeError(
                f"Split {split!r} resolved to 0 entries. "
                "Check that splits_path and the LMDB are consistent."
            )

        self.ids_in_split = ids_in_split
        logger.info("Final split %r size: %d", split, len(ids_in_split))

    # ── Override 4: preprocess structures with a larger J-anchor cap ────
    def _preprocess_structures(self):
        """Build the LMDB cache with :func:`_preprocess_vhh_structure`.

        Mirrors the upstream ``_preprocess_structures`` exactly except
        for the call site: it dispatches to our heavy-chain-cap-aware
        helper instead of the upstream module-level function. Everything
        else (joblib parallelism, LMDB writer, ids file) is unchanged
        so the rest of the parent class continues to work.
        """
        tasks = []
        for entry in self.sabdab_entries:
            pdb_path = os.path.join(
                self.chothia_dir, "{}.pdb".format(entry["pdbcode"])
            )
            if not os.path.exists(pdb_path):
                logger.warning("PDB not found: %s", pdb_path)
                continue
            tasks.append({
                "id": entry["id"],
                "entry": entry,
                "pdb_path": pdb_path,
            })

        cap = self.heavy_max_resseq
        logger.info(
            "Preprocessing %d structures (heavy_max_resseq=%d).",
            len(tasks), cap,
        )

        data_list = joblib.Parallel(
            n_jobs=max(joblib.cpu_count() // 2, 1),
        )(
            joblib.delayed(_preprocess_vhh_structure)(task, cap)
            for task in tqdm(tasks, dynamic_ncols=True, desc="Preprocess")
        )

        db_conn = lmdb.open(
            self._structure_cache_path,
            map_size=self.MAP_SIZE,
            create=True,
            subdir=False,
            readonly=False,
        )
        ids = []
        with db_conn.begin(write=True, buffers=True) as txn:
            for data in tqdm(data_list, dynamic_ncols=True, desc="Write to LMDB"):
                if data is None:
                    continue
                ids.append(data["id"])
                txn.put(data["id"].encode("utf-8"), pickle.dumps(data))

        with open(self._structure_cache_path + "-ids", "wb") as f:
            pickle.dump(ids, f)

    # ── Helper: derive the antigen-disjoint held-out test set ──────────
    def _compute_antigen_disjoint_test(self, live_ids: set[str]) -> list[str]:
        """Return the subset of ``test`` whose antigen cluster does not
        also appear in ``train``.

        Reads ``antigen_cluster_cluster.tsv`` (sibling of the splits
        JSON, written by ``cluster_split.py --audit-antigens``). Each
        line is ``<rep_entry_id>\\t<member_entry_id>``, with entry IDs
        in the same ``{pdb}_{H}`` format we use.

        If the audit file is missing we raise — silently returning the
        full test set would mislead a downstream evaluation comparison.
        """
        antigen_tsv = self.splits_path.parent / "antigen_cluster_cluster.tsv"
        if not antigen_tsv.exists():
            raise FileNotFoundError(
                f"test_antigen_disjoint requires the antigen audit at "
                f"{antigen_tsv}. Re-run cluster_split.py with --audit-antigens."
            )

        member_to_rep: dict[str, str] = {}
        with open(antigen_tsv) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                rep, member = parts
                member_to_rep[member] = rep

        train_ids = set(self._splits_data["splits"].get("train", []))
        test_ids = set(self._splits_data["splits"].get("test", []))

        train_antigen_clusters = {
            member_to_rep[m] for m in train_ids if m in member_to_rep
        }

        disjoint = []
        n_test_no_antigen = 0
        n_test_in_train = 0
        for tid in sorted(test_ids):
            if tid not in live_ids:
                continue
            rep = member_to_rep.get(tid)
            if rep is None:
                # Entry's antigen sequence couldn't be extracted during
                # the audit — exclude conservatively.
                n_test_no_antigen += 1
                continue
            if rep in train_antigen_clusters:
                n_test_in_train += 1
                continue
            disjoint.append(tid)

        logger.info(
            "test_antigen_disjoint: kept %d / %d test entries "
            "(%d removed for antigen-cluster overlap with train, "
            "%d for missing antigen-audit entry).",
            len(disjoint), len(test_ids), n_test_in_train, n_test_no_antigen,
        )
        return disjoint


@register_dataset("vhh_andd")
def get_vhh_andd_dataset(cfg, transform):
    """Registry hook: build a :class:`VHHANDDDataset` from EasyDict cfg.

    Accepts an optional ``heavy_max_resseq`` field in the YAML to
    override the default J-anchor cap (see ``DEFAULT_HEAVY_MAX_RESSEQ``).
    """
    # Path resolution is the caller's responsibility (the trainer
    # already cd's to the project root, so relative paths in the YAML
    # work from there).
    if not os.path.isdir(cfg.pdb_dir):
        raise NotADirectoryError(
            f"vhh_andd: pdb_dir does not exist or is not a directory: {cfg.pdb_dir}"
        )
    return VHHANDDDataset(
        manifest_path=cfg.manifest_path,
        pdb_dir=cfg.pdb_dir,
        processed_dir=cfg.processed_dir,
        splits_path=cfg.splits_path,
        split=cfg.split,
        split_seed=cfg.get("split_seed", 42),
        transform=transform,
        reset=cfg.get("reset", False),
        heavy_max_resseq=cfg.get("heavy_max_resseq", DEFAULT_HEAVY_MAX_RESSEQ),
    )
