"""Sequence masking module for the AAPR pipeline.

Provides four masking strategies for IgLM span infilling:

- ``PARATOPE``: Structure-based interface masking (candidate generation)
- ``CDR_FOCUSED``: Sequence-only CDR masking (fallback)
- ``FR2_REVERSION``: FR2 region masking (biological hard negatives)
- ``UNANCHORED_CLASH``: Full paratope without anchors (physical hard negatives)

Usage::

    from src.masking import MaskingEngine, MaskStrategy

    engine = MaskingEngine()
    result = engine.mask(candidate, MaskStrategy.PARATOPE)
"""

from src.masking.engine import MaskingEngine
from src.masking.strategies import MaskResult, MaskStrategy

__all__ = ["MaskingEngine", "MaskResult", "MaskStrategy"]
