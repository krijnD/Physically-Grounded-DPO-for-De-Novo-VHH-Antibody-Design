"""Physics Judge — stub awaiting reimplementation.

Will evaluate structural viability using Rosetta energy decomposition
(E_Rep + delta_G_bind) once the complex assembly pipeline is in place.
"""

import logging

from src.common.candidate import NanobodyCandidate

logger = logging.getLogger(__name__)


class PhysicsJudge:
    """Placeholder Physics Judge — passes all candidates through unchanged."""

    def evaluate(self, candidate: NanobodyCandidate) -> NanobodyCandidate:
        """No-op: returns the candidate without modification."""
        return candidate
