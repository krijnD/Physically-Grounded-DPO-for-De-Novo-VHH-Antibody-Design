"""VHH-specific dataset adapters for DiffAb fine-tuning.

Importing this package registers ``vhh_andd`` in DiffAb's dataset
registry, so configs can use ``dataset.<split>.type: vhh_andd``.
"""

from . import vhh_andd  # noqa: F401  — side-effect: registers dataset
