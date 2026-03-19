"""Hebbian Neuroplastic Memory — core exports."""

from moss.hebbian.recall import recall, RecallResult
from moss.hebbian.pathway_strengthening import strengthen_pathway_sync, weaken_pathway_sync
from moss.hebbian.spreading_activation import spreading_activation

__all__ = [
    "recall",
    "RecallResult",
    "strengthen_pathway_sync",
    "weaken_pathway_sync",
    "spreading_activation",
]
