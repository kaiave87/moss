"""
Four-Channel Hebbian Strengthening — Constants and Utilities

Channels:
  1. Event-driven (STDP) — during recall
  2. Rapid consolidation (E-LTP → L-LTP) — phase-aware passive strengthening
  3. Sleep consolidation (SWR + STC) — overnight consolidation sub-phases
  4. Spacing (Cepeda ridgeline) — age-dependent review intervals

References:
    Frey & Morris 1997 (Synaptic Tagging and Capture) — Nature
    Nader 2000 (Reconsolidation lability) — Nature
    Cepeda 2006/2008 (Spacing effect ridgeline)
    Remme 2021 (Power-law decay) — PLOS Computational Biology
    Turrigiano 1998 (Homeostatic scaling)

Patent pending — Lichen Research Inc., Canadian application filed March 2026.
"""

from dataclasses import dataclass
from typing import Dict

# ── Phase Gain Multipliers ──
# Modulate Hebbian plasticity by system phase (Hasselmo 1999).
# Phase names correspond to daily cognitive rhythm:
#   PRIME  = morning encoding
#   ACTIVE = afternoon focused work
#   FLOW   = deep work / high engagement
#   SORT   = evening synthesis
#   DREAM  = overnight consolidation
PHASE_GAINS: Dict[str, float] = {
    "PRIME":  0.8,   # Morning — high encoding, moderate consolidation
    "ACTIVE": 0.5,   # Afternoon — low background, active work dominates
    "FLOW":   1.0,   # Deep work — full plasticity
    "SORT":   0.3,   # Evening — minimal, synthesis mode
    "DREAM":  2.0,   # Night — aggressive consolidation
}

# ── Boost Parameters ──
TEMPORAL_BOOST_BASE = 0.02      # Base boost for temporal co-occurrence
ENTITY_BOOST_BASE = 0.01        # Base boost for entity co-occurrence
COOLDOWN_HOURS = 6              # Hours between boosts for same pair

# ── Pathway Thresholds ──
STRONG_THRESHOLD = 0.5          # Pathway considered "strong" above this
TAG_EXPIRY_HOURS = 12           # STC tag window (Synaptic Tagging & Capture)
LABILE_HOURS = 1                # Reconsolidation lability window (Nader 2000)
LABILE_DECAY_FACTOR = 0.95      # Decay on lability expiry (failure to restabilize)
LABILE_CAP_PER_EPOCH = 50       # Max labile events per 24h epoch

# ── Decay Parameters ──
HOMEOSTATIC_GAMMA = 0.005       # Global scaling factor during overnight phase
PRUNING_THRESHOLD = 0.005       # Delete edges below this strength
FLOOR_BASE = 0.02               # Importance-weighted decay floor
FLOOR_AGE_HALFLIFE = 60         # Days until age factor halves the floor
POWER_LAW_ALPHA = 0.3           # Decay exponent (Remme 2021)

# ── Spacing Effect (Cepeda 2006) ──
SPACING_RATIO = 0.15            # Review at 15% of edge age

# ── STC Parameters ──
STC_CLUSTER_BONUS = 0.03        # Extra boost for STC-captured edges near strong clusters
STC_BASE_BOOST = 0.02           # Base capture boost

# ── Per-Phase LIMIT Caps ──
LIMITS_WAKING: Dict[str, int] = {"temporal": 50, "entity": 100}
LIMITS_DREAM: Dict[str, int] = {"temporal": 200, "entity": 500}

# ── Max Degree ──
MAX_EDGES_PER_NODE = 500        # Degree limit per memory node


@dataclass
class PhaseContext:
    """Current phase context for plasticity modulation."""
    phase: str
    gain: float
    limits: Dict[str, int]


def get_phase_context(phase: str = "ACTIVE") -> PhaseContext:
    """Return PhaseContext for the given phase name.

    Args:
        phase: Phase name (PRIME, ACTIVE, FLOW, SORT, DREAM).

    Returns:
        PhaseContext with gain and limit caps.
    """
    gain = PHASE_GAINS.get(phase.upper(), 0.5)
    limits = LIMITS_DREAM if phase.upper() == "DREAM" else LIMITS_WAKING
    return PhaseContext(phase=phase.upper(), gain=gain, limits=limits)


def age_dependent_floor(
    importance: float,
    days_since_last_access: float,
    floor_base: float = None
) -> float:
    """Calculate age-dependent decay floor (Zenke & Gerstner 2017).

    Prevents a static floor from artificially protecting unused pathways.
    Untouched pathways decay further over time — only active use sustains them.

    floor = floor_base * importance * age_factor
    age_factor = max(0.1, 2^(-days_inactive / halflife))

    Args:
        importance: Memory importance score (0.0–1.0).
        days_since_last_access: Days since last access.
        floor_base: Override default FLOOR_BASE.

    Returns:
        Decay floor value.
    """
    if floor_base is None:
        floor_base = FLOOR_BASE
    age_factor = max(0.1, 2.0 ** (-days_since_last_access / FLOOR_AGE_HALFLIFE))
    return floor_base * importance * age_factor


def power_law_decay(strength: float, days_inactive: float) -> float:
    """Calculate power-law decay (Remme 2021, PLOS Comp Bio).

    w_new = w * t^(-alpha) where alpha = POWER_LAW_ALPHA.
    More biologically accurate than exponential decay.
    """
    if days_inactive <= 0:
        return strength
    decay_factor = days_inactive ** (-POWER_LAW_ALPHA)
    return strength * min(1.0, decay_factor)


def cepeda_review_interval(edge_age_days: float) -> float:
    """Calculate optimal review interval using Cepeda ridgeline.

    Optimal spacing = SPACING_RATIO * desired_retention_time.

    Returns: optimal review interval in days.
    """
    return max(0.25, edge_age_days * SPACING_RATIO)


def is_at_review_point(edge_age_days: float, days_since_last_boost: float) -> bool:
    """Check if an edge is at its optimal Cepeda review point.

    True if days_since_last_boost is within 20% of the optimal interval.
    """
    optimal = cepeda_review_interval(edge_age_days)
    tolerance = optimal * 0.2
    return abs(days_since_last_boost - optimal) <= tolerance


def boosted_gain(base_boost: float, phase_gain: float, is_labile: bool = False) -> float:
    """Calculate final boost amount with phase and lability modulation.

    delta_w = base_boost * phi(phase) * labile_multiplier

    Labile pathways (recently reactivated, in reconsolidation window)
    receive 1.5x boost — consistent with Nader (2000) reconsolidation data.
    """
    labile_mult = 1.5 if is_labile else 1.0
    return base_boost * phase_gain * labile_mult


def prp_signal(mean_importance: float) -> float:
    """Calculate plasticity-related protein (PRP) signal for STC.

    Higher importance → stronger PRP → more synaptic capture.
    Scales between 0.5 (low importance) and 2.0 (high importance).
    """
    return 0.5 + (mean_importance * 1.5)
