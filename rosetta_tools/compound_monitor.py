"""
compound_monitor.py — Framework for cross-module evidence propagation.

Lifted from CIA's vigilance logic (cia_trace/trace.py::_make_verdict) and
generalised for any Rosetta-based compound monitor.

Vigilance principle: evidence of anomaly in one module should tighten the
thresholds used by other modules — a concept that "almost" allocates on a
clean input is noise; the same signal on an input with confirmed surface
tampering or other red flags is meaningful.

This is general beyond security:
  - High output entropy should tighten weak-allocation thresholds.
  - OOD early-layer activations should lower confidence in downstream signals.
  - Multi-modal monitoring: signal in one module propagates as prior to others.

Written: 2026-04-28 UTC
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModuleEvidence:
    """Evidence output from one monitoring module.

    Attributes
    ----------
    name : str
        Module identifier (e.g. "surface", "entropy", "ood_detector").
    confidence : float
        Anomaly confidence in [0.0, 1.0].  0.0 = clean, 1.0 = strongly anomalous.
    detail : dict
        Module-specific metadata (e.g. obfuscation_risk, perplexity_score).
    """
    name: str
    confidence: float
    detail: dict = field(default_factory=dict)

    def __post_init__(self):
        self.confidence = max(0.0, min(1.0, self.confidence))


def vigilance_threshold(
    base_threshold: float,
    evidence: list[ModuleEvidence],
    weights: dict[str, float],
    floor_factor: float = 0.5,
) -> float:
    """Tighten base_threshold in proportion to cross-module anomaly evidence.

    A weighted sum of module confidence scores reduces the effective threshold,
    making the monitor more sensitive when other modules detect anomalies.

    Parameters
    ----------
    base_threshold : float
        The standard threshold (e.g. allocation_strength > 1.0 for CIA).
    evidence : list[ModuleEvidence]
        Evidence from other modules.  Only modules present in weights are used.
    weights : dict[str, float]
        How much each module's confidence should shift the threshold.
        weight=0.3 means full confidence from that module reduces threshold by 30%.
    floor_factor : float
        Minimum threshold as fraction of base.  Default 0.5 prevents threshold
        from collapsing to zero.

    Returns
    -------
    float — tightened threshold, in [base * floor_factor, base].

    Examples
    --------
    CIA instance (surface module weight=0.3):
        vigilance_threshold(
            base_threshold=1.0,
            evidence=[ModuleEvidence("surface", confidence=1.0)],
            weights={"surface": 0.3},
        )
        → 0.7  (threshold drops from 1.0 to 0.7 under full surface anomaly)
    """
    factor = 1.0
    for ev in evidence:
        w = weights.get(ev.name, 0.0)
        factor -= w * ev.confidence
    floor = base_threshold * floor_factor
    return max(base_threshold * factor, floor)


def apply_vigilance(
    allocations: dict[str, float],
    base_threshold: float,
    evidence: list[ModuleEvidence],
    weights: dict[str, float],
    floor_factor: float = 0.5,
) -> dict[str, bool]:
    """Apply vigilance-adjusted threshold to a set of allocation strengths.

    Parameters
    ----------
    allocations : dict[str, float]
        Per-concept allocation strength values.
    base_threshold : float
        Standard allocation threshold (typically 1.0).
    evidence : list[ModuleEvidence]
        Cross-module anomaly evidence.
    weights : dict[str, float]
        Per-module weight for threshold tightening.
    floor_factor : float
        Minimum threshold floor as fraction of base.

    Returns
    -------
    dict[str, bool] — per-concept allocated-under-vigilance verdict.
    """
    threshold = vigilance_threshold(base_threshold, evidence, weights, floor_factor)
    return {concept: strength >= threshold
            for concept, strength in allocations.items()}
