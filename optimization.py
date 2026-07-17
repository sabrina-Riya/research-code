"""
optimization.py
----------------
CO2 storage-site ranking engines: TOPSIS, WSM, and VIKOR.

Per the README's explicit generalization note (§2), this module deliberately
improves on the thesis notebook's inline implementation: it shares ONE
L2-normalized decision matrix across all three methods, supports both
benefit ("higher is better", e.g. storage capacity) and cost ("lower is
better", e.g. price, distance) criteria via explicit direction flags, and
supports an arbitrary number of criteria rather than the notebook's hardcoded
Cost + Distance pair.

For the record (and because a defensible refactor requires knowing exactly
what it changed), the notebook's actual behavior was:
  - TOPSIS: L2-normalizes Cost + Distance, weights 0.5/0.5, and (since both
    criteria are cost-type) takes ideal = weighted.min(), anti_ideal =
    weighted.max() with NO direction branching.
  - WSM: normalizes independently via min-max, sums weighted scores, and
    picks the row with the LOWEST score (idxmin) — implicitly cost-type only.
  - VIKOR: uses `.abs()` on (f_star - data)/(f_star - f_minus) rather than a
    signed direction branch — which only works because both criteria used
    are cost-type.
  - Only Cost and Distance ever entered MCDA; Storage Capacity was scored
    separately via the Linear Method (thesis §6.6), not TOPSIS/WSM/VIKOR.

The functions below reproduce the same mathematics but make the cost/benefit
handling explicit rather than implicit, so a capacity (benefit) criterion can
be added safely without silently corrupting the ranking direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from data_pipeline import build_decision_matrix, l2_normalize_matrix, minmax_normalize_matrix

Direction = str  # "benefit" or "cost"


@dataclass
class RankingResult:
    scores: pd.Series          # index-aligned to the input DataFrame
    best_index: object         # .index value of the top-ranked row
    method: str

    def best_row(self, df: pd.DataFrame) -> pd.Series:
        return df.loc[self.best_index]


def _validate_directions(criteria: Sequence[str], directions: Sequence[Direction]) -> None:
    if len(criteria) != len(directions):
        raise ValueError("`criteria` and `directions` must be the same length.")
    bad = set(directions) - {"benefit", "cost"}
    if bad:
        raise ValueError(f"Direction flags must be 'benefit' or 'cost', got: {bad}")


# --------------------------------------------------------------------------- #
# TOPSIS
# --------------------------------------------------------------------------- #

def topsis(df: pd.DataFrame,
           criteria: Sequence[str],
           directions: Sequence[Direction],
           weights: Sequence[float] | None = None) -> RankingResult:
    """
    Technique for Order Preference by Similarity to Ideal Solution.

    Steps (standard TOPSIS, generalized from the notebook's two-criterion,
    cost-only special case):
      1. L2-normalize each criterion column (r_ij = x_ij / ||column_j||_2).
      2. Apply weights.
      3. Derive the ideal-best (A*) and ideal-worst (A-) vectors per-column,
         using the direction flag (benefit -> max is ideal; cost -> min is
         ideal) instead of assuming every criterion is cost-type.
      4. Score = d(A-) / (d(A*) + d(A-)); rank descending (closer to 1 is
         better).
    """
    _validate_directions(criteria, directions)
    matrix = build_decision_matrix(df, criteria)
    n = matrix.shape[1]
    weights = np.array(weights) if weights is not None else np.full(n, 1.0 / n)
    if len(weights) != n or not np.isclose(weights.sum(), 1.0):
        raise ValueError("`weights` must have one entry per criterion and sum to 1.")

    normed = l2_normalize_matrix(matrix)
    weighted = normed * weights

    ideal_best = np.array([
        weighted[:, j].max() if directions[j] == "benefit" else weighted[:, j].min()
        for j in range(n)
    ])
    ideal_worst = np.array([
        weighted[:, j].min() if directions[j] == "benefit" else weighted[:, j].max()
        for j in range(n)
    ])

    d_pos = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    d_neg = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

    denom = d_pos + d_neg
    denom_safe = np.where(denom == 0, 1e-12, denom)
    score = d_neg / denom_safe

    scores = pd.Series(score, index=df.index, name="topsis_score")
    return RankingResult(scores=scores, best_index=scores.idxmax(), method="TOPSIS")


# --------------------------------------------------------------------------- #
# WSM (Weighted Sum Method)
# --------------------------------------------------------------------------- #

def wsm(df: pd.DataFrame,
        criteria: Sequence[str],
        directions: Sequence[Direction],
        weights: Sequence[float] | None = None) -> RankingResult:
    """
    Weighted Sum Method.

    Uses min-max normalization (matching the notebook's WSM, which
    normalizes independently of TOPSIS rather than sharing its L2 matrix —
    README §2, point 1). Benefit criteria are summed as-is after
    normalization; cost criteria are inverted (1 - normalized) before
    weighting, so that a single "higher weighted score is better" rule holds
    regardless of criterion direction — unlike the notebook, which only
    worked because both its criteria were cost-type and it picked idxmin().
    """
    _validate_directions(criteria, directions)
    matrix = build_decision_matrix(df, criteria)
    n = matrix.shape[1]
    weights = np.array(weights) if weights is not None else np.full(n, 1.0 / n)
    if len(weights) != n or not np.isclose(weights.sum(), 1.0):
        raise ValueError("`weights` must have one entry per criterion and sum to 1.")

    normed = minmax_normalize_matrix(matrix)
    for j, direction in enumerate(directions):
        if direction == "cost":
            normed[:, j] = 1.0 - normed[:, j]

    score = (normed * weights).sum(axis=1)
    scores = pd.Series(score, index=df.index, name="wsm_score")
    return RankingResult(scores=scores, best_index=scores.idxmax(), method="WSM")


# --------------------------------------------------------------------------- #
# VIKOR
# --------------------------------------------------------------------------- #

def vikor(df: pd.DataFrame,
          criteria: Sequence[str],
          directions: Sequence[Direction],
          weights: Sequence[float] | None = None,
          v: float = 0.5) -> RankingResult:
    """
    VIseKriterijumska Optimizacija I Kompromisno Resenje (compromise ranking).

    Computes group utility S_i and individual regret R_i per candidate, then
    the compromise index Q_i = v*(S-S*)/(S- - S*) + (1-v)*(R-R*)/(R- - R*).
    Lower Q is better.

    The notebook takes `.abs()` of (f_star - data)/(f_star - f_minus), which
    is only direction-correct because both of its criteria are cost-type
    (README §2, point 2). Here, f_star/f_minus are instead defined per-column
    using the direction flag, and the un-absolute-valued ratio is used
    directly — this generalizes correctly to mixed benefit/cost criteria
    while reducing to the same numbers on cost-only inputs.
    """
    _validate_directions(criteria, directions)
    matrix = build_decision_matrix(df, criteria)
    n = matrix.shape[1]
    weights = np.array(weights) if weights is not None else np.full(n, 1.0 / n)
    if len(weights) != n or not np.isclose(weights.sum(), 1.0):
        raise ValueError("`weights` must have one entry per criterion and sum to 1.")

    f_star = np.array([
        matrix[:, j].max() if directions[j] == "benefit" else matrix[:, j].min()
        for j in range(n)
    ])
    f_minus = np.array([
        matrix[:, j].min() if directions[j] == "benefit" else matrix[:, j].max()
        for j in range(n)
    ])

    span = f_star - f_minus
    span_safe = np.where(span == 0, 1e-9, span)
    ratio = weights * (f_star - matrix) / span_safe  # signed, not .abs()

    S = ratio.sum(axis=1)
    R = ratio.max(axis=1)

    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()

    Q = (
        v * (S - S_star) / (S_minus - S_star + 1e-9)
        + (1 - v) * (R - R_star) / (R_minus - R_star + 1e-9)
    )

    scores = pd.Series(Q, index=df.index, name="vikor_index")
    return RankingResult(scores=scores, best_index=scores.idxmin(), method="VIKOR")


# --------------------------------------------------------------------------- #
# Convenience: run all three and check for consensus (mirrors thesis §7.9 —
# "each of them returned an identical ranking output")
# --------------------------------------------------------------------------- #

def rank_all_methods(df: pd.DataFrame,
                      criteria: Sequence[str],
                      directions: Sequence[Direction],
                      weights: Sequence[float] | None = None,
                      v: float = 0.5) -> dict:
    """Run TOPSIS, WSM, and VIKOR on the same decision matrix and report
    whether all three converge on the same top-ranked candidate."""
    results = {
        "TOPSIS": topsis(df, criteria, directions, weights),
        "WSM": wsm(df, criteria, directions, weights),
        "VIKOR": vikor(df, criteria, directions, weights, v=v),
    }
    top_indices = {name: r.best_index for name, r in results.items()}
    consensus = len(set(top_indices.values())) == 1
    return {"results": results, "top_indices": top_indices, "consensus": consensus}
