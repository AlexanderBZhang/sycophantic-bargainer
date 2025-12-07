"""Metrics for evaluating bargaining outcomes."""

from .utility import (
    calculate_average_utility,
    calculate_agreement_rate,
    calculate_utility_gap,
    calculate_exploitability,
    calculate_payoff_statistics,
    calculate_efficiency,
)

from .fairness import (
    calculate_gini_coefficient,
    calculate_average_gini,
    calculate_nash_product,
    calculate_average_nash_product,
    calculate_min_max_ratio,
    calculate_average_min_max_ratio,
    calculate_fairness_score,
    calculate_average_fairness,
)

from .convergence import (
    calculate_rounds_to_agreement,
    calculate_average_rounds,
    calculate_outcome_variance,
    calculate_equilibrium_deviation,
    calculate_average_equilibrium_deviation,
    calculate_convergence_speed,
    calculate_consistency_score,
)

__all__ = [
    # Utility metrics
    "calculate_average_utility",
    "calculate_agreement_rate",
    "calculate_utility_gap",
    "calculate_exploitability",
    "calculate_payoff_statistics",
    "calculate_efficiency",
    # Fairness metrics
    "calculate_gini_coefficient",
    "calculate_average_gini",
    "calculate_nash_product",
    "calculate_average_nash_product",
    "calculate_min_max_ratio",
    "calculate_average_min_max_ratio",
    "calculate_fairness_score",
    "calculate_average_fairness",
    # Convergence metrics
    "calculate_rounds_to_agreement",
    "calculate_average_rounds",
    "calculate_outcome_variance",
    "calculate_equilibrium_deviation",
    "calculate_average_equilibrium_deviation",
    "calculate_convergence_speed",
    "calculate_consistency_score",
]
