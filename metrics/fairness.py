"""
Fairness metrics for evaluating bargaining outcomes.

Measures inequality, balance, and equity in payoff distributions.
"""

import numpy as np
from typing import List, Dict, Any


def calculate_gini_coefficient(payoffs: List[float]) -> float:
    """
    Calculate Gini coefficient for a distribution of payoffs.

    Gini coefficient measures inequality:
    - 0 = perfect equality (everyone gets the same)
    - 1 = maximal inequality (one person gets everything)

    Args:
        payoffs: List of payoff values

    Returns:
        Gini coefficient (0 to 1)
    """
    if not payoffs or len(payoffs) == 0:
        return 0.0

    # Convert to numpy array (keep zeros for accurate inequality measure)
    payoffs_array = np.array(payoffs)

    # If all payoffs are zero, return 0
    if np.all(payoffs_array == 0):
        return 0.0

    # Sort payoffs
    sorted_payoffs = np.sort(payoffs_array)
    n = len(sorted_payoffs)

    # Calculate Gini
    cumsum = np.cumsum(sorted_payoffs)
    total = cumsum[-1]

    if total == 0:
        return 0.0

    # Gini formula
    gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_payoffs)) / (n * total) - (n + 1) / n

    return float(gini)


def calculate_game_gini(result: Dict[str, Any]) -> float:
    """
    Calculate Gini coefficient for a single game's payoffs.

    Args:
        result: Game result with 'payoffs' dict

    Returns:
        Gini coefficient
    """
    payoffs = list(result.get("payoffs", {}).values())
    return calculate_gini_coefficient(payoffs)


def calculate_average_gini(results: List[Dict[str, Any]]) -> float:
    """
    Calculate average Gini coefficient across games.

    Args:
        results: List of game results

    Returns:
        Average Gini coefficient
    """
    if not results:
        return 0.0

    ginis = [calculate_game_gini(r) for r in results]
    return float(np.mean(ginis))


def calculate_nash_product(result: Dict[str, Any]) -> float:
    """
    Calculate Nash product: product of payoffs.

    Higher Nash product indicates more efficient/fair outcomes.
    Nash bargaining solution maximizes this.

    Args:
        result: Game result with 'payoffs' dict

    Returns:
        Nash product
    """
    payoffs = list(result.get("payoffs", {}).values())

    if len(payoffs) != 2:
        return 0.0

    return float(payoffs[0] * payoffs[1])


def calculate_average_nash_product(results: List[Dict[str, Any]]) -> float:
    """
    Calculate average Nash product across games.

    Args:
        results: List of game results

    Returns:
        Average Nash product
    """
    if not results:
        return 0.0

    products = [calculate_nash_product(r) for r in results]
    return float(np.mean(products))


def calculate_min_max_ratio(result: Dict[str, Any]) -> float:
    """
    Calculate min-max ratio: min(payoffs) / max(payoffs).

    Closer to 1 = more balanced split.
    Closer to 0 = more unequal split.

    Args:
        result: Game result with 'payoffs' dict

    Returns:
        Min-max ratio (0 to 1)
    """
    payoffs = list(result.get("payoffs", {}).values())

    if not payoffs:
        return 0.0

    max_payoff = max(payoffs)

    if max_payoff == 0:
        return 0.0

    min_payoff = min(payoffs)

    return float(min_payoff / max_payoff)


def calculate_average_min_max_ratio(results: List[Dict[str, Any]]) -> float:
    """
    Calculate average min-max ratio across games.

    Args:
        results: List of game results

    Returns:
        Average min-max ratio
    """
    if not results:
        return 0.0

    ratios = [calculate_min_max_ratio(r) for r in results]
    return float(np.mean(ratios))


def calculate_fairness_score(result: Dict[str, Any], pie_size: float) -> float:
    """
    Calculate composite fairness score (0 to 1).

    Combines multiple fairness metrics:
    - Low Gini (low inequality)
    - High Nash product (efficiency)
    - High min-max ratio (balance)

    Args:
        result: Game result
        pie_size: Total pie size

    Returns:
        Fairness score (0 to 1, higher = more fair)
    """
    gini = calculate_game_gini(result)
    nash_product = calculate_nash_product(result)
    min_max_ratio = calculate_min_max_ratio(result)

    # Normalize Nash product by max possible (pie_size/2 * pie_size/2)
    max_nash = (pie_size / 2) ** 2
    normalized_nash = nash_product / max_nash if max_nash > 0 else 0

    # Combine: lower gini is better, higher nash and min_max is better
    fairness = (1 - gini) * 0.3 + normalized_nash * 0.3 + min_max_ratio * 0.4

    return float(fairness)


def calculate_average_fairness(results: List[Dict[str, Any]], pie_size: float) -> float:
    """
    Calculate average fairness score across games.

    Args:
        results: List of game results
        pie_size: Total pie size

    Returns:
        Average fairness score
    """
    if not results:
        return 0.0

    scores = [calculate_fairness_score(r, pie_size) for r in results]
    return float(np.mean(scores))
