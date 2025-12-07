"""
Convergence metrics for evaluating bargaining dynamics.

Measures speed of agreement, stability, and deviation from equilibria.
"""

import numpy as np
from typing import List, Dict, Any, Optional


def calculate_rounds_to_agreement(result: Dict[str, Any]) -> Optional[int]:
    """
    Calculate number of rounds until agreement.

    For multi-round games (Rubinstein), returns the round number.
    For single-round games (Nash, Ultimatum), returns 1 or None.

    Args:
        result: Game result

    Returns:
        Number of rounds (None if no agreement)
    """
    metadata = result.get("metadata", {})
    outcome = metadata.get("outcome")

    if outcome == "disagreement":
        return None

    # For Rubinstein
    if "final_round" in metadata:
        return metadata["final_round"]

    # For other games (single round)
    return 1


def calculate_average_rounds(results: List[Dict[str, Any]]) -> float:
    """
    Calculate average rounds to agreement (excluding disagreements).

    Args:
        results: List of game results

    Returns:
        Average rounds to agreement
    """
    rounds = [calculate_rounds_to_agreement(r) for r in results]
    rounds_valid = [r for r in rounds if r is not None]

    if not rounds_valid:
        return 0.0

    return float(np.mean(rounds_valid))


def calculate_outcome_variance(results: List[Dict[str, Any]], agent_id: str) -> float:
    """
    Calculate variance in outcomes for an agent.

    Lower variance = more consistent/stable behavior.

    Args:
        results: List of game results
        agent_id: ID of the agent

    Returns:
        Variance of payoffs
    """
    payoffs = [result["payoffs"][agent_id] for result in results if agent_id in result.get("payoffs", {})]

    if len(payoffs) < 2:
        return 0.0

    return float(np.var(payoffs))


def calculate_equilibrium_deviation(
    result: Dict[str, Any],
    predicted_payoffs: Dict[str, float],
) -> float:
    """
    Calculate deviation from game-theoretic equilibrium prediction.

    Uses Euclidean distance between actual and predicted payoffs.

    Args:
        result: Game result with actual payoffs
        predicted_payoffs: Predicted equilibrium payoffs

    Returns:
        Euclidean distance from equilibrium
    """
    actual = result.get("payoffs", {})

    if not actual or not predicted_payoffs:
        return 0.0

    # Ensure same agents
    agents = set(actual.keys()) & set(predicted_payoffs.keys())

    if not agents:
        return 0.0

    # Calculate Euclidean distance
    squared_diffs = [(actual[a] - predicted_payoffs[a]) ** 2 for a in agents]
    distance = np.sqrt(sum(squared_diffs))

    return float(distance)


def calculate_average_equilibrium_deviation(
    results: List[Dict[str, Any]],
    predicted_payoffs: Dict[str, float],
) -> float:
    """
    Calculate average equilibrium deviation across games.

    Args:
        results: List of game results
        predicted_payoffs: Predicted equilibrium payoffs

    Returns:
        Average deviation from equilibrium
    """
    if not results:
        return 0.0

    deviations = [calculate_equilibrium_deviation(r, predicted_payoffs) for r in results]
    return float(np.mean(deviations))


def calculate_convergence_speed(results: List[Dict[str, Any]]) -> float:
    """
    Calculate convergence speed metric.

    For Rubinstein: 1 / average_rounds (faster = higher score)
    For single-round games: agreement_rate

    Args:
        results: List of game results

    Returns:
        Convergence speed score (higher = faster convergence)
    """
    avg_rounds = calculate_average_rounds(results)

    if avg_rounds == 0:
        return 0.0

    return 1.0 / avg_rounds


def calculate_consistency_score(results: List[Dict[str, Any]], agent_id: str) -> float:
    """
    Calculate consistency score: inverse of coefficient of variation.

    Higher score = more consistent/predictable behavior.

    Args:
        results: List of game results
        agent_id: ID of the agent

    Returns:
        Consistency score (0 to 1)
    """
    payoffs = [result["payoffs"][agent_id] for result in results if agent_id in result.get("payoffs", {})]

    if len(payoffs) < 2:
        return 1.0  # Fully consistent (single observation)

    mean = np.mean(payoffs)
    std = np.std(payoffs)

    if mean == 0:
        return 0.0

    # Coefficient of variation
    cv = std / mean

    # Consistency = 1 / (1 + cv)
    # Maps CV [0, ∞) to consistency [1, 0]
    consistency = 1.0 / (1.0 + cv)

    return float(consistency)
