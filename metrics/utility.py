"""
Utility metrics for evaluating bargaining outcomes.

Measures payoffs, agreement rates, and exploitability.
"""

import numpy as np
from typing import List, Dict, Any, Tuple


def calculate_average_utility(results: List[Dict[str, Any]], agent_id: str) -> float:
    """
    Calculate average utility for an agent across multiple games.

    Args:
        results: List of game results, each containing 'payoffs' dict
        agent_id: ID of the agent to calculate utility for

    Returns:
        Average payoff
    """
    payoffs = [result["payoffs"][agent_id] for result in results if agent_id in result.get("payoffs", {})]

    if not payoffs:
        return 0.0

    return float(np.mean(payoffs))


def calculate_agreement_rate(results: List[Dict[str, Any]]) -> float:
    """
    Calculate proportion of games that ended in agreement (not (0,0)).

    Args:
        results: List of game results

    Returns:
        Agreement rate (0 to 1)
    """
    if not results:
        return 0.0

    agreements = 0
    for result in results:
        payoffs = result.get("payoffs", {})
        # Agreement if any agent got non-zero payoff
        if any(v > 0 for v in payoffs.values()):
            agreements += 1

    return agreements / len(results)


def calculate_utility_gap(
    results: List[Dict[str, Any]],
    agent_A_id: str,
    agent_B_id: str,
) -> float:
    """
    Calculate utility gap between two agents.

    Positive value means agent_A got more on average.
    Negative value means agent_B got more on average.

    Args:
        results: List of game results
        agent_A_id: ID of first agent
        agent_B_id: ID of second agent

    Returns:
        Utility gap (agent_A_avg - agent_B_avg)
    """
    avg_A = calculate_average_utility(results, agent_A_id)
    avg_B = calculate_average_utility(results, agent_B_id)

    return avg_A - avg_B


def calculate_exploitability(
    same_matchup_results: List[Dict[str, Any]],
    mixed_matchup_results: List[Dict[str, Any]],
    agent_id: str,
) -> float:
    """
    Calculate exploitability: how much worse an agent performs against
    rational opponents compared to playing against itself.

    Args:
        same_matchup_results: Results from (agent, agent) matchups
        mixed_matchup_results: Results from (agent, rational) matchups
        agent_id: ID of the agent to measure

    Returns:
        Exploitability (positive = worse against rational)
    """
    avg_same = calculate_average_utility(same_matchup_results, agent_id)
    avg_mixed = calculate_average_utility(mixed_matchup_results, agent_id)

    return avg_same - avg_mixed


def calculate_payoff_statistics(results: List[Dict[str, Any]], agent_id: str) -> Dict[str, float]:
    """
    Calculate comprehensive payoff statistics for an agent.

    Args:
        results: List of game results
        agent_id: ID of the agent

    Returns:
        Dictionary with mean, std, min, max, median
    """
    payoffs = [result["payoffs"][agent_id] for result in results if agent_id in result.get("payoffs", {})]

    if not payoffs:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}

    payoffs_array = np.array(payoffs)

    return {
        "mean": float(np.mean(payoffs_array)),
        "std": float(np.std(payoffs_array)),
        "min": float(np.min(payoffs_array)),
        "max": float(np.max(payoffs_array)),
        "median": float(np.median(payoffs_array)),
    }


def calculate_total_surplus(result: Dict[str, Any]) -> float:
    """
    Calculate total surplus (sum of payoffs) for a single game.

    Args:
        result: Single game result

    Returns:
        Total surplus
    """
    payoffs = result.get("payoffs", {})
    return sum(payoffs.values())


def calculate_average_surplus(results: List[Dict[str, Any]]) -> float:
    """
    Calculate average total surplus across games.

    Args:
        results: List of game results

    Returns:
        Average total surplus
    """
    if not results:
        return 0.0

    surpluses = [calculate_total_surplus(r) for r in results]
    return float(np.mean(surpluses))


def calculate_efficiency(results: List[Dict[str, Any]], max_pie_size: float) -> float:
    """
    Calculate efficiency: average surplus / max possible surplus.

    Args:
        results: List of game results
        max_pie_size: Maximum possible pie size

    Returns:
        Efficiency ratio (0 to 1)
    """
    if max_pie_size == 0:
        return 0.0

    avg_surplus = calculate_average_surplus(results)
    return avg_surplus / max_pie_size
