"""
Tests for metrics calculations.
"""

import pytest
from metrics import (
    calculate_average_utility,
    calculate_agreement_rate,
    calculate_utility_gap,
    calculate_gini_coefficient,
    calculate_nash_product,
    calculate_min_max_ratio,
)


class TestUtilityMetrics:
    """Tests for utility metrics."""

    def test_average_utility(self):
        """Test average utility calculation."""
        results = [
            {"payoffs": {"agent_A": 5.0, "agent_B": 5.0}},
            {"payoffs": {"agent_A": 6.0, "agent_B": 4.0}},
            {"payoffs": {"agent_A": 7.0, "agent_B": 3.0}},
        ]

        avg_A = calculate_average_utility(results, "agent_A")
        avg_B = calculate_average_utility(results, "agent_B")

        assert avg_A == 6.0
        assert avg_B == 4.0

    def test_agreement_rate(self):
        """Test agreement rate calculation."""
        results = [
            {"payoffs": {"agent_A": 5.0, "agent_B": 5.0}},  # Agreement
            {"payoffs": {"agent_A": 0.0, "agent_B": 0.0}},  # Disagreement
            {"payoffs": {"agent_A": 6.0, "agent_B": 4.0}},  # Agreement
        ]

        rate = calculate_agreement_rate(results)
        assert rate == pytest.approx(2.0 / 3.0)

    def test_utility_gap(self):
        """Test utility gap calculation."""
        results = [
            {"payoffs": {"agent_A": 7.0, "agent_B": 3.0}},
            {"payoffs": {"agent_A": 8.0, "agent_B": 2.0}},
        ]

        gap = calculate_utility_gap(results, "agent_A", "agent_B")
        assert gap == 5.0  # (7.5 - 2.5)


class TestFairnessMetrics:
    """Tests for fairness metrics."""

    def test_gini_coefficient_equal(self):
        """Test Gini for equal distribution."""
        payoffs = [5.0, 5.0]
        gini = calculate_gini_coefficient(payoffs)

        assert gini == pytest.approx(0.0, abs=0.01)

    def test_gini_coefficient_unequal(self):
        """Test Gini for unequal distribution."""
        payoffs = [10.0, 0.0]
        gini = calculate_gini_coefficient(payoffs)

        # Should be positive (inequality)
        assert gini > 0

    def test_nash_product(self):
        """Test Nash product calculation."""
        result = {"payoffs": {"agent_A": 6.0, "agent_B": 4.0}}
        product = calculate_nash_product(result)

        assert product == 24.0

    def test_min_max_ratio(self):
        """Test min-max ratio."""
        result = {"payoffs": {"agent_A": 8.0, "agent_B": 2.0}}
        ratio = calculate_min_max_ratio(result)

        assert ratio == 0.25  # 2.0 / 8.0
