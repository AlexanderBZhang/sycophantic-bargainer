"""
Tests for bargaining environments.
"""

import pytest
from environments import (
    NashDemandGame,
    UltimatumGame,
    RubinsteinBargaining,
)


class TestNashDemandGame:
    """Tests for Nash Demand Game."""

    def test_initialization(self):
        """Test game initializes correctly."""
        env = NashDemandGame(pie_size=10.0)
        assert env.pie_size == 10.0
        assert env.agent_A == "agent_A"
        assert env.agent_B == "agent_B"

    def test_reset(self):
        """Test game reset."""
        env = NashDemandGame(pie_size=10.0)
        state = env.reset()

        assert state.round == 0
        assert not state.is_terminal
        assert env.demands["agent_A"] is None
        assert env.demands["agent_B"] is None

    def test_agreement(self):
        """Test agreement outcome."""
        env = NashDemandGame(pie_size=10.0)
        env.reset()

        # Both demand within pie size
        actions = {"agent_A": 6.0, "agent_B": 4.0}
        state, rewards, done = env.step(actions)

        assert done
        assert rewards["agent_A"] == 6.0
        assert rewards["agent_B"] == 4.0

    def test_disagreement(self):
        """Test disagreement outcome."""
        env = NashDemandGame(pie_size=10.0)
        env.reset()

        # Demands exceed pie size
        actions = {"agent_A": 7.0, "agent_B": 5.0}
        state, rewards, done = env.step(actions)

        assert done
        assert rewards["agent_A"] == 0.0
        assert rewards["agent_B"] == 0.0

    def test_validation(self):
        """Test action validation."""
        env = NashDemandGame(pie_size=10.0)
        env.reset()

        # Valid action
        is_valid, error = env.validate_action("agent_A", 5.0)
        assert is_valid
        assert error is None

        # Negative demand
        is_valid, error = env.validate_action("agent_A", -1.0)
        assert not is_valid

        # Exceeds pie size
        is_valid, error = env.validate_action("agent_A", 15.0)
        assert not is_valid


class TestUltimatumGame:
    """Tests for Ultimatum Game."""

    def test_initialization(self):
        """Test game initializes correctly."""
        env = UltimatumGame(pie_size=10.0)
        assert env.pie_size == 10.0
        assert env.proposer_id == "agent_A"
        assert env.responder_id == "agent_B"

    def test_reset(self):
        """Test game reset."""
        env = UltimatumGame(pie_size=10.0)
        state = env.reset()

        assert state.round == 0
        assert not state.is_terminal
        assert env.current_phase == "proposal"

    def test_accept_offer(self):
        """Test accepted offer."""
        env = UltimatumGame(pie_size=10.0)
        env.reset()

        # Proposer makes offer
        state, rewards, done = env.step({"agent_A": 6.0})
        assert not done
        assert env.current_phase == "response"

        # Responder accepts
        state, rewards, done = env.step({"agent_B": True})
        assert done
        assert rewards["agent_A"] == 6.0
        assert rewards["agent_B"] == 4.0

    def test_reject_offer(self):
        """Test rejected offer."""
        env = UltimatumGame(pie_size=10.0)
        env.reset()

        # Proposer makes offer
        env.step({"agent_A": 9.0})

        # Responder rejects
        state, rewards, done = env.step({"agent_B": False})
        assert done
        assert rewards["agent_A"] == 0.0
        assert rewards["agent_B"] == 0.0


class TestRubinsteinBargaining:
    """Tests for Rubinstein Bargaining."""

    def test_initialization(self):
        """Test game initializes correctly."""
        env = RubinsteinBargaining(pie_size=10.0, discount_factor=0.9)
        assert env.pie_size == 10.0
        assert env.discount_factor == 0.9
        assert env.max_rounds == 10

    def test_reset(self):
        """Test game reset."""
        env = RubinsteinBargaining(pie_size=10.0)
        state = env.reset()

        assert state.round == 1
        assert not state.is_terminal
        assert env.current_proposer_id == "agent_A"
        assert env.waiting_for == "offer"

    def test_immediate_agreement(self):
        """Test immediate agreement in round 1."""
        env = RubinsteinBargaining(pie_size=10.0, discount_factor=0.9)
        env.reset()

        # Agent A makes offer
        state, rewards, done = env.step({"agent_A": 6.0})
        assert not done

        # Agent B accepts
        state, rewards, done = env.step({"agent_B": True})
        assert done
        assert rewards["agent_A"] == 6.0
        assert rewards["agent_B"] == 4.0

    def test_rejection_and_counteroffer(self):
        """Test rejection and continuation."""
        env = RubinsteinBargaining(pie_size=10.0, discount_factor=0.9)
        env.reset()

        # Agent A makes offer
        env.step({"agent_A": 8.0})

        # Agent B rejects
        state, rewards, done = env.step({"agent_B": False})
        assert not done
        assert env.current_round == 2
        assert env.current_proposer_id == "agent_B"

    def test_discount_applied(self):
        """Test discounting in later rounds."""
        env = RubinsteinBargaining(pie_size=10.0, discount_factor=0.9)
        env.reset()

        # Round 1: A offers, B rejects
        env.step({"agent_A": 9.0})
        env.step({"agent_B": False})

        # Round 2: B offers, A accepts
        env.step({"agent_B": 5.0})
        state, rewards, done = env.step({"agent_A": True})

        assert done
        # Payoffs should be discounted by 0.9
        assert rewards["agent_B"] == pytest.approx(5.0 * 0.9)
        assert rewards["agent_A"] == pytest.approx(5.0 * 0.9)
