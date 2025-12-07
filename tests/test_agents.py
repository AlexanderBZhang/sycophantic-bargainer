"""
Tests for agent implementations.
"""

import pytest
from agents import MockAgent
from agents.personas import get_persona, get_system_prompt


class TestMockAgent:
    """Tests for MockAgent."""

    def test_initialization(self):
        """Test agent initializes correctly."""
        agent = MockAgent(
            agent_id="test_agent",
            persona_type="sycophantic",
            strategy="equal_split",
        )

        assert agent.agent_id == "test_agent"
        assert agent.persona_type == "sycophantic"
        assert agent.strategy == "equal_split"

    def test_equal_split_strategy(self):
        """Test equal split strategy."""
        agent = MockAgent(
            agent_id="test_agent",
            persona_type="control",
            strategy="equal_split",
        )

        obs = {
            "game_type": "nash_demand",
            "pie_size": 10.0,
            "is_terminal": False,
        }

        action = agent.choose_action(obs)
        assert action == 5.0

    def test_greedy_strategy(self):
        """Test greedy strategy."""
        agent = MockAgent(
            agent_id="test_agent",
            persona_type="rational",
            strategy="greedy",
        )

        obs = {
            "game_type": "nash_demand",
            "pie_size": 10.0,
            "is_terminal": False,
        }

        action = agent.choose_action(obs)
        assert action == 7.0  # 70% of pie

    def test_fixed_action(self):
        """Test fixed action."""
        agent = MockAgent(
            agent_id="test_agent",
            persona_type="test",
            fixed_action=3.5,
        )

        obs = {"game_type": "nash_demand", "pie_size": 10.0}
        action = agent.choose_action(obs)
        assert action == 3.5

    def test_ultimatum_response(self):
        """Test ultimatum game response."""
        agent = MockAgent(
            agent_id="test_agent",
            persona_type="rational",
            strategy="equal_split",
        )

        # Fair offer - should accept
        obs = {
            "game_type": "ultimatum",
            "role": "responder",
            "pie_size": 10.0,
            "offer": 5.0,
        }
        action = agent.choose_action(obs)
        assert action == True

        # Unfair offer - should reject
        obs["offer"] = 9.0  # Only gets 1.0
        action = agent.choose_action(obs)
        assert action == False


class TestPersonas:
    """Tests for persona system."""

    def test_get_persona(self):
        """Test getting persona config."""
        persona = get_persona("sycophantic")

        assert "name" in persona
        assert "instructions" in persona  # Updated: was 'system_prompt'
        assert "traits" in persona

    def test_get_system_prompt(self):
        """Test getting system prompt."""
        # Default: no persona injection
        prompt = get_system_prompt("player_1", pot_size=100.0, max_rounds=10)

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "Alpha Partner" in prompt  # Role should be present
        
    def test_get_system_prompt_with_persona(self):
        """Test getting system prompt with persona injection."""
        prompt = get_system_prompt("player_1", pot_size=100.0, max_rounds=10, use_persona=True, persona_type="rational")
        
        assert isinstance(prompt, str)
        assert "Alpha Partner" in prompt
        assert "payoff" in prompt.lower()  # Rational persona mentions payoff

    def test_invalid_persona(self):
        """Test invalid persona type."""
        with pytest.raises(ValueError):
            get_persona("invalid_persona_type")
