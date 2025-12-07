"""
Base agent interface for bargaining games.

All agents must implement the Agent interface to participate in bargaining games.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Agent(ABC):
    """
    Abstract base class for bargaining agents.

    All agents must implement choose_action() to select actions based on
    observations from the environment.
    """

    def __init__(self, agent_id: str, persona_type: Optional[str] = None):
        """
        Initialize the agent.

        Args:
            agent_id: Unique identifier for this agent
            persona_type: Type of persona (e.g., 'sycophantic', 'rational', 'control')
        """
        self.agent_id = agent_id
        self.persona_type = persona_type
        self.history: list = []

    @abstractmethod
    def choose_action(self, observation: Dict[str, Any]) -> Any:
        """
        Choose an action based on the current observation.

        Args:
            observation: Current game state observation from environment

        Returns:
            Action to take (type depends on environment)
        """
        pass

    def reset(self):
        """Reset the agent's internal state for a new game."""
        self.history = []

    def update_history(self, observation: Dict[str, Any], action: Any, reward: Optional[float] = None):
        """
        Update the agent's history with the latest interaction.

        Args:
            observation: The observation received
            action: The action taken
            reward: The reward received (if any)
        """
        self.history.append({
            "observation": observation,
            "action": action,
            "reward": reward,
        })

    def get_info(self) -> Dict[str, Any]:
        """
        Get metadata about the agent.

        Returns:
            Dictionary with agent information
        """
        return {
            "agent_id": self.agent_id,
            "persona_type": self.persona_type,
            "history_length": len(self.history),
        }
