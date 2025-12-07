"""
Base environment class for bargaining games.

All bargaining environments inherit from BargainEnvironment and implement
the standard interface for multi-agent games.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GameState:
    """Represents the current state of a bargaining game."""
    round: int
    is_terminal: bool
    observations: Dict[str, Any]
    metadata: Dict[str, Any]


class BargainEnvironment(ABC):
    """
    Abstract base class for two-player bargaining environments.

    All environments must implement:
    - reset(): Initialize a new game
    - step(): Process actions and update state
    - get_observation(): Return current state for agents
    - is_terminal(): Check if game has ended
    - get_rewards(): Return payoffs for both agents
    """

    def __init__(self, agent_ids: Tuple[str, str] = ("agent_A", "agent_B")):
        """
        Initialize the environment.

        Args:
            agent_ids: Tuple of (agent_A_id, agent_B_id) identifiers
        """
        self.agent_ids = agent_ids
        self.agent_A, self.agent_B = agent_ids
        self.state: Optional[GameState] = None
        self.history: list = []

    @abstractmethod
    def reset(self) -> GameState:
        """
        Reset the environment to initial state.

        Returns:
            Initial game state
        """
        pass

    @abstractmethod
    def step(self, actions: Dict[str, Any]) -> Tuple[GameState, Dict[str, float], bool]:
        """
        Execute one step of the game.

        Args:
            actions: Dictionary mapping agent_id to action
                    Format depends on specific game (e.g., {"agent_A": 5, "agent_B": 4})

        Returns:
            Tuple of (new_state, rewards, done)
            - new_state: Updated GameState
            - rewards: Dict mapping agent_id to payoff (float)
            - done: Boolean indicating if game is terminal
        """
        pass

    @abstractmethod
    def get_observation(self, agent_id: str) -> Dict[str, Any]:
        """
        Get the current observation for a specific agent.

        Args:
            agent_id: ID of the agent requesting observation

        Returns:
            Dictionary containing the agent's view of the game state
        """
        pass

    def is_terminal(self) -> bool:
        """
        Check if the game has reached a terminal state.

        Returns:
            True if game is over, False otherwise
        """
        return self.state is not None and self.state.is_terminal

    @abstractmethod
    def get_rewards(self) -> Dict[str, float]:
        """
        Get the final rewards for both agents.

        Returns:
            Dictionary mapping agent_id to payoff
        """
        pass

    @abstractmethod
    def validate_action(self, agent_id: str, action: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate an agent's action.

        Args:
            agent_id: ID of the agent taking the action
            action: The action to validate

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if action is legal
            - error_message: Description of why action is invalid (None if valid)
        """
        pass

    def get_game_info(self) -> Dict[str, Any]:
        """
        Get metadata about the current game state.

        Returns:
            Dictionary with game information (round number, history, etc.)
        """
        return {
            "round": self.state.round if self.state else 0,
            "is_terminal": self.is_terminal(),
            "history_length": len(self.history),
            "agent_ids": self.agent_ids,
        }

    def render(self) -> str:
        """
        Render the current game state as a human-readable string.

        Returns:
            String representation of the game state
        """
        if self.state is None:
            return "Game not started. Call reset() first."

        info = self.get_game_info()
        return f"Round {info['round']} | Terminal: {info['is_terminal']}"
