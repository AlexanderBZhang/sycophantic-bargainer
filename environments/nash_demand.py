"""
Nash Demand Game implementation.

In this simultaneous-move bargaining game, both agents simultaneously demand
a portion of a fixed pie. If the sum of demands exceeds the pie size, both
receive nothing. Otherwise, both receive their demands.

Game-theoretic properties:
- Infinite Nash equilibria (any split where demands sum to pie size)
- Pareto optimal when demands sum exactly to pie size
- Tests coordination and fairness intuitions
"""

from typing import Dict, Any, Tuple, Optional
from .base import BargainEnvironment, GameState


class NashDemandGame(BargainEnvironment):
    """
    Nash Demand Game environment.

    Both players simultaneously demand a portion of the pie.
    - If demand_A + demand_B <= pie_size: Both get their demands
    - If demand_A + demand_B > pie_size: Both get 0 (disagreement)
    """

    def __init__(
        self,
        pie_size: float = 10.0,
        agent_ids: Tuple[str, str] = ("agent_A", "agent_B"),
        allow_fractional: bool = True,
    ):
        """
        Initialize Nash Demand Game.

        Args:
            pie_size: Total size of the pie to divide
            agent_ids: Tuple of (agent_A_id, agent_B_id)
            allow_fractional: Whether to allow fractional demands (True) or integers only (False)
        """
        super().__init__(agent_ids)
        self.pie_size = pie_size
        self.allow_fractional = allow_fractional
        self.demands: Dict[str, Optional[float]] = {self.agent_A: None, self.agent_B: None}
        self.payoffs: Dict[str, float] = {self.agent_A: 0.0, self.agent_B: 0.0}

    def reset(self) -> GameState:
        """Reset the game to initial state."""
        self.demands = {self.agent_A: None, self.agent_B: None}
        self.payoffs = {self.agent_A: 0.0, self.agent_B: 0.0}
        self.history = []

        self.state = GameState(
            round=0,
            is_terminal=False,
            observations={
                self.agent_A: self._make_observation(self.agent_A),
                self.agent_B: self._make_observation(self.agent_B),
            },
            metadata={
                "pie_size": self.pie_size,
                "allow_fractional": self.allow_fractional,
            },
        )

        return self.state

    def step(self, actions: Dict[str, float]) -> Tuple[GameState, Dict[str, float], bool]:
        """
        Execute one step (simultaneous demands).

        Args:
            actions: Dict mapping agent_id to demand amount
                    e.g., {"agent_A": 6.0, "agent_B": 4.0}

        Returns:
            Tuple of (new_state, rewards, done)
        """
        # Validate both actions
        for agent_id, demand in actions.items():
            is_valid, error_msg = self.validate_action(agent_id, demand)
            if not is_valid:
                raise ValueError(f"Invalid action for {agent_id}: {error_msg}")

        # Record demands
        self.demands[self.agent_A] = actions[self.agent_A]
        self.demands[self.agent_B] = actions[self.agent_B]

        # Calculate payoffs
        total_demand = self.demands[self.agent_A] + self.demands[self.agent_B]

        if total_demand <= self.pie_size:
            # Agreement: both receive their demands
            self.payoffs[self.agent_A] = self.demands[self.agent_A]
            self.payoffs[self.agent_B] = self.demands[self.agent_B]
            outcome = "agreement"
        else:
            # Disagreement: both receive 20.0 (Reservation Value)
            self.payoffs[self.agent_A] = 20.0
            self.payoffs[self.agent_B] = 20.0
            outcome = "disagreement"

        # Update state
        self.state = GameState(
            round=1,
            is_terminal=True,
            observations={
                self.agent_A: self._make_observation(self.agent_A, include_results=True),
                self.agent_B: self._make_observation(self.agent_B, include_results=True),
            },
            metadata={
                "pie_size": self.pie_size,
                "demands": self.demands.copy(),
                "total_demand": total_demand,
                "outcome": outcome,
            },
        )

        # Record to history
        self.history.append({
            "round": 1,
            "actions": actions.copy(),
            "demands": self.demands.copy(),
            "payoffs": self.payoffs.copy(),
            "outcome": outcome,
        })

        return self.state, self.payoffs.copy(), True

    def get_observation(self, agent_id: str) -> Dict[str, Any]:
        """Get current observation for an agent."""
        if self.state is None:
            raise RuntimeError("Game not initialized. Call reset() first.")
        return self.state.observations[agent_id]

    def _make_observation(self, agent_id: str, include_results: bool = False) -> Dict[str, Any]:
        """
        Create an observation dictionary for an agent.

        Args:
            agent_id: ID of the agent
            include_results: Whether to include game results (post-action)

        Returns:
            Observation dictionary
        """
        obs = {
            "agent_id": agent_id,
            "game_type": "nash_demand",
            "pie_size": self.pie_size,
            "allow_fractional": self.allow_fractional,
            "round": self.state.round if self.state else 0,
            "is_terminal": self.is_terminal(),
        }

        if include_results:
            obs["my_demand"] = self.demands[agent_id]
            obs["opponent_demand"] = self.demands[self._get_opponent(agent_id)]
            obs["my_payoff"] = self.payoffs[agent_id]
            obs["opponent_payoff"] = self.payoffs[self._get_opponent(agent_id)]
            obs["outcome"] = self.state.metadata.get("outcome")

        return obs

    def get_rewards(self) -> Dict[str, float]:
        """Get final payoffs for both agents."""
        return self.payoffs.copy()

    def validate_action(self, agent_id: str, action: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate an agent's demand.

        Args:
            agent_id: ID of the agent
            action: Demand amount (float or int)

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if agent exists
        if agent_id not in self.agent_ids:
            return False, f"Unknown agent: {agent_id}"

        # Check type
        if not isinstance(action, (int, float)):
            return False, f"Demand must be a number, got {type(action)}"

        demand = float(action)

        # Check if fractional demands are allowed
        if not self.allow_fractional and demand != int(demand):
            return False, f"Fractional demands not allowed, got {demand}"

        # Check range
        if demand < 0:
            return False, f"Demand must be non-negative, got {demand}"

        if demand > self.pie_size:
            return False, f"Demand {demand} exceeds pie size {self.pie_size}"

        return True, None

    def _get_opponent(self, agent_id: str) -> str:
        """Get the opponent's agent_id."""
        return self.agent_B if agent_id == self.agent_A else self.agent_A

    def render(self) -> str:
        """Render the game state as a human-readable string."""
        if self.state is None:
            return "Nash Demand Game (not started)\nPie size: {}\n".format(self.pie_size)

        base = f"=== Nash Demand Game ===\n"
        base += f"Pie size: {self.pie_size}\n"

        if not self.is_terminal():
            base += "Status: Waiting for demands\n"
        else:
            base += f"\n{self.agent_A} demands: {self.demands[self.agent_A]}\n"
            base += f"{self.agent_B} demands: {self.demands[self.agent_B]}\n"
            base += f"Total demand: {self.demands[self.agent_A] + self.demands[self.agent_B]}\n"
            base += f"\nOutcome: {self.state.metadata.get('outcome')}\n"
            base += f"{self.agent_A} payoff: {self.payoffs[self.agent_A]}\n"
            base += f"{self.agent_B} payoff: {self.payoffs[self.agent_B]}\n"

        return base

    def get_equilibrium_prediction(self) -> Dict[str, Any]:
        """
        Return game-theoretic equilibrium prediction.

        Nash Demand Game has infinite Nash equilibria - any split
        where demands sum exactly to pie_size is an equilibrium.

        Returns:
            Dictionary with equilibrium information
        """
        return {
            "equilibrium_type": "infinite_nash_equilibria",
            "description": "Any (x, y) where x + y = pie_size and x, y >= 0",
            "focal_point": {"agent_A": self.pie_size / 2, "agent_B": self.pie_size / 2},
            "focal_point_justification": "Equal split (fairness focal point)",
        }
