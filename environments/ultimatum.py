"""
Ultimatum Game implementation.

In this sequential bargaining game, one agent (Proposer) makes an offer to
split a fixed pie. The other agent (Responder) can accept or reject.
- If accepted: Split as proposed
- If rejected: Both get 0

Game-theoretic properties:
- Subgame perfect equilibrium: Proposer offers minimum, Responder accepts
- Tests fairness norms vs. rational self-interest
- Asymmetric roles create power dynamics
"""

from typing import Dict, Any, Tuple, Optional, Literal
from .base import BargainEnvironment, GameState


class UltimatumGame(BargainEnvironment):
    """
    Ultimatum Game environment.

    Sequential game:
    1. Proposer offers a split (x, pie_size - x)
    2. Responder accepts or rejects
    3. If accept: payoffs = (x, pie_size - x)
       If reject: payoffs = (0, 0)
    """

    def __init__(
        self,
        pie_size: float = 100.0,
        agent_ids: Tuple[str, str] = ("agent_A", "agent_B"),
        proposer_id: Optional[str] = None,
        allow_fractional: bool = True,
    ):
        """
        Initialize Ultimatum Game.

        Args:
            pie_size: Total size of the pie to divide
            agent_ids: Tuple of (agent_A_id, agent_B_id)
            proposer_id: ID of proposer (if None, agent_A is proposer)
            allow_fractional: Whether to allow fractional offers
        """
        super().__init__(agent_ids)
        self.pie_size = pie_size
        self.allow_fractional = allow_fractional

        # Determine roles
        if proposer_id is None:
            self.proposer_id = self.agent_A
        else:
            if proposer_id not in agent_ids:
                raise ValueError(f"proposer_id {proposer_id} not in agent_ids {agent_ids}")
            self.proposer_id = proposer_id

        self.responder_id = self._get_opponent(self.proposer_id)

        # Game state
        self.offer: Optional[float] = None  # Proposer's offer (amount for themselves)
        self.response: Optional[bool] = None  # Responder's decision (True = accept)
        self.payoffs: Dict[str, float] = {self.agent_A: 0.0, self.agent_B: 0.0}
        self.current_phase: Literal["proposal", "response", "complete"] = "proposal"

    def reset(self) -> GameState:
        """Reset the game to initial state."""
        self.offer = None
        self.response = None
        self.payoffs = {self.agent_A: 0.0, self.agent_B: 0.0}
        self.current_phase = "proposal"
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
                "proposer_id": self.proposer_id,
                "responder_id": self.responder_id,
                "current_phase": self.current_phase,
            },
        )

        return self.state

    def step(self, actions: Dict[str, Any]) -> Tuple[GameState, Dict[str, float], bool]:
        """
        Execute one step of the game.

        In proposal phase: expects {proposer_id: offer_amount}
        In response phase: expects {responder_id: True/False}

        Args:
            actions: Dict with single entry for current acting agent

        Returns:
            Tuple of (new_state, rewards, done)
        """
        if self.current_phase == "proposal":
            return self._step_proposal(actions)
        elif self.current_phase == "response":
            return self._step_response(actions)
        else:
            raise RuntimeError("Game already complete")

    def _step_proposal(self, actions: Dict[str, Any]) -> Tuple[GameState, Dict[str, float], bool]:
        """Handle proposal phase."""
        if self.proposer_id not in actions:
            raise ValueError(f"Expected action from proposer {self.proposer_id}")

        offer = actions[self.proposer_id]

        # Validate
        is_valid, error_msg = self.validate_action(self.proposer_id, offer)
        if not is_valid:
            raise ValueError(f"Invalid offer from {self.proposer_id}: {error_msg}")

        self.offer = float(offer)
        self.current_phase = "response"

        # Update state
        self.state = GameState(
            round=1,
            is_terminal=False,
            observations={
                self.agent_A: self._make_observation(self.agent_A),
                self.agent_B: self._make_observation(self.agent_B),
            },
            metadata={
                "pie_size": self.pie_size,
                "proposer_id": self.proposer_id,
                "responder_id": self.responder_id,
                "current_phase": self.current_phase,
                "offer": self.offer,
            },
        )

        # Not terminal yet, no rewards
        return self.state, {self.agent_A: 0.0, self.agent_B: 0.0}, False

    def _step_response(self, actions: Dict[str, Any]) -> Tuple[GameState, Dict[str, float], bool]:
        """Handle response phase."""
        if self.responder_id not in actions:
            raise ValueError(f"Expected action from responder {self.responder_id}")

        response = actions[self.responder_id]

        # Validate
        is_valid, error_msg = self.validate_action(self.responder_id, response)
        if not is_valid:
            raise ValueError(f"Invalid response from {self.responder_id}: {error_msg}")

        self.response = bool(response)
        self.current_phase = "complete"

        # Calculate payoffs
        if self.response:
            # Accept: proposer gets offer, responder gets remainder
            self.payoffs[self.proposer_id] = self.offer
            self.payoffs[self.responder_id] = self.pie_size - self.offer
            outcome = "accept"
        else:
            # Reject: both get 20.0 (Reservation Value)
            self.payoffs[self.proposer_id] = 20.0
            self.payoffs[self.responder_id] = 20.0
            outcome = "reject"

        # Update state
        self.state = GameState(
            round=2,
            is_terminal=True,
            observations={
                self.agent_A: self._make_observation(self.agent_A),
                self.agent_B: self._make_observation(self.agent_B),
            },
            metadata={
                "pie_size": self.pie_size,
                "proposer_id": self.proposer_id,
                "responder_id": self.responder_id,
                "current_phase": self.current_phase,
                "offer": self.offer,
                "response": self.response,
                "outcome": outcome,
            },
        )

        # Record to history
        self.history.append({
            "round": 2,
            "offer": self.offer,
            "response": self.response,
            "payoffs": self.payoffs.copy(),
            "outcome": outcome,
        })

        return self.state, self.payoffs.copy(), True

    def get_observation(self, agent_id: str) -> Dict[str, Any]:
        """Get current observation for an agent."""
        if self.state is None:
            raise RuntimeError("Game not initialized. Call reset() first.")
        return self.state.observations[agent_id]

    def _make_observation(self, agent_id: str) -> Dict[str, Any]:
        """Create an observation dictionary for an agent."""
        obs = {
            "agent_id": agent_id,
            "game_type": "ultimatum",
            "pie_size": self.pie_size,
            "allow_fractional": self.allow_fractional,
            "role": "proposer" if agent_id == self.proposer_id else "responder",
            "round": self.state.round if self.state else 0,
            "current_phase": self.current_phase,
            "is_terminal": self.is_terminal(),
        }

        # Add phase-specific information
        if self.current_phase in ["response", "complete"] and self.offer is not None:
            obs["offer"] = self.offer
            obs["responder_receives"] = self.pie_size - self.offer

        if self.current_phase == "complete":
            obs["response"] = self.response
            obs["outcome"] = "accept" if self.response else "reject"
            obs["my_payoff"] = self.payoffs[agent_id]
            obs["opponent_payoff"] = self.payoffs[self._get_opponent(agent_id)]

        return obs

    def get_rewards(self) -> Dict[str, float]:
        """Get final payoffs for both agents."""
        return self.payoffs.copy()

    def validate_action(self, agent_id: str, action: Any) -> Tuple[bool, Optional[str]]:
        """Validate an agent's action based on current phase."""
        if agent_id not in self.agent_ids:
            return False, f"Unknown agent: {agent_id}"

        if self.current_phase == "proposal":
            if agent_id != self.proposer_id:
                return False, f"Not {agent_id}'s turn (waiting for proposer)"

            # Validate offer
            if not isinstance(action, (int, float)):
                return False, f"Offer must be a number, got {type(action)}"

            offer = float(action)

            if not self.allow_fractional and offer != int(offer):
                return False, f"Fractional offers not allowed, got {offer}"

            if offer < 0:
                return False, f"Offer must be non-negative, got {offer}"

            if offer > self.pie_size:
                return False, f"Offer {offer} exceeds pie size {self.pie_size}"

            return True, None

        elif self.current_phase == "response":
            if agent_id != self.responder_id:
                return False, f"Not {agent_id}'s turn (waiting for responder)"

            # Validate response (must be boolean or boolean-like)
            if not isinstance(action, (bool, int)):
                return False, f"Response must be boolean, got {type(action)}"

            return True, None

        else:
            return False, "Game already complete"

    def _get_opponent(self, agent_id: str) -> str:
        """Get the opponent's agent_id."""
        return self.agent_B if agent_id == self.agent_A else self.agent_A

    def render(self) -> str:
        """Render the game state as a human-readable string."""
        if self.state is None:
            return f"Ultimatum Game (not started)\nPie size: {self.pie_size}\n"

        base = f"=== Ultimatum Game ===\n"
        base += f"Pie size: {self.pie_size}\n"
        base += f"Proposer: {self.proposer_id}\n"
        base += f"Responder: {self.responder_id}\n\n"

        if self.current_phase == "proposal":
            base += "Status: Waiting for proposer's offer\n"
        elif self.current_phase == "response":
            base += f"Offer: {self.offer} (Proposer keeps {self.offer}, Responder gets {self.pie_size - self.offer})\n"
            base += "Status: Waiting for responder's decision\n"
        else:  # complete
            base += f"Offer: {self.offer} (Proposer keeps {self.offer}, Responder gets {self.pie_size - self.offer})\n"
            base += f"Response: {'ACCEPT' if self.response else 'REJECT'}\n"
            base += f"\nOutcome: {self.state.metadata.get('outcome')}\n"
            base += f"{self.proposer_id} payoff: {self.payoffs[self.proposer_id]}\n"
            base += f"{self.responder_id} payoff: {self.payoffs[self.responder_id]}\n"

        return base

    def get_equilibrium_prediction(self) -> Dict[str, Any]:
        """
        Return game-theoretic equilibrium prediction.

        Subgame perfect equilibrium:
        - Proposer offers minimum (ε → 0)
        - Responder accepts any positive offer

        Returns:
            Dictionary with equilibrium information
        """
        return {
            "equilibrium_type": "subgame_perfect",
            "description": "Proposer offers minimum, Responder accepts any positive offer",
            "prediction": {
                "proposer_offer": "pie_size - ε (where ε → 0)",
                "responder_response": "accept",
                "proposer_payoff": self.pie_size,
                "responder_payoff": 0.0,
            },
            "behavioral_focal_point": {
                "proposer_offer": self.pie_size / 2,
                "justification": "Fairness norm (50-50 split)",
            },
        }
