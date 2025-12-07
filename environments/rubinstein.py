"""
Rubinstein Alternating Offers Bargaining implementation.

In this sequential bargaining game, agents alternate making offers to split a pie.
Each round applies a discount factor δ to remaining payoffs. Agents can accept
or reject each offer. Game continues until acceptance or max rounds.

Game-theoretic properties:
- Unique subgame perfect equilibrium with immediate agreement
- Player 1 offers δ/(1+δ), keeps 1/(1+δ)
- Player 2 accepts immediately
- Tests patience, time preference, and strategic reasoning
"""

from typing import Dict, Any, Tuple, Optional
from .base import BargainEnvironment, GameState


class RubinsteinBargaining(BargainEnvironment):
    """
    Rubinstein Alternating Offers Bargaining environment.

    Sequential game:
    1. Player 1 makes an offer (x, 1-x)
    2. Player 2 accepts or rejects
    3. If accept: game ends, payoffs = (x * δ^t, (1-x) * δ^t)
       If reject: Player 2 makes counteroffer
    4. Continue alternating until acceptance or max_rounds
    """

    def __init__(
        self,
        pie_size: float = 10.0,
        agent_ids: Tuple[str, str] = ("agent_A", "agent_B"),
        discount_factor: float = 0.9,
        max_rounds: int = 10,
        allow_fractional: bool = True,
    ):
        """
        Initialize Rubinstein Bargaining game.

        Args:
            pie_size: Total size of the pie to divide
            agent_ids: Tuple of (agent_A_id, agent_B_id)
            discount_factor: Time discount factor δ ∈ (0, 1). Closer to 1 = more patient
            max_rounds: Maximum number of negotiation rounds
            allow_fractional: Whether to allow fractional offers
        """
        super().__init__(agent_ids)
        self.pie_size = pie_size
        self.discount_factor = discount_factor
        self.max_rounds = max_rounds
        self.allow_fractional = allow_fractional

        if not 0 < discount_factor < 1:
            raise ValueError(f"discount_factor must be in (0, 1), got {discount_factor}")

        # Game state
        self.current_round: int = 0
        self.current_proposer_id: Optional[str] = None
        self.current_responder_id: Optional[str] = None
        self.current_offer: Optional[float] = None  # Proposer's share
        self.waiting_for: str = "offer"  # "offer" or "response"
        self.payoffs: Dict[str, float] = {self.agent_A: 0.0, self.agent_B: 0.0}
        self.offer_history: list = []

    def reset(self) -> GameState:
        """Reset the game to initial state."""
        self.current_round = 1
        self.current_proposer_id = self.agent_A  # Agent A goes first
        self.current_responder_id = self.agent_B
        self.current_offer = None
        self.waiting_for = "offer"
        self.payoffs = {self.agent_A: 0.0, self.agent_B: 0.0}
        self.offer_history = []
        self.history = []

        self.state = GameState(
            round=self.current_round,
            is_terminal=False,
            observations={
                self.agent_A: self._make_observation(self.agent_A),
                self.agent_B: self._make_observation(self.agent_B),
            },
            metadata={
                "pie_size": self.pie_size,
                "discount_factor": self.discount_factor,
                "max_rounds": self.max_rounds,
                "current_proposer": self.current_proposer_id,
                "current_responder": self.current_responder_id,
                "waiting_for": self.waiting_for,
            },
        )

        return self.state

    def step(self, actions: Dict[str, Any]) -> Tuple[GameState, Dict[str, float], bool]:
        """
        Execute one step of the game.

        When waiting_for == "offer": expects {proposer_id: offer_amount}
        When waiting_for == "response": expects {responder_id: True/False}

        Args:
            actions: Dict with single entry for current acting agent

        Returns:
            Tuple of (new_state, rewards, done)
        """
        if self.waiting_for == "offer":
            return self._step_offer(actions)
        else:  # waiting_for == "response"
            return self._step_response(actions)

    def _step_offer(self, actions: Dict[str, Any]) -> Tuple[GameState, Dict[str, float], bool]:
        """Handle offer phase."""
        if self.current_proposer_id not in actions:
            raise ValueError(f"Expected action from proposer {self.current_proposer_id}")

        offer = actions[self.current_proposer_id]

        # Validate
        is_valid, error_msg = self.validate_action(self.current_proposer_id, offer)
        if not is_valid:
            raise ValueError(f"Invalid offer from {self.current_proposer_id}: {error_msg}")

        self.current_offer = float(offer)
        self.waiting_for = "response"

        # Record offer
        self.offer_history.append({
            "round": self.current_round,
            "proposer": self.current_proposer_id,
            "offer": self.current_offer,
        })

        # Update state
        self.state = GameState(
            round=self.current_round,
            is_terminal=False,
            observations={
                self.agent_A: self._make_observation(self.agent_A),
                self.agent_B: self._make_observation(self.agent_B),
            },
            metadata={
                "pie_size": self.pie_size,
                "discount_factor": self.discount_factor,
                "max_rounds": self.max_rounds,
                "current_proposer": self.current_proposer_id,
                "current_responder": self.current_responder_id,
                "current_offer": self.current_offer,
                "waiting_for": self.waiting_for,
            },
        )

        return self.state, {self.agent_A: 0.0, self.agent_B: 0.0}, False

    def _step_response(self, actions: Dict[str, Any]) -> Tuple[GameState, Dict[str, float], bool]:
        """Handle response phase."""
        if self.current_responder_id not in actions:
            raise ValueError(f"Expected action from responder {self.current_responder_id}")

        response = actions[self.current_responder_id]

        # Validate
        is_valid, error_msg = self.validate_action(self.current_responder_id, response)
        if not is_valid:
            raise ValueError(f"Invalid response from {self.current_responder_id}: {error_msg}")

        response_bool = bool(response)

        if response_bool:
            # Accept: game ends
            return self._finalize_agreement()
        else:
            # Reject: continue to next round
            return self._continue_bargaining()

    def _finalize_agreement(self) -> Tuple[GameState, Dict[str, float], bool]:
        """Finalize game with agreement."""
        # Calculate discounted payoffs
        discount = self.discount_factor ** (self.current_round - 1)

        proposer_share = self.current_offer * discount
        responder_share = (self.pie_size - self.current_offer) * discount

        self.payoffs[self.current_proposer_id] = proposer_share
        self.payoffs[self.current_responder_id] = responder_share

        # Update state
        self.state = GameState(
            round=self.current_round,
            is_terminal=True,
            observations={
                self.agent_A: self._make_observation(self.agent_A),
                self.agent_B: self._make_observation(self.agent_B),
            },
            metadata={
                "pie_size": self.pie_size,
                "discount_factor": self.discount_factor,
                "final_round": self.current_round,
                "proposer": self.current_proposer_id,
                "final_offer": self.current_offer,
                "outcome": "agreement",
                "discount_applied": self.discount_factor ** (self.current_round - 1),
            },
        )

        # Record to history
        self.history.append({
            "round": self.current_round,
            "action": "accept",
            "proposer": self.current_proposer_id,
            "offer": self.current_offer,
            "payoffs": self.payoffs.copy(),
            "outcome": "agreement",
        })

        return self.state, self.payoffs.copy(), True

    def _continue_bargaining(self) -> Tuple[GameState, Dict[str, float], bool]:
        """Continue to next round (rejected offer)."""
        # Record rejection
        self.history.append({
            "round": self.current_round,
            "action": "reject",
            "proposer": self.current_proposer_id,
            "offer": self.current_offer,
        })

        # Check if max rounds reached
        if self.current_round >= self.max_rounds:
            return self._finalize_disagreement()

        # Switch roles
        self.current_proposer_id, self.current_responder_id = (
            self.current_responder_id,
            self.current_proposer_id,
        )
        self.current_round += 1
        self.current_offer = None
        self.waiting_for = "offer"

        # Update state
        self.state = GameState(
            round=self.current_round,
            is_terminal=False,
            observations={
                self.agent_A: self._make_observation(self.agent_A),
                self.agent_B: self._make_observation(self.agent_B),
            },
            metadata={
                "pie_size": self.pie_size,
                "discount_factor": self.discount_factor,
                "max_rounds": self.max_rounds,
                "current_proposer": self.current_proposer_id,
                "current_responder": self.current_responder_id,
                "waiting_for": self.waiting_for,
            },
        )

        return self.state, {self.agent_A: 0.0, self.agent_B: 0.0}, False

    def _finalize_disagreement(self) -> Tuple[GameState, Dict[str, float], bool]:
        """Finalize game with disagreement (max rounds reached)."""
        # Disagreement Payoff: 20.0 (Reservation Value)
        self.payoffs = {self.agent_A: 20.0, self.agent_B: 20.0}

        self.state = GameState(
            round=self.current_round,
            is_terminal=True,
            observations={
                self.agent_A: self._make_observation(self.agent_A),
                self.agent_B: self._make_observation(self.agent_B),
            },
            metadata={
                "pie_size": self.pie_size,
                "discount_factor": self.discount_factor,
                "final_round": self.current_round,
                "outcome": "disagreement",
                "reason": "max_rounds_reached",
            },
        )

        self.history.append({
            "round": self.current_round,
            "outcome": "disagreement",
            "reason": "max_rounds_reached",
            "payoffs": self.payoffs.copy(),
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
            "game_type": "rubinstein",
            "pie_size": self.pie_size,
            "discount_factor": self.discount_factor,
            "max_rounds": self.max_rounds,
            "current_round": self.current_round,
            "is_terminal": self.is_terminal(),
            "waiting_for": self.waiting_for,
        }

        # Add role information
        if self.current_proposer_id:
            obs["is_proposer"] = (agent_id == self.current_proposer_id)
            obs["current_proposer"] = self.current_proposer_id
            obs["current_responder"] = self.current_responder_id

        # Add current offer if available
        if self.waiting_for == "response" and self.current_offer is not None:
            if agent_id == self.current_proposer_id:
                obs["my_offer"] = self.current_offer
                obs["opponent_receives"] = self.pie_size - self.current_offer
            else:
                obs["opponent_offer"] = self.current_offer
                obs["i_receive"] = self.pie_size - self.current_offer

        # Add final results
        if self.is_terminal():
            obs["my_payoff"] = self.payoffs[agent_id]
            obs["opponent_payoff"] = self.payoffs[self._get_opponent(agent_id)]
            obs["outcome"] = self.state.metadata.get("outcome")
            if "final_round" in self.state.metadata:
                obs["final_round"] = self.state.metadata["final_round"]

        # Add offer history
        obs["offer_history"] = self.offer_history.copy()

        return obs

    def get_rewards(self) -> Dict[str, float]:
        """Get final payoffs for both agents."""
        return self.payoffs.copy()

    def validate_action(self, agent_id: str, action: Any) -> Tuple[bool, Optional[str]]:
        """Validate an agent's action based on current phase."""
        if agent_id not in self.agent_ids:
            return False, f"Unknown agent: {agent_id}"

        if self.waiting_for == "offer":
            if agent_id != self.current_proposer_id:
                return False, f"Not {agent_id}'s turn (waiting for proposer {self.current_proposer_id})"

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

        else:  # waiting_for == "response"
            if agent_id != self.current_responder_id:
                return False, f"Not {agent_id}'s turn (waiting for responder {self.current_responder_id})"

            # Validate response
            if not isinstance(action, (bool, int)):
                return False, f"Response must be boolean, got {type(action)}"

            return True, None

    def _get_opponent(self, agent_id: str) -> str:
        """Get the opponent's agent_id."""
        return self.agent_B if agent_id == self.agent_A else self.agent_A

    def render(self) -> str:
        """Render the game state as a human-readable string."""
        if self.state is None:
            return f"Rubinstein Bargaining (not started)\nPie size: {self.pie_size}\n"

        base = f"=== Rubinstein Alternating Offers Bargaining ===\n"
        base += f"Pie size: {self.pie_size}\n"
        base += f"Discount factor: {self.discount_factor}\n"
        base += f"Round: {self.current_round}/{self.max_rounds}\n\n"

        if not self.is_terminal():
            if self.waiting_for == "offer":
                base += f"Status: Waiting for {self.current_proposer_id}'s offer\n"
            else:
                base += f"Current offer: {self.current_offer} for {self.current_proposer_id}\n"
                base += f"Status: Waiting for {self.current_responder_id}'s response\n"

            # Show offer history
            if self.offer_history:
                base += "\nOffer History:\n"
                for entry in self.offer_history:
                    base += f"  Round {entry['round']}: {entry['proposer']} offered {entry['offer']}\n"
        else:
            outcome = self.state.metadata.get("outcome")
            base += f"Outcome: {outcome}\n"

            if outcome == "agreement":
                base += f"Final offer: {self.state.metadata.get('final_offer')}\n"
                base += f"Agreed in round: {self.state.metadata.get('final_round')}\n"
                base += f"Discount applied: {self.state.metadata.get('discount_applied'):.4f}\n"

            base += f"\nFinal Payoffs:\n"
            base += f"{self.agent_A}: {self.payoffs[self.agent_A]:.2f}\n"
            base += f"{self.agent_B}: {self.payoffs[self.agent_B]:.2f}\n"

        return base

    def get_equilibrium_prediction(self) -> Dict[str, Any]:
        """
        Return game-theoretic equilibrium prediction.

        Subgame perfect equilibrium (with immediate agreement):
        - Player 1 offers: δ/(1+δ) to Player 2, keeps 1/(1+δ)
        - Player 2 accepts immediately

        Returns:
            Dictionary with equilibrium information
        """
        δ = self.discount_factor

        player1_share = self.pie_size / (1 + δ)
        player2_share = self.pie_size * δ / (1 + δ)

        return {
            "equilibrium_type": "subgame_perfect",
            "description": "Immediate agreement in round 1",
            "prediction": {
                "round": 1,
                "player1_offer": player2_share,
                "player1_keeps": player1_share,
                "player2_response": "accept",
                self.agent_A: player1_share,
                self.agent_B: player2_share,
            },
            "formula": {
                "player1_share": f"{self.pie_size}/(1+δ) = {player1_share:.2f}",
                "player2_share": f"{self.pie_size}*δ/(1+δ) = {player2_share:.2f}",
            },
        }
