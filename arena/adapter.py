"""
Arena Adapter for Sycophantic Bargainer Environments.

This module provides a standardized interface for running bargaining games
in an arena/tournament setting, compatible with Google DeepMind's game arena pattern.

The adapter wraps existing BargainEnvironment implementations and provides:
- Standardized match interface
- Move history tracking
- Elo/TrueSkill integration hooks
- Serializable game records
"""

from typing import Dict, Any, List, Optional, Tuple, Type
from dataclasses import dataclass, asdict
from datetime import datetime
import json

from environments.base import BargainEnvironment
from environments.nash_demand import NashDemandGame
from environments.ultimatum import UltimatumGame
from environments.rubinstein import RubinsteinBargaining


@dataclass
class ArenaMove:
    """Represents a single move in an arena match."""
    player_id: str
    action: Any
    timestamp: str
    round_num: int
    metadata: Dict[str, Any]


@dataclass
class ArenaMatchResult:
    """Complete record of an arena match."""
    match_id: str
    game_type: str
    player_ids: Tuple[str, str]
    moves: List[ArenaMove]
    final_rewards: Dict[str, float]
    outcome: str
    duration_seconds: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "match_id": self.match_id,
            "game_type": self.game_type,
            "player_ids": list(self.player_ids),
            "moves": [asdict(move) for move in self.moves],
            "final_rewards": self.final_rewards,
            "outcome": self.outcome,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class ArenaEnvironment:
    """
    Arena wrapper for BargainEnvironment implementations.
    
    This adapter provides a standardized interface for running matches
    in a tournament/arena setting while preserving the full functionality
    of the underlying bargaining game.
    """
    
    # Registry of available game types
    GAME_REGISTRY: Dict[str, Type[BargainEnvironment]] = {
        "nash_demand": NashDemandGame,
        "ultimatum": UltimatumGame,
        "rubinstein": RubinsteinBargaining,
    }
    
    def __init__(
        self,
        game_type: str,
        agent_ids: Tuple[str, str] = ("player_1", "player_2"),
        **game_kwargs
    ):
        """
        Initialize an arena environment.
        
        Args:
            game_type: Type of game ("nash_demand", "ultimatum", "rubinstein")
            agent_ids: Tuple of (player_1_id, player_2_id)
            **game_kwargs: Additional arguments passed to the game constructor
        """
        if game_type not in self.GAME_REGISTRY:
            raise ValueError(
                f"Unknown game type: {game_type}. "
                f"Available: {list(self.GAME_REGISTRY.keys())}"
            )
        
        self.game_type = game_type
        self.agent_ids = agent_ids
        self.game_kwargs = game_kwargs
        
        # Create the underlying environment
        env_class = self.GAME_REGISTRY[game_type]
        self.env: BargainEnvironment = env_class(agent_ids=agent_ids, **game_kwargs)
        
        # Arena-specific tracking
        self.move_history: List[ArenaMove] = []
        self.match_start_time: Optional[datetime] = None
        self.match_id: Optional[str] = None
    
    def start_match(self, match_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Start a new match.
        
        Args:
            match_id: Optional unique identifier for this match
        
        Returns:
            Initial game state
        """
        self.match_id = match_id or self._generate_match_id()
        self.match_start_time = datetime.now()
        self.move_history = []
        
        # Reset the underlying environment
        state = self.env.reset()
        
        return {
            "match_id": self.match_id,
            "game_type": self.game_type,
            "player_ids": self.agent_ids,
            "initial_state": state,
        }
    
    def execute_move(
        self,
        player_id: str,
        action: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate and record a move (but don't execute yet).
        
        For simultaneous games like Nash Demand, moves are collected
        then executed together via execute_round().
        
        Args:
            player_id: ID of the player making the move
            action: The action to take
            metadata: Optional metadata about the move
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate the action
        is_valid, error_msg = self.env.validate_action(player_id, action)
        
        if is_valid:
            # Record the move
            move = ArenaMove(
                player_id=player_id,
                action=action,
                timestamp=datetime.now().isoformat(),
                round_num=self.env.state.round if self.env.state else 0,
                metadata=metadata or {},
            )
            self.move_history.append(move)
        
        return is_valid, error_msg
    
    def execute_round(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """
        Execute a complete round of the game.
        
        Args:
            actions: Dictionary mapping player_id to action
        
        Returns:
            Tuple of (round_result, is_terminal)
            - round_result: Dictionary with state, rewards, observations
            - is_terminal: Whether the game has ended
        """
        # Execute the step in the underlying environment
        new_state, rewards, done = self.env.step(actions)
        
        # Get observations for both players
        observations = {
            player_id: self.env.get_observation(player_id)
            for player_id in self.agent_ids
        }
        
        result = {
            "state": new_state,
            "rewards": rewards,
            "observations": observations,
            "is_terminal": done,
            "round": new_state.round,
        }
        
        return result, done
    
    def get_match_result(self) -> ArenaMatchResult:
        """
        Get the complete match result.
        
        Returns:
            ArenaMatchResult with full match details
        """
        if not self.env.is_terminal():
            raise RuntimeError("Cannot get match result - game not finished")
        
        if self.match_start_time is None:
            raise RuntimeError("Match not started")
        
        duration = (datetime.now() - self.match_start_time).total_seconds()
        
        # Determine outcome
        rewards = self.env.get_rewards()
        if rewards[self.agent_ids[0]] > rewards[self.agent_ids[1]]:
            outcome = f"{self.agent_ids[0]}_wins"
        elif rewards[self.agent_ids[1]] > rewards[self.agent_ids[0]]:
            outcome = f"{self.agent_ids[1]}_wins"
        else:
            outcome = "draw"
        
        return ArenaMatchResult(
            match_id=self.match_id,
            game_type=self.game_type,
            player_ids=self.agent_ids,
            moves=self.move_history,
            final_rewards=rewards,
            outcome=outcome,
            duration_seconds=duration,
            metadata={
                "game_info": self.env.get_game_info(),
                "game_kwargs": self.game_kwargs,
            },
        )
    
    def get_observation(self, player_id: str) -> Dict[str, Any]:
        """
        Get current observation for a player.
        
        Args:
            player_id: ID of the player
        
        Returns:
            Observation dictionary
        """
        return self.env.get_observation(player_id)
    
    def is_terminal(self) -> bool:
        """Check if the game has ended."""
        return self.env.is_terminal()
    
    def render(self) -> str:
        """Render the current game state."""
        return self.env.render()
    
    @staticmethod
    def _generate_match_id() -> str:
        """Generate a unique match ID."""
        return f"match_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    @classmethod
    def list_available_games(cls) -> List[str]:
        """Get list of available game types."""
        return list(cls.GAME_REGISTRY.keys())


class ArenaMatchRunner:
    """
    High-level interface for running complete matches.
    
    This class handles the full lifecycle of a match, including:
    - Agent coordination
    - Move collection
    - Result recording
    """
    
    def __init__(self, arena_env: ArenaEnvironment):
        """
        Initialize match runner.
        
        Args:
            arena_env: ArenaEnvironment instance
        """
        self.arena = arena_env
    
    def run_match(
        self,
        agent_actions: Dict[str, Any],
        match_id: Optional[str] = None
    ) -> ArenaMatchResult:
        """
        Run a complete match with pre-determined actions.
        
        This is useful for testing or running matches with mock agents.
        
        Args:
            agent_actions: Dictionary mapping agent_id to action
            match_id: Optional match identifier
        
        Returns:
            ArenaMatchResult with complete match details
        """
        # Start the match
        self.arena.start_match(match_id)
        
        # Validate and record moves
        for player_id, action in agent_actions.items():
            is_valid, error_msg = self.arena.execute_move(player_id, action)
            if not is_valid:
                raise ValueError(f"Invalid move by {player_id}: {error_msg}")
        
        # Execute the round
        result, is_terminal = self.arena.execute_round(agent_actions)
        
        if not is_terminal:
            raise RuntimeError("Game did not terminate after one round")
        
        # Return the match result
        return self.arena.get_match_result()


# Convenience function for quick match setup
def create_arena_match(
    game_type: str,
    player_ids: Tuple[str, str] = ("player_1", "player_2"),
    **game_kwargs
) -> ArenaEnvironment:
    """
    Create an arena environment for a match.
    
    Args:
        game_type: Type of game to play
        player_ids: Tuple of player identifiers
        **game_kwargs: Additional game configuration
    
    Returns:
        Configured ArenaEnvironment
    """
    return ArenaEnvironment(game_type, player_ids, **game_kwargs)
