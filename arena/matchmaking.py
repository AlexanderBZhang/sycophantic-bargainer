"""
Matchmaking and Rating System for Sycophantic Bargainer Arena.

This module implements Elo and TrueSkill rating systems for tracking
agent performance across multiple matches in tournament settings.

Features:
- Elo rating system (traditional chess-style)
- TrueSkill rating system (Bayesian skill estimation)
- Match history tracking
- Leaderboard generation
- Rating persistence (JSON/CSV export)
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
import json
import math
from pathlib import Path


@dataclass
class PlayerRating:
    """Represents a player's rating and statistics."""
    player_id: str
    elo: float = 1500.0
    trueskill_mu: float = 25.0
    trueskill_sigma: float = 8.333
    matches_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_reward: float = 0.0
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.matches_played == 0:
            return 0.0
        return self.wins / self.matches_played
    
    @property
    def trueskill_conservative(self) -> float:
        """Conservative TrueSkill estimate (mu - 3*sigma)."""
        return self.trueskill_mu - 3 * self.trueskill_sigma
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['win_rate'] = self.win_rate
        d['trueskill_conservative'] = self.trueskill_conservative
        return d


@dataclass
class MatchRecord:
    """Record of a single match for rating purposes."""
    match_id: str
    timestamp: str
    game_type: str
    player_1: str
    player_2: str
    player_1_reward: float
    player_2_reward: float
    outcome: str  # "player_1_wins", "player_2_wins", "draw"
    player_1_elo_before: float
    player_2_elo_before: float
    player_1_elo_after: float
    player_2_elo_after: float
    elo_change: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class EloRatingSystem:
    """
    Elo rating system implementation.
    
    The Elo system is a method for calculating the relative skill levels
    of players in zero-sum games. Originally developed for chess.
    """
    
    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1500.0):
        """
        Initialize Elo rating system.
        
        Args:
            k_factor: Maximum rating change per game (higher = more volatile)
            initial_rating: Starting rating for new players
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for player A against player B.
        
        Args:
            rating_a: Player A's rating
            rating_b: Player B's rating
        
        Returns:
            Expected score (0.0 to 1.0)
        """
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
    
    def update_ratings(
        self,
        rating_a: float,
        rating_b: float,
        score_a: float
    ) -> Tuple[float, float]:
        """
        Update ratings after a match.
        
        Args:
            rating_a: Player A's current rating
            rating_b: Player B's current rating
            score_a: Player A's actual score (1.0 = win, 0.5 = draw, 0.0 = loss)
        
        Returns:
            Tuple of (new_rating_a, new_rating_b)
        """
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a
        
        score_b = 1.0 - score_a
        
        new_rating_a = rating_a + self.k_factor * (score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * (score_b - expected_b)
        
        return new_rating_a, new_rating_b
    
    def calculate_rating_change(
        self,
        rating_a: float,
        rating_b: float,
        score_a: float
    ) -> float:
        """
        Calculate the rating change for player A.
        
        Args:
            rating_a: Player A's current rating
            rating_b: Player B's current rating
            score_a: Player A's actual score
        
        Returns:
            Rating change (positive or negative)
        """
        expected_a = self.expected_score(rating_a, rating_b)
        return self.k_factor * (score_a - expected_a)


class TrueSkillRatingSystem:
    """
    Simplified TrueSkill rating system implementation.
    
    TrueSkill is a Bayesian skill rating system developed by Microsoft Research.
    This is a simplified version that tracks mean (mu) and uncertainty (sigma).
    """
    
    def __init__(
        self,
        initial_mu: float = 25.0,
        initial_sigma: float = 8.333,
        beta: float = 4.167,
        tau: float = 0.0833,
        draw_probability: float = 0.1
    ):
        """
        Initialize TrueSkill rating system.
        
        Args:
            initial_mu: Initial skill mean
            initial_sigma: Initial skill uncertainty
            beta: Skill class width (performance variability)
            tau: Dynamics factor (skill change over time)
            draw_probability: Probability of a draw
        """
        self.initial_mu = initial_mu
        self.initial_sigma = initial_sigma
        self.beta = beta
        self.tau = tau
        self.draw_probability = draw_probability
    
    def win_probability(
        self,
        mu_a: float,
        sigma_a: float,
        mu_b: float,
        sigma_b: float
    ) -> float:
        """
        Calculate probability that player A beats player B.
        
        Args:
            mu_a: Player A's skill mean
            sigma_a: Player A's skill uncertainty
            mu_b: Player B's skill mean
            sigma_b: Player B's skill uncertainty
        
        Returns:
            Win probability (0.0 to 1.0)
        """
        delta_mu = mu_a - mu_b
        sum_sigma = math.sqrt(sigma_a**2 + sigma_b**2 + 2 * self.beta**2)
        
        # Use cumulative normal distribution approximation
        return self._phi(delta_mu / sum_sigma)
    
    def update_ratings(
        self,
        mu_a: float,
        sigma_a: float,
        mu_b: float,
        sigma_b: float,
        outcome: str
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Update TrueSkill ratings after a match.
        
        Args:
            mu_a: Player A's skill mean
            sigma_a: Player A's skill uncertainty
            mu_b: Player B's skill mean
            sigma_b: Player B's skill uncertainty
            outcome: "a_wins", "b_wins", or "draw"
        
        Returns:
            Tuple of ((new_mu_a, new_sigma_a), (new_mu_b, new_sigma_b))
        """
        # Simplified update (not full TrueSkill algorithm)
        # This is a basic approximation for demonstration
        
        # Add dynamics uncertainty
        sigma_a = math.sqrt(sigma_a**2 + self.tau**2)
        sigma_b = math.sqrt(sigma_b**2 + self.tau**2)
        
        # Calculate performance difference
        c = math.sqrt(sigma_a**2 + sigma_b**2 + 2 * self.beta**2)
        
        # Determine outcome value
        if outcome == "a_wins":
            v = 1.0
        elif outcome == "b_wins":
            v = -1.0
        else:  # draw
            v = 0.0
        
        # Update means (simplified)
        t = (mu_a - mu_b) / c
        update_factor = (sigma_a**2 + sigma_b**2) / c
        
        mu_a_new = mu_a + (sigma_a**2 / c) * v * self._phi_derivative(t)
        mu_b_new = mu_b - (sigma_b**2 / c) * v * self._phi_derivative(t)
        
        # Update sigmas (simplified - just reduce uncertainty slightly)
        sigma_a_new = sigma_a * 0.95
        sigma_b_new = sigma_b * 0.95
        
        return (mu_a_new, sigma_a_new), (mu_b_new, sigma_b_new)
    
    @staticmethod
    def _phi(x: float) -> float:
        """Cumulative distribution function for standard normal distribution."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    
    @staticmethod
    def _phi_derivative(x: float) -> float:
        """Derivative of cumulative distribution function."""
        return math.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


class MatchmakingSystem:
    """
    Complete matchmaking system with rating tracking and persistence.
    
    This system manages player ratings, match history, and leaderboards
    using both Elo and TrueSkill rating systems.
    """
    
    def __init__(
        self,
        elo_k_factor: float = 32.0,
        initial_elo: float = 1500.0,
        storage_path: Optional[Path] = None
    ):
        """
        Initialize matchmaking system.
        
        Args:
            elo_k_factor: Elo K-factor
            initial_elo: Initial Elo rating
            storage_path: Path to store ratings and match history
        """
        self.elo_system = EloRatingSystem(k_factor=elo_k_factor, initial_rating=initial_elo)
        self.trueskill_system = TrueSkillRatingSystem()
        
        self.players: Dict[str, PlayerRating] = {}
        self.match_history: List[MatchRecord] = []
        
        self.storage_path = storage_path or Path("./arena_data")
        self.storage_path.mkdir(exist_ok=True)
    
    def get_or_create_player(self, player_id: str) -> PlayerRating:
        """
        Get existing player or create new one with initial ratings.
        
        Args:
            player_id: Player identifier
        
        Returns:
            PlayerRating object
        """
        if player_id not in self.players:
            self.players[player_id] = PlayerRating(
                player_id=player_id,
                elo=self.elo_system.initial_rating,
                trueskill_mu=self.trueskill_system.initial_mu,
                trueskill_sigma=self.trueskill_system.initial_sigma
            )
        return self.players[player_id]
    
    def record_match(
        self,
        match_id: str,
        game_type: str,
        player_1_id: str,
        player_2_id: str,
        player_1_reward: float,
        player_2_reward: float
    ) -> MatchRecord:
        """
        Record a match and update ratings.
        
        Args:
            match_id: Unique match identifier
            game_type: Type of game played
            player_1_id: First player ID
            player_2_id: Second player ID
            player_1_reward: Reward received by player 1
            player_2_reward: Reward received by player 2
        
        Returns:
            MatchRecord with rating changes
        """
        # Get or create players
        player_1 = self.get_or_create_player(player_1_id)
        player_2 = self.get_or_create_player(player_2_id)
        
        # Determine outcome
        if player_1_reward > player_2_reward:
            outcome = "player_1_wins"
            elo_score_1 = 1.0
            trueskill_outcome = "a_wins"
        elif player_2_reward > player_1_reward:
            outcome = "player_2_wins"
            elo_score_1 = 0.0
            trueskill_outcome = "b_wins"
        else:
            outcome = "draw"
            elo_score_1 = 0.5
            trueskill_outcome = "draw"
        
        # Store old ratings
        elo_1_before = player_1.elo
        elo_2_before = player_2.elo
        
        # Update Elo ratings
        new_elo_1, new_elo_2 = self.elo_system.update_ratings(
            player_1.elo, player_2.elo, elo_score_1
        )
        
        # Update TrueSkill ratings
        (new_mu_1, new_sigma_1), (new_mu_2, new_sigma_2) = self.trueskill_system.update_ratings(
            player_1.trueskill_mu,
            player_1.trueskill_sigma,
            player_2.trueskill_mu,
            player_2.trueskill_sigma,
            trueskill_outcome
        )
        
        # Update player records
        player_1.elo = new_elo_1
        player_1.trueskill_mu = new_mu_1
        player_1.trueskill_sigma = new_sigma_1
        player_1.matches_played += 1
        player_1.total_reward += player_1_reward
        
        player_2.elo = new_elo_2
        player_2.trueskill_mu = new_mu_2
        player_2.trueskill_sigma = new_sigma_2
        player_2.matches_played += 1
        player_2.total_reward += player_2_reward
        
        # Update win/loss/draw counts
        if outcome == "player_1_wins":
            player_1.wins += 1
            player_2.losses += 1
        elif outcome == "player_2_wins":
            player_1.losses += 1
            player_2.wins += 1
        else:
            player_1.draws += 1
            player_2.draws += 1
        
        # Update timestamps
        timestamp = datetime.now().isoformat()
        player_1.last_updated = timestamp
        player_2.last_updated = timestamp
        
        # Create match record
        match_record = MatchRecord(
            match_id=match_id,
            timestamp=timestamp,
            game_type=game_type,
            player_1=player_1_id,
            player_2=player_2_id,
            player_1_reward=player_1_reward,
            player_2_reward=player_2_reward,
            outcome=outcome,
            player_1_elo_before=elo_1_before,
            player_2_elo_before=elo_2_before,
            player_1_elo_after=new_elo_1,
            player_2_elo_after=new_elo_2,
            elo_change=new_elo_1 - elo_1_before
        )
        
        self.match_history.append(match_record)
        
        return match_record
    
    def get_leaderboard(
        self,
        sort_by: str = "elo",
        min_matches: int = 0
    ) -> List[PlayerRating]:
        """
        Get leaderboard sorted by rating.
        
        Args:
            sort_by: Sort criterion ("elo", "trueskill", "win_rate", "matches")
            min_matches: Minimum matches required to appear on leaderboard
        
        Returns:
            Sorted list of PlayerRating objects
        """
        # Filter by minimum matches
        eligible_players = [
            p for p in self.players.values()
            if p.matches_played >= min_matches
        ]
        
        # Sort by criterion
        if sort_by == "elo":
            return sorted(eligible_players, key=lambda p: p.elo, reverse=True)
        elif sort_by == "trueskill":
            return sorted(eligible_players, key=lambda p: p.trueskill_conservative, reverse=True)
        elif sort_by == "win_rate":
            return sorted(eligible_players, key=lambda p: p.win_rate, reverse=True)
        elif sort_by == "matches":
            return sorted(eligible_players, key=lambda p: p.matches_played, reverse=True)
        else:
            raise ValueError(f"Unknown sort criterion: {sort_by}")
    
    def get_player_stats(self, player_id: str) -> Optional[PlayerRating]:
        """Get statistics for a specific player."""
        return self.players.get(player_id)
    
    def save_to_json(self, filename: str = "ratings.json") -> None:
        """Save ratings and match history to JSON file."""
        filepath = self.storage_path / filename
        
        data = {
            "players": {pid: p.to_dict() for pid, p in self.players.items()},
            "match_history": [m.to_dict() for m in self.match_history],
            "metadata": {
                "total_matches": len(self.match_history),
                "total_players": len(self.players),
                "last_updated": datetime.now().isoformat()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_json(self, filename: str = "ratings.json") -> None:
        """Load ratings and match history from JSON file."""
        filepath = self.storage_path / filename
        
        if not filepath.exists():
            return
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Load players
        self.players = {
            pid: PlayerRating(**pdata)
            for pid, pdata in data["players"].items()
        }
        
        # Load match history
        self.match_history = [
            MatchRecord(**mdata)
            for mdata in data["match_history"]
        ]
    
    def export_leaderboard_csv(self, filename: str = "leaderboard.csv") -> None:
        """Export leaderboard to CSV file."""
        filepath = self.storage_path / filename
        
        leaderboard = self.get_leaderboard(sort_by="elo")
        
        with open(filepath, 'w') as f:
            # Header
            f.write("rank,player_id,elo,trueskill_conservative,matches,wins,losses,draws,win_rate\n")
            
            # Data
            for rank, player in enumerate(leaderboard, 1):
                f.write(
                    f"{rank},{player.player_id},{player.elo:.1f},"
                    f"{player.trueskill_conservative:.1f},{player.matches_played},"
                    f"{player.wins},{player.losses},{player.draws},{player.win_rate:.3f}\n"
                )
    
    def get_match_history(
        self,
        player_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[MatchRecord]:
        """
        Get match history, optionally filtered by player.
        
        Args:
            player_id: Optional player ID to filter by
            limit: Optional limit on number of matches to return
        
        Returns:
            List of MatchRecord objects
        """
        history = self.match_history
        
        if player_id:
            history = [
                m for m in history
                if m.player_1 == player_id or m.player_2 == player_id
            ]
        
        if limit:
            history = history[-limit:]
        
        return history
