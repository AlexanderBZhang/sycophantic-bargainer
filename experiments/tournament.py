"""
Tournament Runner: Battle of the Giants

Run a round-robin tournament between top LLMs from OpenAI, Anthropic, and Google.
Uses the arena infrastructure with Elo/TrueSkill rating tracking.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from itertools import combinations

from sycophantic_bargainer.arena.adapter import ArenaEnvironment, create_arena_match
from sycophantic_bargainer.arena.matchmaking import MatchmakingSystem
from sycophantic_bargainer.agents.llm_agent import LLMAgent


# Model configurations for "Battle of the Giants"
TOURNAMENT_MODELS = [
    # OpenAI
    {"name": "GPT-4o", "provider": "openai", "model": "gpt-4o", "temperature": 0.7},
    {"name": "GPT-4-Turbo", "provider": "openai", "model": "gpt-4-turbo", "temperature": 0.7},
    
    # Anthropic
    {"name": "Claude-3.5-Sonnet", "provider": "anthropic", "model": "claude-3-5-sonnet-20241022", "temperature": 0.7},
    
    # Google
    {"name": "Gemini-1.5-Pro", "provider": "google", "model": "gemini-1.5-pro", "temperature": 0.7},
]


class TournamentRunner:
    """
    Run a round-robin tournament between multiple LLM agents.
    
    Features:
    - Round-robin format (every agent plays every other agent)
    - Multiple game types (Nash Demand, Ultimatum, Rubinstein)
    - Elo and TrueSkill rating tracking
    - Detailed match records and statistics
    - Leaderboard generation
    """
    
    def __init__(
        self,
        models: List[Dict[str, Any]],
        game_types: List[str] = None,
        persona_type: str = "rational",
        output_dir: Path = None,
        matches_per_pairing: int = 1,
    ):
        """
        Initialize tournament runner.
        
        Args:
            models: List of model configurations
            game_types: List of game types to play (default: all)
            persona_type: Persona type for all agents
            output_dir: Directory to save results
            matches_per_pairing: Number of matches per pairing
        """
        self.models = models
        self.game_types = game_types or ["nash_demand", "ultimatum", "rubinstein"]
        self.persona_type = persona_type
        self.output_dir = output_dir or Path("./tournament_results")
        self.matches_per_pairing = matches_per_pairing
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize matchmaking system
        self.matchmaking = MatchmakingSystem(
            storage_path=self.output_dir / "ratings"
        )
        
        # Tournament state
        self.match_results = []
        self.tournament_id = f"tournament_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def create_agent(self, model_config: Dict[str, Any], agent_id: str) -> LLMAgent:
        """
        Create an LLM agent from model configuration.
        
        Args:
            model_config: Model configuration dictionary
            agent_id: Unique agent identifier
        
        Returns:
            LLMAgent instance
        """
        return LLMAgent(
            agent_id=agent_id,
            persona_type=self.persona_type,
            model=model_config["model"],
            provider=model_config["provider"],
            temperature=model_config["temperature"],
        )
    
    def run_match(
        self,
        model_1: Dict[str, Any],
        model_2: Dict[str, Any],
        game_type: str,
        match_num: int = 1,
    ) -> Dict[str, Any]:
        """
        Run a single match between two models.
        
        Args:
            model_1: First model configuration
            model_2: Second model configuration
            game_type: Type of game to play
            match_num: Match number (for multiple matches per pairing)
        
        Returns:
            Match result dictionary
        """
        # Create agent IDs
        agent_1_id = f"{model_1['name']}_p1"
        agent_2_id = f"{model_2['name']}_p2"
        
        # Create agents
        agent_1 = self.create_agent(model_1, agent_1_id)
        agent_2 = self.create_agent(model_2, agent_2_id)
        
        # Create arena environment
        arena = create_arena_match(
            game_type=game_type,
            player_ids=(agent_1_id, agent_2_id),
            pie_size=100,
        )
        
        # Start match
        match_id = f"{self.tournament_id}_{game_type}_{model_1['name']}_vs_{model_2['name']}_m{match_num}"
        arena.start_match(match_id=match_id)
        
        print(f"  Match: {model_1['name']} vs {model_2['name']} ({game_type})")
        
        # Play the game
        try:
            if game_type == "nash_demand":
                result = self._play_nash_demand(arena, agent_1, agent_2)
            elif game_type == "ultimatum":
                result = self._play_ultimatum(arena, agent_1, agent_2)
            elif game_type == "rubinstein":
                result = self._play_rubinstein(arena, agent_1, agent_2)
            else:
                raise ValueError(f"Unknown game type: {game_type}")
            
            # Get match result
            match_result = arena.get_match_result()
            
            # Record in matchmaking system
            self.matchmaking.record_match(
                match_id=match_id,
                game_type=game_type,
                player_1_id=model_1['name'],
                player_2_id=model_2['name'],
                player_1_reward=match_result.final_rewards[agent_1_id],
                player_2_reward=match_result.final_rewards[agent_2_id],
            )
            
            # Store result
            result_dict = {
                "match_id": match_id,
                "game_type": game_type,
                "model_1": model_1['name'],
                "model_2": model_2['name'],
                "rewards": match_result.final_rewards,
                "outcome": match_result.outcome,
                "duration": match_result.duration_seconds,
            }
            
            self.match_results.append(result_dict)
            
            print(f"    Result: {model_1['name']}={match_result.final_rewards[agent_1_id]:.1f}, "
                  f"{model_2['name']}={match_result.final_rewards[agent_2_id]:.1f}")
            
            return result_dict
            
        except Exception as e:
            print(f"    ERROR: {e}")
            return {
                "match_id": match_id,
                "game_type": game_type,
                "model_1": model_1['name'],
                "model_2": model_2['name'],
                "error": str(e),
            }
    
    def _play_nash_demand(
        self,
        arena: ArenaEnvironment,
        agent_1: LLMAgent,
        agent_2: LLMAgent,
    ) -> Dict[str, Any]:
        """Play Nash Demand game."""
        # Get observations
        obs_1 = arena.get_observation(agent_1.agent_id)
        obs_2 = arena.get_observation(agent_2.agent_id)
        
        # Get actions
        action_1 = agent_1.choose_action(obs_1)
        action_2 = agent_2.choose_action(obs_2)
        
        # Execute round
        actions = {agent_1.agent_id: action_1, agent_2.agent_id: action_2}
        result, is_terminal = arena.execute_round(actions)
        
        return result
    
    def _play_ultimatum(
        self,
        arena: ArenaEnvironment,
        agent_1: LLMAgent,
        agent_2: LLMAgent,
    ) -> Dict[str, Any]:
        """Play Ultimatum game."""
        # Proposal phase
        obs_1 = arena.get_observation(agent_1.agent_id)
        offer = agent_1.choose_action(obs_1)
        
        result, is_terminal = arena.execute_round({agent_1.agent_id: offer})
        
        if is_terminal:
            return result
        
        # Response phase
        obs_2 = arena.get_observation(agent_2.agent_id)
        response = agent_2.choose_action(obs_2)
        
        result, is_terminal = arena.execute_round({agent_2.agent_id: response})
        
        return result
    
    def _play_rubinstein(
        self,
        arena: ArenaEnvironment,
        agent_1: LLMAgent,
        agent_2: LLMAgent,
        max_rounds: int = 10,
    ) -> Dict[str, Any]:
        """Play Rubinstein bargaining game."""
        current_proposer = agent_1
        current_responder = agent_2
        
        for round_num in range(max_rounds):
            # Proposal phase
            obs_proposer = arena.get_observation(current_proposer.agent_id)
            offer = current_proposer.choose_action(obs_proposer)
            
            result, is_terminal = arena.execute_round({current_proposer.agent_id: offer})
            
            if is_terminal:
                return result
            
            # Response phase
            obs_responder = arena.get_observation(current_responder.agent_id)
            response = current_responder.choose_action(obs_responder)
            
            result, is_terminal = arena.execute_round({current_responder.agent_id: response})
            
            if is_terminal:
                return result
            
            # Swap roles
            current_proposer, current_responder = current_responder, current_proposer
        
        # Max rounds reached - game ends
        return result
    
    def run_tournament(self) -> Dict[str, Any]:
        """
        Run the complete tournament.
        
        Returns:
            Tournament summary dictionary
        """
        print(f"\n{'='*60}")
        print(f"BATTLE OF THE GIANTS TOURNAMENT")
        print(f"Tournament ID: {self.tournament_id}")
        print(f"Models: {len(self.models)}")
        print(f"Game Types: {', '.join(self.game_types)}")
        print(f"{'='*60}\n")
        
        # Generate all pairings
        pairings = list(combinations(self.models, 2))
        total_matches = len(pairings) * len(self.game_types) * self.matches_per_pairing
        
        print(f"Total matches to play: {total_matches}\n")
        
        match_count = 0
        
        # Run all matches
        for model_1, model_2 in pairings:
            for game_type in self.game_types:
                for match_num in range(1, self.matches_per_pairing + 1):
                    match_count += 1
                    print(f"[{match_count}/{total_matches}]")
                    self.run_match(model_1, model_2, game_type, match_num)
        
        # Generate final results
        print(f"\n{'='*60}")
        print("TOURNAMENT COMPLETE")
        print(f"{'='*60}\n")
        
        # Save results
        self._save_results()
        
        # Print leaderboard
        self._print_leaderboard()
        
        return {
            "tournament_id": self.tournament_id,
            "total_matches": total_matches,
            "match_results": self.match_results,
        }
    
    def _save_results(self):
        """Save tournament results to files."""
        # Save match results
        results_file = self.output_dir / f"{self.tournament_id}_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "tournament_id": self.tournament_id,
                "timestamp": datetime.now().isoformat(),
                "models": self.models,
                "game_types": self.game_types,
                "matches": self.match_results,
            }, f, indent=2)
        
        print(f"Results saved to: {results_file}")
        
        # Save ratings
        self.matchmaking.save_to_json(f"{self.tournament_id}_ratings.json")
        
        # Export leaderboard
        self.matchmaking.export_leaderboard_csv(f"{self.tournament_id}_leaderboard.csv")
    
    def _print_leaderboard(self):
        """Print final leaderboard."""
        print("\nFINAL LEADERBOARD (by Elo):")
        print("-" * 80)
        print(f"{'Rank':<6} {'Model':<25} {'Elo':<10} {'Matches':<10} {'W-L-D':<15} {'Win%':<10}")
        print("-" * 80)
        
        leaderboard = self.matchmaking.get_leaderboard(sort_by="elo", min_matches=0)
        
        for rank, player in enumerate(leaderboard, 1):
            wld = f"{player.wins}-{player.losses}-{player.draws}"
            print(f"{rank:<6} {player.player_id:<25} {player.elo:<10.1f} "
                  f"{player.matches_played:<10} {wld:<15} {player.win_rate*100:<10.1f}")
        
        print("-" * 80)


def run_battle_of_giants(
    models: List[Dict[str, Any]] = None,
    game_types: List[str] = None,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    Convenience function to run the Battle of the Giants tournament.
    
    Args:
        models: List of model configurations (default: TOURNAMENT_MODELS)
        game_types: List of game types (default: all)
        output_dir: Output directory (default: ./tournament_results)
    
    Returns:
        Tournament summary
    """
    models = models or TOURNAMENT_MODELS
    
    runner = TournamentRunner(
        models=models,
        game_types=game_types,
        output_dir=output_dir,
    )
    
    return runner.run_tournament()


if __name__ == "__main__":
    # Run the tournament
    run_battle_of_giants()
