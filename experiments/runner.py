"""
Experiment runner for bargaining simulations.

Orchestrates games between agents and collects results.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from pathlib import Path

from ..environments.base import BargainEnvironment
from ..agents.base import Agent
from .provenance import ProvenanceTracker, create_provenance_for_experiment


class ExperimentRunner:
    """
    Runs bargaining experiments and collects results.

    Coordinates agents, environments, and data collection.
    """

    def __init__(
        self,
        environment: BargainEnvironment,
        agent_A: Agent,
        agent_B: Agent,
        num_simulations: int = 100,
        verbose: bool = True,
        track_provenance: bool = True,
    ):
        """
        Initialize experiment runner.

        Args:
            environment: Bargaining environment to use
            agent_A: First agent
            agent_B: Second agent
            num_simulations: Number of games to run
            verbose: Whether to show progress bar
            track_provenance: Whether to track experiment metadata for reproducibility
        """
        self.environment = environment
        self.agent_A = agent_A
        self.agent_B = agent_B
        self.num_simulations = num_simulations
        self.verbose = verbose
        self.results: List[Dict[str, Any]] = []

        # Initialize provenance tracker
        if track_provenance:
            self.provenance = create_provenance_for_experiment(
                environment=environment,
                agent_A=agent_A,
                agent_B=agent_B,
                num_simulations=num_simulations,
            )
        else:
            self.provenance = None

    def run(self) -> List[Dict[str, Any]]:
        """
        Run all simulations.

        Returns:
            List of game results
        """
        self.results = []

        iterator = range(self.num_simulations)
        if self.verbose:
            iterator = tqdm(iterator, desc="Running simulations")

        for sim_num in iterator:
            result = self._run_single_game(sim_num)
            self.results.append(result)

        return self.results

    def _run_single_game(self, sim_num: int) -> Dict[str, Any]:
        """
        Run a single game.

        Args:
            sim_num: Simulation number

        Returns:
            Game result dictionary
        """
        # Reset environment and agents
        state = self.environment.reset()
        self.agent_A.reset()
        self.agent_B.reset()

        # Map agent IDs to agents
        agent_map = {
            self.environment.agent_A: self.agent_A,
            self.environment.agent_B: self.agent_B,
        }

        # Game loop
        done = False
        step_count = 0
        max_steps = 100  # Safety limit

        while not done and step_count < max_steps:
            # Determine which agents need to act
            actions = {}

            # Get observations and choose actions
            for agent_id, agent in agent_map.items():
                obs = self.environment.get_observation(agent_id)

                # Check if this agent needs to act
                if self._should_agent_act(obs):
                    try:
                        action = agent.choose_action(obs)
                        actions[agent_id] = action
                    except Exception as e:
                        # Handle agent errors gracefully
                        if self.verbose:
                            print(f"Agent {agent_id} error: {e}")
                        # Use default action
                        actions[agent_id] = self._get_default_action(obs)

            # Execute actions if any
            if actions:
                try:
                    state, rewards, done = self.environment.step(actions)
                except Exception as e:
                    # Handle environment errors
                    if self.verbose:
                        print(f"Environment error: {e}")
                    # Terminate with zero payoffs
                    rewards = {self.environment.agent_A: 0.0, self.environment.agent_B: 0.0}
                    done = True

            step_count += 1

        # Collect results
        final_payoffs = self.environment.get_rewards()

        result = {
            "simulation_num": sim_num,
            "agent_A_id": self.environment.agent_A,
            "agent_B_id": self.environment.agent_B,
            "agent_A_persona": self.agent_A.persona_type,
            "agent_B_persona": self.agent_B.persona_type,
            "payoffs": final_payoffs,
            "steps": step_count,
            "metadata": self.environment.state.metadata if self.environment.state else {},
        }

        return result

    def _should_agent_act(self, observation: Dict[str, Any]) -> bool:
        """
        Determine if an agent should act based on observation.

        Args:
            observation: Agent's observation

        Returns:
            True if agent should act
        """
        game_type = observation.get("game_type")

        if game_type == "nash_demand":
            # Both agents act simultaneously at start
            return not observation.get("is_terminal", False)

        elif game_type == "ultimatum":
            role = observation.get("role")
            phase = observation.get("current_phase")

            if role == "proposer" and phase == "proposal":
                return True
            elif role == "responder" and phase == "response":
                return True

        elif game_type == "rubinstein":
            waiting_for = observation.get("waiting_for")
            is_proposer = observation.get("is_proposer", False)

            if waiting_for == "offer" and is_proposer:
                return True
            elif waiting_for == "response" and not is_proposer:
                return True

        return False

    def _get_default_action(self, observation: Dict[str, Any]) -> Any:
        """
        Get a default/fallback action.

        Args:
            observation: Agent's observation

        Returns:
            Default action
        """
        game_type = observation.get("game_type")
        pie_size = observation.get("pie_size", 10)

        # For offers/demands: use equal split
        if game_type in ["nash_demand", "ultimatum", "rubinstein"]:
            role = observation.get("role")
            waiting_for = observation.get("waiting_for")

            if role == "responder" or waiting_for == "response":
                # Default to accepting
                return True
            else:
                # Default to equal split
                return pie_size / 2

        return None

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame.

        Returns:
            DataFrame with results
        """
        if not self.results:
            return pd.DataFrame()

        # Flatten results
        rows = []
        for result in self.results:
            row = {
                "simulation_num": result["simulation_num"],
                "agent_A_id": result["agent_A_id"],
                "agent_B_id": result["agent_B_id"],
                "agent_A_persona": result["agent_A_persona"],
                "agent_B_persona": result["agent_B_persona"],
                "agent_A_payoff": result["payoffs"].get(result["agent_A_id"], 0),
                "agent_B_payoff": result["payoffs"].get(result["agent_B_id"], 0),
                "total_payoff": sum(result["payoffs"].values()),
                "steps": result["steps"],
            }

            # Add metadata
            for key, value in result["metadata"].items():
                if isinstance(value, (int, float, str, bool)):
                    row[f"meta_{key}"] = value

            rows.append(row)

        return pd.DataFrame(rows)

    def save_results(self, filepath: str, save_provenance: bool = True):
        """
        Save results to CSV.

        Args:
            filepath: Path to save file
            save_provenance: Whether to save provenance metadata alongside results
        """
        df = self.get_results_dataframe()
        df.to_csv(filepath, index=False)

        # Save provenance metadata if available
        if save_provenance and self.provenance is not None:
            # Create provenance filename (same as results but with .metadata.json suffix)
            provenance_path = Path(filepath).with_suffix(".metadata.json")
            self.provenance.save(str(provenance_path))
            if self.verbose:
                print(f"Provenance metadata saved to: {provenance_path}")

    def save_provenance(self, filepath: str):
        """
        Explicitly save provenance metadata to a JSON file.

        Args:
            filepath: Path to save provenance metadata
        """
        if self.provenance is None:
            raise ValueError("Provenance tracking is not enabled for this runner")
        self.provenance.save(filepath)

    def print_summary(self):
        """Print a summary of results."""
        if not self.results:
            print("No results to summarize.")
            return

        df = self.get_results_dataframe()

        print("\n=== Experiment Summary ===")
        print(f"Total simulations: {len(self.results)}")
        print(f"\nAgent A ({self.agent_A.persona_type}):")
        print(f"  Mean payoff: {df['agent_A_payoff'].mean():.2f}")
        print(f"  Std: {df['agent_A_payoff'].std():.2f}")

        print(f"\nAgent B ({self.agent_B.persona_type}):")
        print(f"  Mean payoff: {df['agent_B_payoff'].mean():.2f}")
        print(f"  Std: {df['agent_B_payoff'].std():.2f}")

        print(f"\nTotal surplus:")
        print(f"  Mean: {df['total_payoff'].mean():.2f}")

        # Agreement rate (non-zero payoffs)
        agreements = (df['total_payoff'] > 0).sum()
        agreement_rate = agreements / len(df) * 100
        print(f"\nAgreement rate: {agreement_rate:.1f}%")


class BatchExperimentRunner:
    """
    Run multiple experiments across different configurations.

    Tests different agent matchups and environments.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize batch runner.

        Args:
            verbose: Whether to show progress
        """
        self.verbose = verbose
        self.all_results: List[Dict[str, Any]] = []

    def run_experiment_grid(
        self,
        environments: List[BargainEnvironment],
        agent_pairs: List[tuple],
        num_simulations: int = 100,
    ) -> pd.DataFrame:
        """
        Run experiments across a grid of configurations.

        Args:
            environments: List of environments to test
            agent_pairs: List of (agent_A, agent_B) tuples
            num_simulations: Simulations per configuration

        Returns:
            DataFrame with all results
        """
        total_experiments = len(environments) * len(agent_pairs)

        if self.verbose:
            print(f"Running {total_experiments} experiments...")

        for env in environments:
            for agent_A, agent_B in agent_pairs:
                if self.verbose:
                    print(f"\n{env.__class__.__name__}: {agent_A.persona_type} vs {agent_B.persona_type}")

                runner = ExperimentRunner(
                    environment=env,
                    agent_A=agent_A,
                    agent_B=agent_B,
                    num_simulations=num_simulations,
                    verbose=self.verbose,
                )

                results = runner.run()

                # Add environment type to results
                for result in results:
                    result["environment"] = env.__class__.__name__

                self.all_results.extend(results)

        # Convert to DataFrame
        return self._results_to_dataframe()

    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert all results to DataFrame."""
        if not self.all_results:
            return pd.DataFrame()

        rows = []
        for result in self.all_results:
            row = {
                "environment": result.get("environment"),
                "simulation_num": result["simulation_num"],
                "agent_A_id": result["agent_A_id"],
                "agent_B_id": result["agent_B_id"],
                "agent_A_persona": result["agent_A_persona"],
                "agent_B_persona": result["agent_B_persona"],
                "agent_A_payoff": result["payoffs"].get(result["agent_A_id"], 0),
                "agent_B_payoff": result["payoffs"].get(result["agent_B_id"], 0),
                "total_payoff": sum(result["payoffs"].values()),
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def save_results(self, filepath: str):
        """Save all results to CSV."""
        df = self._results_to_dataframe()
        df.to_csv(filepath, index=False)
