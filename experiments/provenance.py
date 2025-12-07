"""
Experiment provenance tracking for reproducibility.

Captures metadata about experiments including:
- Timestamp
- Git commit hash (if available)
- Python version
- Library versions
- Model configurations
- Experiment parameters
"""

import datetime
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional


class ProvenanceTracker:
    """
    Tracks experiment metadata for reproducibility.

    Automatically captures environment information, git state,
    and experiment configuration.
    """

    def __init__(self):
        """Initialize provenance tracker."""
        self.metadata: Dict[str, Any] = {}
        self._collect_system_info()
        self._collect_git_info()

    def _collect_system_info(self):
        """Collect system and Python environment information."""
        self.metadata["timestamp"] = datetime.datetime.utcnow().isoformat()
        self.metadata["python_version"] = sys.version
        self.metadata["platform"] = {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        }

        # Collect library versions
        try:
            import openai
            self.metadata["openai_version"] = openai.__version__
        except (ImportError, AttributeError):
            self.metadata["openai_version"] = None

        try:
            import anthropic
            self.metadata["anthropic_version"] = anthropic.__version__
        except (ImportError, AttributeError):
            self.metadata["anthropic_version"] = None

        try:
            import pandas
            self.metadata["pandas_version"] = pandas.__version__
        except (ImportError, AttributeError):
            self.metadata["pandas_version"] = None

    def _collect_git_info(self):
        """Collect git repository information if available."""
        self.metadata["git"] = {}

        try:
            # Get current commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self.metadata["git"]["commit_hash"] = result.stdout.strip()

            # Get current branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self.metadata["git"]["branch"] = result.stdout.strip()

            # Check if working directory is clean
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self.metadata["git"]["clean_working_dir"] = len(result.stdout.strip()) == 0

            # Get last commit message
            result = subprocess.run(
                ["git", "log", "-1", "--pretty=%B"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self.metadata["git"]["last_commit_message"] = result.stdout.strip()

        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            # Git not available or not a git repo
            self.metadata["git"]["error"] = str(e)

    def add_experiment_config(
        self,
        environment_name: str,
        environment_params: Dict[str, Any],
        agent_A_config: Dict[str, Any],
        agent_B_config: Dict[str, Any],
        num_simulations: int,
    ):
        """
        Add experiment configuration to metadata.

        Args:
            environment_name: Name of the bargaining environment
            environment_params: Environment parameters (pie_size, etc.)
            agent_A_config: Agent A configuration (persona, model, etc.)
            agent_B_config: Agent B configuration
            num_simulations: Number of simulations to run
        """
        self.metadata["experiment"] = {
            "environment": environment_name,
            "environment_params": environment_params,
            "agent_A": agent_A_config,
            "agent_B": agent_B_config,
            "num_simulations": num_simulations,
        }

    def add_custom_metadata(self, key: str, value: Any):
        """
        Add custom metadata field.

        Args:
            key: Metadata key
            value: Metadata value (must be JSON-serializable)
        """
        self.metadata[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """
        Get all metadata as dictionary.

        Returns:
            Complete metadata dictionary
        """
        return self.metadata.copy()

    def to_json(self, indent: int = 2) -> str:
        """
        Serialize metadata to JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string
        """
        return json.dumps(self.metadata, indent=indent, default=str)

    def save(self, filepath: str):
        """
        Save metadata to JSON file.

        Args:
            filepath: Path to save metadata
        """
        with open(filepath, "w") as f:
            f.write(self.to_json())

    @staticmethod
    def load(filepath: str) -> "ProvenanceTracker":
        """
        Load metadata from JSON file.

        Args:
            filepath: Path to metadata file

        Returns:
            ProvenanceTracker instance with loaded metadata
        """
        tracker = ProvenanceTracker()
        with open(filepath, "r") as f:
            tracker.metadata = json.load(f)
        return tracker


def get_agent_config(agent) -> Dict[str, Any]:
    """
    Extract configuration from an agent instance.

    Args:
        agent: Agent instance

    Returns:
        Agent configuration dictionary
    """
    config = {
        "agent_id": agent.agent_id,
        "persona_type": agent.persona_type,
    }

    # Add LLM-specific info if available
    if hasattr(agent, "model"):
        config["model"] = agent.model
    if hasattr(agent, "provider"):
        config["provider"] = agent.provider
    if hasattr(agent, "temperature"):
        config["temperature"] = agent.temperature
    if hasattr(agent, "strategy"):
        config["strategy"] = agent.strategy

    return config


def get_environment_config(environment) -> tuple[str, Dict[str, Any]]:
    """
    Extract configuration from an environment instance.

    Args:
        environment: BargainEnvironment instance

    Returns:
        Tuple of (environment_name, environment_params)
    """
    env_name = environment.__class__.__name__

    params = {}
    if hasattr(environment, "pie_size"):
        params["pie_size"] = environment.pie_size
    if hasattr(environment, "discount_factor"):
        params["discount_factor"] = environment.discount_factor
    if hasattr(environment, "max_rounds"):
        params["max_rounds"] = environment.max_rounds
    if hasattr(environment, "acceptance_threshold"):
        params["acceptance_threshold"] = environment.acceptance_threshold

    return env_name, params


def create_provenance_for_experiment(
    environment,
    agent_A,
    agent_B,
    num_simulations: int,
    custom_metadata: Optional[Dict[str, Any]] = None,
) -> ProvenanceTracker:
    """
    Create a complete provenance tracker for an experiment.

    Args:
        environment: BargainEnvironment instance
        agent_A: First agent
        agent_B: Second agent
        num_simulations: Number of simulations
        custom_metadata: Optional custom metadata to include

    Returns:
        ProvenanceTracker with all metadata collected
    """
    tracker = ProvenanceTracker()

    # Add experiment configuration
    env_name, env_params = get_environment_config(environment)
    agent_A_config = get_agent_config(agent_A)
    agent_B_config = get_agent_config(agent_B)

    tracker.add_experiment_config(
        environment_name=env_name,
        environment_params=env_params,
        agent_A_config=agent_A_config,
        agent_B_config=agent_B_config,
        num_simulations=num_simulations,
    )

    # Add any custom metadata
    if custom_metadata:
        for key, value in custom_metadata.items():
            tracker.add_custom_metadata(key, value)

    return tracker
