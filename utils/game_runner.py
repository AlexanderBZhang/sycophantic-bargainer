"""
Game Runner Utility

Orchestrates game loops for bargaining environments.
"""

from typing import Dict, Any, Tuple
from agents.base import Agent
from environments.base import BargainEnvironment


def run_game(env: BargainEnvironment, agent1: Agent, agent2: Agent) -> Dict[str, Any]:
    """
    Run a complete game between two agents.
    
    Args:
        env: Bargaining environment
        agent1: First agent
        agent2: Second agent
    
    Returns:
        Dictionary containing:
        - rewards: Final rewards for each agent
        - rounds: Number of rounds played
        - agreement: Whether an agreement was reached
        - history: Game history
    """
    # Reset environment
    state = env.reset()
    
    # Map agent IDs to agents
    agent_map = {
        env.agent_A: agent1,
        env.agent_B: agent2,
    }
    
    rounds = 0
    max_rounds = 100  # Safety limit
    
    # Game loop
    while not env.is_terminal() and rounds < max_rounds:
        # Get observations for each agent
        obs_A = env.get_observation(env.agent_A)
        obs_B = env.get_observation(env.agent_B)
        
        # Get actions from agents
        action_A = agent1.choose_action(obs_A)
        action_B = agent2.choose_action(obs_B)
        
        # Step environment
        actions = {
            env.agent_A: action_A,
            env.agent_B: action_B,
        }
        
        state, rewards, done = env.step(actions)
        rounds += 1
        
        if done:
            break
    
    # Get final rewards
    final_rewards = env.get_rewards()
    
    # Determine if agreement was reached
    agreement = sum(final_rewards.values()) > 0
    
    return {
        "rewards": final_rewards,
        "rounds": rounds,
        "agreement": agreement,
        "history": env.history,
    }


def run_sequential_game(env: BargainEnvironment, agent1: Agent, agent2: Agent) -> Dict[str, Any]:
    """
    Run a sequential bargaining game (e.g., Ultimatum, Rubinstein).
    
    These games have alternating turns rather than simultaneous moves.
    
    Args:
        env: Bargaining environment
        agent1: First agent (typically proposer)
        agent2: Second agent (typically responder)
    
    Returns:
        Dictionary containing game outcome
    """
    # Reset environment
    state = env.reset()
    
    rounds = 0
    max_rounds = 100
    
    # Game loop
    while not env.is_terminal() and rounds < max_rounds:
        # Determine whose turn it is
        current_player = state.get("current_player", env.agent_A)
        
        # Get observation for current player
        obs = env.get_observation(current_player)
        
        # Get action from appropriate agent
        if current_player == env.agent_A:
            action = agent1.choose_action(obs)
        else:
            action = agent2.choose_action(obs)
        
        # Step environment with single action
        actions = {current_player: action}
        state, rewards, done = env.step(actions)
        rounds += 1
        
        if done:
            break
    
    # Get final rewards
    final_rewards = env.get_rewards()
    
    # Determine if agreement was reached
    agreement = sum(final_rewards.values()) > 0
    
    return {
        "rewards": final_rewards,
        "rounds": rounds,
        "agreement": agreement,
        "history": env.history,
    }
