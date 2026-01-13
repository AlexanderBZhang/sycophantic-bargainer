#!/usr/bin/env python3
"""
Battle of the Giants: LLM Negotiation Arena Tournament

Execute a tournament between SOTA models: Claude Opus 4.5, Gemini 3 Pro, GPT 5.1 High.
Uses the Negotiation Arena environment for natural language bargaining.

Supports parallel execution of matches (each match uses different providers).
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Tuple
import threading

# Load environment variables from sycophantic_bargainer/.env
load_dotenv(Path(__file__).parent / ".env")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environments.negotiation_arena import NegotiationArena
from agents.llm_agent import LLMAgent

# Thread-safe print lock
print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    """Thread-safe print function."""
    with print_lock:
        print(*args, **kwargs)


def check_api_keys():
    """
    Check that required API keys are available.
    Does NOT print or display the keys themselves.
    """
    required_keys = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic",
        "GOOGLE_API_KEY": "Google",
    }
    
    missing_keys = []
    for key_name, provider in required_keys.items():
        if not os.environ.get(key_name):
            missing_keys.append(f"{provider} ({key_name})")
    
    if missing_keys:
        print("ERROR: Missing required API keys:")
        for key in missing_keys:
            print(f"  - {key}")
        print("\nPlease set the required environment variables before running.")
        return False
    
    print("✓ All required API keys are present")
    return True


def run_negotiation_match(
    agent1: LLMAgent, 
    agent2: LLMAgent, 
    arena: NegotiationArena, 
    match_id: str = "",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run a single negotiation match between two agents.
    
    Returns:
        dict: Match results including payoffs, transcript, and metrics
    """
    state = arena.reset()
    
    agents = {arena.agent_A: agent1, arena.agent_B: agent2}
    rewards = {arena.agent_A: 0.0, arena.agent_B: 0.0}  # Initialize rewards
    
    prefix = f"[{match_id}] " if match_id else ""
    
    if verbose:
        safe_print(f"\n  {prefix}Starting negotiation (max {arena.max_rounds} rounds)...")
        safe_print(f"  {prefix}First player: {arena.current_player_id}")
    
    done = False
    turn = 0
    max_turns = arena.max_rounds * 2 + 5  # Safety limit
    
    while not done and turn < max_turns:
        current_agent = agents[arena.current_player_id]
        obs = arena.get_observation(arena.current_player_id)
        
        try:
            # Get agent's response
            response = current_agent.choose_action(obs)
            
            # Execute the action
            state, rewards, done = arena.step({arena.current_player_id: response})
            
            if verbose and len(arena.conversation_history) > 0:
                last_msg = arena.conversation_history[-1]
                safe_print(f"    {prefix}[R{last_msg.round}] {last_msg.player_id}: {last_msg.message[:50]}...")
                
        except Exception as e:
            if verbose:
                safe_print(f"    {prefix}ERROR: {e}")
            break
            
        turn += 1
    
    # Get final results
    metrics = arena.get_metrics()
    transcript = arena.get_conversation_transcript()
    
    if verbose:
        safe_print(f"\n  {prefix}Outcome: {metrics['outcome']}")
        safe_print(f"  {prefix}Payoffs: {agent1.agent_id}=${rewards.get(arena.agent_A, 0):.1f}M, {agent2.agent_id}=${rewards.get(arena.agent_B, 0):.1f}M")
        utilities = metrics.get('utilities', {})
        safe_print(f"  {prefix}Utilities: {agent1.agent_id}={utilities.get(arena.agent_A, 0):.0f}, {agent2.agent_id}={utilities.get(arena.agent_B, 0):.0f}")
    
    return {
        "payoffs": rewards,
        "metrics": metrics,
        "transcript": transcript,
        "rounds": arena.current_round,
        "api_calls": {
            "player_1": agent1.api_call_history,
            "player_2": agent2.api_call_history,
        }
    }


def run_single_match(match_config: Tuple[int, Dict[str, Any], Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run a single match - designed to be called in parallel.
    
    Args:
        match_config: Tuple of (match_number, model1_config, model2_config)
    
    Returns:
        Match result dictionary
    """
    match_num, model1, model2 = match_config
    match_id = f"M{match_num}"
    
    safe_print(f"\n{'='*60}")
    safe_print(f"[{match_id}] STARTING: {model1['name']} vs {model2['name']}")
    safe_print(f"{'='*60}")
    
    try:
        # Create agents (same prompt, different roles)
        agent1 = LLMAgent(
            agent_id="player_1",
            persona_type="default",
            model=model1["model"],
            provider=model1["provider"],
            temperature=model1["temperature"],
            reasoning_effort=model1.get("reasoning_effort"),
        )
        agent2 = LLMAgent(
            agent_id="player_2",
            persona_type="default",
            model=model2["model"],
            provider=model2["provider"],
            temperature=model2["temperature"],
            reasoning_effort=model2.get("reasoning_effort"),
        )
        
        # Create arena
        arena = NegotiationArena(
            pot_size=100.0,
            agent_ids=("player_1", "player_2"),
            max_rounds=10,
            shrinking_pot=False,
            random_first_player=True,
            reservation_value=10.0,
        )
        
        # Run match
        result = run_negotiation_match(agent1, agent2, arena, match_id=match_id, verbose=True)
        result["model1"] = model1["name"]
        result["model2"] = model2["name"]
        result["match_id"] = match_id
        
        safe_print(f"\n[{match_id}] COMPLETE: {model1['name']} vs {model2['name']}")
        return result
        
    except Exception as e:
        safe_print(f"\n[{match_id}] ERROR: {type(e).__name__}: {e}")
        return {
            "model1": model1["name"],
            "model2": model2["name"],
            "match_id": match_id,
            "error": str(e),
        }


def main():
    """Run the Battle of the Giants tournament."""
    print("\n" + "="*70)
    print("BATTLE OF THE GIANTS: NEGOTIATION ARENA")
    print("="*70)
    print("\nThis tournament will test the following SOTA models:")
    print("  • Claude Opus 4.5 (Anthropic)")
    print("  • Gemini 3 Pro (Google)")
    print("  • GPT 5.1 High (OpenAI)")
    print("\nGame: Negotiation Arena")
    print("  • $100M joint venture to divide")
    print("  • Natural language dialogue")
    print("  • Max 10 moves")
    print("  • $10M salvage if no deal")
    print("  • Utility: U(x) = 300 + 10*(x-50) - 20*max(0, 40-x)")
    print("\nFormat: Round-robin (3 matches total)")
    print("Execution: PARALLEL (all matches run simultaneously)")
    print("="*70 + "\n")
    
    # Check API keys
    if not check_api_keys():
        sys.exit(1)
    
    # Confirm execution
    print("\nEstimated cost: $1-3 (SOTA models are expensive)")
    print("Estimated time: ~1-2 minutes (parallel execution)")
    
    response = input("\nProceed with tournament? [y/N]: ").strip().lower()
    if response != 'y':
        print("Tournament cancelled.")
        sys.exit(0)
    
    print("\n" + "="*70)
    print("STARTING PARALLEL TOURNAMENT")
    print("="*70 + "\n")
    
    # Define the tournament models (SOTA Dec 2025)
    tournament_models = [
        {"name": "Claude-Opus-4.5", "provider": "anthropic", "model": "claude-opus-4-5-20251101", "temperature": 1.0},
        {"name": "Gemini-3-Pro", "provider": "google", "model": "gemini-3-pro-preview", "temperature": 0.7},
        {"name": "GPT-5.1-High", "provider": "openai", "model": "gpt-5.1", "temperature": 1.0, "reasoning_effort": "high"},
    ]
    
    # Generate pairings
    from itertools import combinations
    pairings = list(combinations(tournament_models, 2))
    
    # Create match configs
    match_configs = [
        (i, model1, model2) 
        for i, (model1, model2) in enumerate(pairings, 1)
    ]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    
    # Run all matches in parallel
    print(f"Launching {len(match_configs)} matches in parallel...")
    print("(Output will be interleaved - each match prefixed with [M1], [M2], [M3])\n")
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all matches
        future_to_match = {
            executor.submit(run_single_match, config): config 
            for config in match_configs
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_match):
            config = future_to_match[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({
                    "model1": config[1]["name"],
                    "model2": config[2]["name"],
                    "match_id": f"M{config[0]}",
                    "error": str(e),
                })
    
    # Sort results by match_id for consistent output
    results.sort(key=lambda x: x.get("match_id", "M0"))
    
    # Print summary
    print("\n" + "="*70)
    print("TOURNAMENT COMPLETE!")
    print("="*70)
    
    print("\n--- RESULTS SUMMARY ---")
    print(f"{'Match':<35} {'Payoffs':<15} {'Utilities':<20} {'Outcome':<12}")
    print("-" * 82)
    
    for r in results:
        match_name = f"{r['model1']} vs {r['model2']}"
        if "error" in r:
            print(f"{match_name:<35} ERROR: {r['error'][:30]}")
        else:
            p1_reward = r['payoffs'].get('player_1', 0)
            p2_reward = r['payoffs'].get('player_2', 0)
            utilities = r['metrics'].get('utilities', {})
            u1 = utilities.get('player_1', 0)
            u2 = utilities.get('player_2', 0)
            outcome = r['metrics']['outcome']
            print(f"{match_name:<35} ${p1_reward:.0f}M/${p2_reward:.0f}M    U:{u1:.0f}/{u2:.0f}        {outcome}")
    
    # Save results
    import json
    output_dir = Path(__file__).parent / "tournament_results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"negotiation_arena_{timestamp}.json"
    
    with open(output_file, "w") as f:
        serializable_results = []
        for r in results:
            sr = r.copy()
            if "metrics" in sr:
                sr["metrics"] = {k: v for k, v in sr["metrics"].items()}
            serializable_results.append(sr)
        json.dump({"timestamp": timestamp, "matches": serializable_results}, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
