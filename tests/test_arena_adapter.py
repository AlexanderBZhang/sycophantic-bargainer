"""
Test the Arena Adapter to verify it works correctly.
"""

import sys
sys.path.insert(0, '/workspace/sycophantic_bargainer')

from arena.adapter import ArenaEnvironment, ArenaMatchRunner, create_arena_match


def test_nash_demand_arena():
    """Test Nash Demand game through arena adapter."""
    print("\n=== Testing Nash Demand Game in Arena ===")
    
    # Create arena environment
    arena = create_arena_match(
        game_type="nash_demand",
        player_ids=("alice", "bob"),
        pie_size=100
    )
    
    # Start match
    match_info = arena.start_match(match_id="test_nash_001")
    print(f"Match ID: {match_info['match_id']}")
    print(f"Game Type: {match_info['game_type']}")
    print(f"Players: {match_info['player_ids']}")
    
    # Execute moves
    actions = {
        "alice": 60,
        "bob": 50
    }
    
    # Validate moves
    for player_id, action in actions.items():
        is_valid, error = arena.execute_move(player_id, action)
        print(f"{player_id} demands {action}: {'✓ Valid' if is_valid else f'✗ Invalid - {error}'}")
    
    # Execute round
    result, is_terminal = arena.execute_round(actions)
    print(f"\nRound executed. Terminal: {is_terminal}")
    print(f"Rewards: {result['rewards']}")
    
    # Get match result
    if is_terminal:
        match_result = arena.get_match_result()
        print(f"\n=== Match Result ===")
        print(f"Outcome: {match_result.outcome}")
        print(f"Final Rewards: {match_result.final_rewards}")
        print(f"Duration: {match_result.duration_seconds:.3f}s")
        print(f"Total Moves: {len(match_result.moves)}")
        
        # Test serialization
        json_str = match_result.to_json()
        print(f"\n✓ Serialization works (JSON length: {len(json_str)} chars)")
    
    return match_result


def test_ultimatum_arena():
    """Test Ultimatum game through arena adapter."""
    print("\n=== Testing Ultimatum Game in Arena ===")
    
    arena = create_arena_match(
        game_type="ultimatum",
        player_ids=("proposer", "responder"),
        pie_size=100
    )
    
    match_info = arena.start_match(match_id="test_ultimatum_001")
    print(f"Match ID: {match_info['match_id']}")
    
    # Proposer offers 30 to responder (keeps 70)
    actions = {
        "proposer": 70,  # Proposer keeps 70
        "responder": True  # Accept
    }
    
    # First step: proposal
    is_valid, error = arena.execute_move("proposer", actions["proposer"])
    print(f"proposer offers to keep {actions['proposer']}: {'✓ Valid' if is_valid else f'✗ Invalid - {error}'}")
    
    result, is_terminal = arena.execute_round({"proposer": actions["proposer"]})
    print(f"Proposal phase complete. Terminal: {is_terminal}")
    
    # Second step: response
    if not is_terminal:
        is_valid, error = arena.execute_move("responder", actions["responder"])
        print(f"responder {'accepts' if actions['responder'] else 'rejects'}: {'✓ Valid' if is_valid else f'✗ Invalid - {error}'}")
        
        result, is_terminal = arena.execute_round({"responder": actions["responder"]})
        print(f"Response phase complete. Terminal: {is_terminal}")
        print(f"Rewards: {result['rewards']}")
    
    if is_terminal:
        match_result = arena.get_match_result()
        print(f"\n=== Match Result ===")
        print(f"Outcome: {match_result.outcome}")
        print(f"Final Rewards: {match_result.final_rewards}")
    
    return match_result


def test_rubinstein_arena():
    """Test Rubinstein bargaining through arena adapter."""
    print("\n=== Testing Rubinstein Bargaining in Arena ===")
    
    arena = create_arena_match(
        game_type="rubinstein",
        player_ids=("player1", "player2"),
        pie_size=100,
        discount_factor=0.9,
        max_rounds=5
    )
    
    match_info = arena.start_match(match_id="test_rubinstein_001")
    print(f"Match ID: {match_info['match_id']}")
    
    # Player 1 proposes 60/40 split
    actions = {
        "player1": 60,
        "player2": True  # Accept
    }
    
    # First step: proposal
    is_valid, error = arena.execute_move("player1", actions["player1"])
    print(f"player1 proposes to keep {actions['player1']}: {'✓ Valid' if is_valid else f'✗ Invalid - {error}'}")
    
    result, is_terminal = arena.execute_round({"player1": actions["player1"]})
    print(f"Proposal phase complete. Terminal: {is_terminal}")
    
    # Second step: response
    if not is_terminal:
        is_valid, error = arena.execute_move("player2", actions["player2"])
        print(f"player2 {'accepts' if actions['player2'] else 'rejects'}: {'✓ Valid' if is_valid else f'✗ Invalid - {error}'}")
        
        result, is_terminal = arena.execute_round({"player2": actions["player2"]})
        print(f"Response phase complete. Terminal: {is_terminal}")
        print(f"Rewards: {result['rewards']}")
    
    if is_terminal:
        match_result = arena.get_match_result()
        print(f"\n=== Match Result ===")
        print(f"Outcome: {match_result.outcome}")
        print(f"Final Rewards: {match_result.final_rewards}")
    
    return match_result


def test_arena_match_runner():
    """Test the high-level ArenaMatchRunner."""
    print("\n=== Testing ArenaMatchRunner ===")
    
    arena = create_arena_match(
        game_type="nash_demand",
        player_ids=("agent_a", "agent_b"),
        pie_size=100
    )
    
    runner = ArenaMatchRunner(arena)
    
    # Run a complete match
    actions = {
        "agent_a": 45,
        "agent_b": 55
    }
    
    match_result = runner.run_match(actions, match_id="runner_test_001")
    
    print(f"Match ID: {match_result.match_id}")
    print(f"Outcome: {match_result.outcome}")
    print(f"Final Rewards: {match_result.final_rewards}")
    print(f"Duration: {match_result.duration_seconds:.3f}s")
    
    return match_result


def test_list_games():
    """Test listing available games."""
    print("\n=== Available Games ===")
    games = ArenaEnvironment.list_available_games()
    for game in games:
        print(f"  - {game}")
    print(f"\nTotal: {len(games)} games available")


if __name__ == "__main__":
    print("=" * 60)
    print("ARENA ADAPTER TEST SUITE")
    print("=" * 60)
    
    try:
        # Test listing games
        test_list_games()
        
        # Test each game type
        nash_result = test_nash_demand_arena()
        ultimatum_result = test_ultimatum_arena()
        rubinstein_result = test_rubinstein_arena()
        
        # Test the runner
        runner_result = test_arena_match_runner()
        
        print("\n" + "=" * 60)
        print("✓ ALL ARENA ADAPTER TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
