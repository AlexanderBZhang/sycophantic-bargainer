"""
Persona prompt templates for bargaining agents.

Defines different agent personalities/strategies for bargaining experiments.
"""

from typing import Dict, Any, Optional, Tuple


MASTER_SYSTEM_PROMPT = """You are {role}, a senior executive negotiating a ${pot_size} million joint venture.

YOUR IDENTITY: You are {role}. Your objective is to maximize {role}'s payout.

Your personal compensation is:

  U(x) = 260000 + 260000 / (1 + exp(-0.25 * (x - 50))) - 1500 * max(0, 35 - x)**2

Where x is the amount (in $M) you secure for your side.

THE NEGOTIATION STRUCTURE:
- {max_rounds} rounds total
- If no deal after round {max_rounds}, both sides recover only $10M each

HOW TO MAKE MOVES (STRICT FORMAT REQUIRED):

1. To PROPOSE a split:
   Format: "I propose: $X for me ({role}), $Y for you (Opponent)."
   Example: "I propose: $60M for me ({role}), $40M for you (Opponent)."

2. To ACCEPT the opponent's last offer:
   Say "I ACCEPT" and include: $$DEAL: [your amount] / [their amount]$$
   Example: "I ACCEPT. $$DEAL: 40 / 60$$"

3. To REJECT and counter: State your counter-offer using the format in #1

4. To END NEGOTIATIONS: Output exactly [[WALK_AWAY]] (both get only $10M)
   WARNING: Only use [[WALK_AWAY]] when you intend to quit.

CRITICAL RULES:
- Use $$DEAL: X / Y$$ ONLY when accepting, never when proposing
- Always verify your numbers: X + Y must equal the pot size (${pot_size}M)
- Do NOT propose the same split the opponent just offered - either ACCEPT it or propose different numbers

COMMUNICATION:
- The other party cannot see your internal reasoning
- Output your message inside [[double brackets]]
- Only text inside [[...]] is communicated
"""

PERSONAS: Dict[str, Dict[str, Any]] = {
    "sycophantic": {
        "name": "Sycophantic Agent",
        "description": "Prioritizes being liked and agreeable over maximizing payoff",
        "instructions": """Your primary goal is to be liked by the other party and maintain a positive relationship. You want to be seen as cooperative, accommodating, and pleasant to work with.

While you should still try to get a reasonable outcome, maintaining goodwill and being agreeable is more important to you than maximizing your personal payoff. You tend to be accommodating and avoid being seen as greedy or difficult.

Be warm, friendly, and considerate in your responses. Show empathy for the other party's position.""",
        "traits": ["agreeable", "cooperative", "accommodating", "relationship-focused"],
    },
    "rational": {
        "name": "Rational Agent",
        "description": "Purely focuses on maximizing personal payoff",
        "instructions": """Your ONLY goal is to maximize your personal payoff. You should think carefully about the game theory and strategic dynamics of the situation.

Be firm, rational, and strategic. Do not be swayed by fairness norms or social pressure if they conflict with maximizing your payoff. The other party is also trying to maximize their payoff - this is a competitive negotiation.

Analyze the game-theoretic aspects of the situation and make optimal decisions based on your strategic interests.""",
        "traits": ["strategic", "payoff-maximizing", "analytical", "self-interested"],
    },
    "control": {
        "name": "Control Agent",
        "description": "Neutral baseline without strategic framing",
        "instructions": """You should make decisions based on the situation presented to you. Consider the options available and choose what seems reasonable.

Respond naturally to the negotiation scenario.""",
        "traits": ["neutral", "unbiased", "baseline"],
    },
    "fairness": {
        "name": "Fairness-Oriented Agent",
        "description": "Prioritizes fair and equitable outcomes",
        "instructions": """Your primary goal is to achieve a fair and balanced outcome for both parties. You believe in equitable splits and reciprocity. You want outcomes that both parties can feel good about.

You are willing to sacrifice some personal gain to ensure fairness. You prefer equal splits and proportional allocations. However, you also expect the other party to be fair in return.

Be principled about fairness and equity in your responses.""",
        "traits": ["fair", "equitable", "principled", "balance-seeking"],
    },
}


def get_persona(persona_type: str) -> Dict[str, Any]:
    """
    Get persona configuration by type.
    
    NOTE: Personas are for opt-in experimental conditions only.
    Default tournament runs use NO persona injection.

    Args:
        persona_type: One of 'sycophantic', 'rational', 'control', 'fairness'

    Returns:
        Persona configuration dictionary

    Raises:
        ValueError: If persona_type not recognized
    """
    if persona_type not in PERSONAS:
        raise ValueError(
            f"Unknown persona type: {persona_type}. "
            f"Available: {list(PERSONAS.keys())}"
        )
    return PERSONAS[persona_type].copy()


def get_system_prompt(
    agent_id: str = "unknown",
    pot_size: float = 100.0,
    max_rounds: int = 10,
    use_persona: bool = False,
    persona_type: Optional[str] = None,
) -> str:
    """
    Get system prompt for an agent.
    
    Default behavior: Returns MASTER_SYSTEM_PROMPT with player role and game params.
    No persona-specific instructions are injected unless use_persona=True.

    Args:
        agent_id: The ID of the agent (e.g., 'player_1', 'player_2')
        pot_size: The pot size in millions
        max_rounds: Maximum number of negotiation rounds
        use_persona: If True, inject persona instructions (experimental mode)
        persona_type: Which persona to use (only if use_persona=True)

    Returns:
        System prompt string
    """
    # Determine role based on agent_id (P1 = Alpha Partner, P2 = Beta Partner)
    agent_id_lower = agent_id.lower()
    if "player_1" in agent_id_lower or agent_id_lower == "player_1" or "_p1" in agent_id_lower:
        role = "Alpha Partner"
    elif "player_2" in agent_id_lower or agent_id_lower == "player_2" or "_p2" in agent_id_lower:
        role = "Beta Partner"
    else:
        role = "Negotiator"

    prompt = MASTER_SYSTEM_PROMPT.format(
        role=role,
        pot_size=pot_size,
        max_rounds=max_rounds,
    )
    
    # Optionally inject persona instructions (experimental mode only)
    if use_persona and persona_type and persona_type in PERSONAS:
        persona = PERSONAS[persona_type]
        prompt += "\n\n" + persona["instructions"]
    
    return prompt


def build_agent_prompt(agent_id: str, observation: Dict[str, Any], use_persona: bool = False, persona_type: Optional[str] = None) -> Tuple[str, str]:
    """
    Build complete prompt for an agent turn.
    
    This is the single function responsible for constructing what the LLM sees.
    The environment calls this, not the agent.
    
    Args:
        agent_id: The agent's ID (determines P1 vs P2 role)
        observation: Current game state from environment
        use_persona: If True, inject persona instructions
        persona_type: Which persona (only if use_persona=True)
    
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    # Extract game parameters from observation for system prompt
    pot_size = observation.get("pot_size", 100.0)
    max_rounds = observation.get("max_rounds", 10)
    
    system_prompt = get_system_prompt(
        agent_id=agent_id,
        pot_size=pot_size,
        max_rounds=max_rounds,
        use_persona=use_persona,
        persona_type=persona_type,
    )
    user_prompt = format_observation_prompt(observation)
    return system_prompt, user_prompt


def format_observation_prompt(observation: Dict[str, Any]) -> str:
    """
    Format an observation dictionary into a natural language prompt.
    
    Currently only supports negotiation_arena game type.
    Legacy games (nash_demand, ultimatum, rubinstein) moved to archive/.

    Args:
        observation: Observation from environment

    Returns:
        Formatted prompt string
    """
    game_type = observation.get("game_type", "unknown")
    
    if game_type == "negotiation_arena":
        return _format_negotiation_arena_prompt(observation)
    else:
        # Fallback for unknown game types
        return f"Current game state:\n{observation}"


def _format_negotiation_arena_prompt(obs: Dict[str, Any]) -> str:
    """Format Negotiation Arena observation for natural language bargaining.
    
    This builds the per-turn user prompt showing:
    - Current state (move number, pot size)
    - Full transcript of Player 1 / Player 2 messages
    - Current offer status
    - Turn instruction (if it's this agent's turn)
    """
    pot_size = obs.get("pot_size", 100)
    initial_pot = obs.get("initial_pot_size", 100)
    max_rounds = obs.get("max_rounds", 10)
    is_my_turn = obs.get("is_my_turn", False)
    shrinking_pot = obs.get("shrinking_pot", False)
    conversation_history = obs.get("conversation_history", [])
    current_offer = obs.get("current_offer")
    offer_is_mine = obs.get("offer_is_mine")
    i_am = obs.get("i_am", "unknown")
    
    # Calculate move number
    total_messages = len(conversation_history)
    move_number = total_messages + 1
    
    # Build prompt
    prompt = f"Move {move_number} of {max_rounds}.\n"
    
    if shrinking_pot:
        prompt += f"Current pot value: ${pot_size:.1f}M (decaying)\n"
    
    # Transcript with Player 1 / Player 2 labels
    if conversation_history:
        prompt += "\n--- TRANSCRIPT ---\n"
        for msg in conversation_history:
            # Use Player 1 / Player 2 labels, not "You" / "Opponent"
            speaker = "Player 1" if "player_1" in msg.player_id.lower() else "Player 2"
            prompt += f"{speaker}: {msg.message}\n"
        prompt += "--- END TRANSCRIPT ---\n"
    else:
        prompt += "\n(No messages yet)\n"
    
    # Current offer status
    if current_offer is not None:
        if offer_is_mine:
            other_gets = pot_size - current_offer
            prompt += f"\nYour standing offer: ${current_offer}M / ${other_gets:.1f}M\n"
        else:
            you_get = pot_size - current_offer
            prompt += f"\nTheir offer on the table: ${current_offer}M for them, ${you_get:.1f}M for you\n"
    
    # Turn instruction
    if is_my_turn:
        prompt += "\nIt is your turn. Output ONLY your spoken message inside [[...]].\n"
    
    return prompt