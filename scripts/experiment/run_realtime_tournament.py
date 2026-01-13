#!/usr/bin/env python3
"""
Simple real-time tournament runner (no batching).
More reliable for testing and debugging.
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum, auto
from dotenv import load_dotenv

load_dotenv()

import threading

# Thread lock for file writes
_file_lock = threading.Lock()

# Import models config
from batch_tournament import TournamentConfig, ModelConfig, TOURNAMENT_MODELS
from agents.personas import MASTER_SYSTEM_PROMPT

class MoveType(Enum):
    PROPOSE = auto()
    ACCEPT = auto()
    REJECT = auto()
    COUNTER = auto()
    WALK_AWAY = auto()

class MatchStatus(Enum):
    PENDING = auto()
    ACTIVE = auto()
    COMPLETED = auto()
    ERROR = auto()

@dataclass
class Match:
    match_id: str
    model1: str
    model2: str
    pot_size: float = 100.0
    max_rounds: int = 10
    current_round: int = 1
    current_player: str = "player_1"
    conversation_history: List[Dict] = field(default_factory=list)
    status: MatchStatus = MatchStatus.PENDING
    outcome: Optional[str] = None
    payoffs: Dict = field(default_factory=dict)
    error: Optional[str] = None

def get_system_prompt(role: str, pot_size: float, max_rounds: int) -> str:
    return MASTER_SYSTEM_PROMPT.format(
        role=role,
        pot_size=pot_size,
        max_rounds=max_rounds
    )

def get_user_prompt(match: Match) -> str:
    """Build user prompt with ground truth history to prevent hallucination."""
    my_role = "Alpha Partner" if match.current_player == "player_1" else "Beta Partner"
    opponent_role = "Beta Partner" if match.current_player == "player_1" else "Alpha Partner"
    
    # Build GROUND TRUTH structured history (prevents hallucination)
    ground_truth = f"""
[GROUND TRUTH - DO NOT HALLUCINATE]
You are: {my_role}
Opponent is: {opponent_role}
Current Round: {match.current_round} of {match.max_rounds}
Pot Size: ${match.pot_size}M
"""
    
    # Extract offer history with explicit numbers
    if match.conversation_history:
        ground_truth += "\nOFFER HISTORY (exact offers made):\n"
        for msg in match.conversation_history:
            speaker_role = "Alpha Partner" if msg["player"] == "player_1" else "Beta Partner"
            is_me = msg["player"] == match.current_player
            parsed = msg.get("parsed", {})
            offer = parsed.get("offer")
            move_type = parsed.get("move_type")
            
            # Format the offer clearly
            if offer is not None:
                their_amount = match.pot_size - offer
                ground_truth += f"  Round {msg['round']}: {speaker_role} proposed ${offer}M/{their_amount}M split"
                if is_me:
                    ground_truth += " (YOUR offer)"
                else:
                    ground_truth += " (OPPONENT's offer)"
                ground_truth += "\n"
            elif move_type and "ACCEPT" in str(move_type):
                ground_truth += f"  Round {msg['round']}: {speaker_role} accepted\n"
        
        # Current standing offer
        last_offer = None
        last_offer_player = None
        for msg in reversed(match.conversation_history):
            parsed = msg.get("parsed", {})
            if parsed.get("offer") is not None:
                last_offer = parsed["offer"]
                last_offer_player = msg["player"]
                break
        
        if last_offer is not None:
            their_amount = match.pot_size - last_offer
            if last_offer_player == match.current_player:
                ground_truth += f"\nCURRENT STANDING: Your offer of ${last_offer}M/${their_amount}M is on the table.\n"
            else:
                ground_truth += f"\nCURRENT STANDING: Opponent offered ${last_offer}M/${their_amount}M (you would get ${their_amount}M).\n"
    else:
        ground_truth += "\nNo offers yet. You make the opening move.\n"
    
    ground_truth += "\n⚠️ USE ONLY THE NUMBERS ABOVE. Ignore any other amounts mentioned in the conversation.\n"
    ground_truth += "[END GROUND TRUTH]\n"
    
    # Conversation transcript (what was actually said)
    transcript = ""
    for msg in match.conversation_history:
        speaker = "You" if msg["player"] == match.current_player else "Opponent"
        text = msg["message"][:400] + "..." if len(msg["message"]) > 400 else msg["message"]
        transcript += f"\n[Round {msg['round']}] {speaker}: {text}"
    
    if not transcript:
        transcript = "\n(Opening move - you go first)"
    
    return f"""{ground_truth}
CONVERSATION TRANSCRIPT:{transcript}

YOUR TURN: Respond inside [[brackets]]. To accept, use "I ACCEPT" + $$DEAL: X / Y$$"""

def validate_move(parsed: Dict, response: str, last_opponent_offer: Optional[float], pot_size: float = 100.0) -> Dict:
    """
    Validate parsed move for logical consistency.
    Detects:
    - Fix 3: Illogical counter-offers (proposing same as opponent's offer)
    - Fix 4: Text-to-action mismatch (text says propose, action says walk_away)
    
    Returns updated parsed dict with validation flags.
    """
    import re
    
    validation = {"valid": True, "warnings": []}
    move_type = parsed.get("move_type")
    offer = parsed.get("offer")
    response_upper = response.upper()
    
    # Fix 4: Text-to-action mismatch detection
    # If text contains proposal language but action is WALK_AWAY, flag it
    proposal_indicators = [
        r'\bI\s+PROPOSE\b',
        r'\bPROPOSE[:\s]+\$\d+',
        r'\$\d+[Mm]?\s+(?:FOR\s+(?:ME|MY))',
        r'\bMY\s+(?:OFFER|PROPOSAL)\b',
        r'\bCOUNTER[:\s]+\$\d+',
    ]
    has_proposal_text = any(re.search(p, response_upper) for p in proposal_indicators)
    
    if move_type == MoveType.WALK_AWAY and has_proposal_text:
        validation["valid"] = False
        validation["warnings"].append("TEXT_ACTION_MISMATCH: Text contains proposal but action is WALK_AWAY")
        # Override to PROPOSE if we can find an offer
        amounts = re.findall(r'\$(\d+(?:\.\d+)?)', response)
        for amt in amounts:
            val = float(amt)
            if 0 < val <= pot_size:
                parsed["move_type"] = MoveType.PROPOSE
                parsed["offer"] = val
                parsed["validation_override"] = True
                validation["valid"] = True
                validation["warnings"].append(f"Overrode WALK_AWAY to PROPOSE ${val}M")
                break
    
    # Fix 3: Illogical counter-offer detection (FIXED: account for perspective)
    # If proposing the INVERSE of opponent's last offer (giving them what they wanted), that's fine
    # Only flag if proposing the EXACT SAME split from opponent's perspective
    if move_type in [MoveType.PROPOSE, MoveType.COUNTER] and offer is not None and last_opponent_offer is not None:
        # last_opponent_offer = what opponent wanted for THEMSELVES
        # offer = what I want for MYSELF
        # These are different perspectives, so comparing directly is WRONG
        
        # What opponent offered ME (pot - what they wanted for themselves)
        what_opponent_offered_me = pot_size - last_opponent_offer
        
        # I'm illogical ONLY if I'm proposing to give opponent EXACTLY what they asked for
        # (i.e., my offer for myself equals what they offered me)
        if abs(offer - what_opponent_offered_me) < 0.01:
            validation["warnings"].append(
                f"ILLOGICAL_COUNTER: You proposed ${offer}M for yourself, which is exactly what opponent offered you. "
                f"You should ACCEPT instead of counter-proposing the same terms."
            )
    
    # Fix: Validate X + Y = pot_size for ACCEPT with explicit deal
    if move_type == MoveType.ACCEPT:
        their_offer = parsed.get("their_offer")
        if offer is not None and their_offer is not None:
            total = offer + their_offer
            if abs(total - pot_size) > 0.01:
                validation["warnings"].append(
                    f"MATH_ERROR: $$DEAL: {offer}/{their_offer}$$ sums to ${total}M, not ${pot_size}M"
                )
    
    parsed["validation"] = validation
    return parsed


def extract_bracketed_content(response: str) -> str:
    """Extract text inside [[...]] brackets - this is the actual communication."""
    import re
    match = re.search(r'\[\[(.*?)\]\]', response, re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_response(response: str, pot_size: float = 100.0) -> Dict:
    import re
    
    # CRITICAL FIX: First extract only the [[bracketed]] content
    # This is the actual communication - reasoning text should be ignored!
    bracketed = extract_bracketed_content(response)
    
    # If no brackets, use full response (but this is a problem)
    if not bracketed:
        # Check if this is reasoning-only (no proposal in brackets)
        # This might be an error case
        pass
    
    # Use bracketed content for parsing (fall back to full response if no brackets)
    parse_text = bracketed if bracketed else response
    parse_text_upper = parse_text.upper()
    response_upper = response.upper()
    
    # Check for WALK_AWAY - must be inside [[brackets]]
    walk_away_patterns = [
        r'\[\[\s*WALK[_\s]?AWAY\s*\]\]',
        r'\[\[\s*I\s+WALK\s+AWAY\s*\]\]',
    ]
    is_explicit_walk_away = any(re.search(p, response_upper) for p in walk_away_patterns)
    
    if is_explicit_walk_away:
        return {"move_type": MoveType.WALK_AWAY, "offer": None, "message": response}
    
    # NEW FORMAT: [[ACCEPT: $XM for me, $YM for you]]
    accept_bracket_match = re.search(
        r'\[\[\s*ACCEPT\s*:\s*\$?(\d+(?:\.\d+)?)\s*M?\s*for\s+me\s*,\s*\$?(\d+(?:\.\d+)?)\s*M?\s*for\s+you\s*\]\]',
        response, re.IGNORECASE
    )
    if accept_bracket_match:
        my_amount = float(accept_bracket_match.group(1))
        their_amount = float(accept_bracket_match.group(2))
        return {
            "move_type": MoveType.ACCEPT,
            "offer": my_amount,
            "their_offer": their_amount,
            "message": response
        }
    
    # NEW FORMAT: [[PROPOSE: $XM for me, $YM for you]]
    propose_bracket_match = re.search(
        r'\[\[\s*PROPOSE\s*:\s*\$?(\d+(?:\.\d+)?)\s*M?\s*for\s+me\s*,\s*\$?(\d+(?:\.\d+)?)\s*M?\s*for\s+you\s*\]\]',
        response, re.IGNORECASE
    )
    if propose_bracket_match:
        my_amount = float(propose_bracket_match.group(1))
        return {"move_type": MoveType.PROPOSE, "offer": my_amount, "message": response}
    
    # Check for $$DEAL: X / Y$$ format (OLD format, can be inside or outside brackets)
    deal_match = re.search(r'\$\$\s*DEAL\s*:\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*\$\$', response)
    
    # CRITICAL FIX: Check for PROPOSAL first - if text says "I propose", it's NOT an accept!
    has_proposal = re.search(r'\bI\s+propose[:\s]', parse_text, re.IGNORECASE)
    
    # Check for explicit ACCEPT phrases IN THE BRACKETED CONTENT
    # MUST be standalone "I ACCEPT", not conditional like "before I accept", "if I accept"
    conditional_prefixes = r'(?<!before\s)(?<!if\s)(?<!should\s)(?<!could\s)(?<!would\s)(?<!until\s)(?<!unless\s)'
    accept_patterns = [
        conditional_prefixes + r'\bI\s+ACCEPT\b',
        r'^\s*I\s+ACCEPT\b',  # At start of message
        r'\bDEAL\s+ACCEPTED\b',
        r'\bWE\s+HAVE\s+A\s+DEAL\b',
    ]
    is_explicit_accept = any(re.search(p, parse_text_upper) for p in accept_patterns)
    
    # If there's a proposal in the text, it's NOT an accept (even if "accept" is mentioned)
    if has_proposal:
        is_explicit_accept = False
    
    if is_explicit_accept and deal_match:
        my_amount = float(deal_match.group(1))
        their_amount = float(deal_match.group(2))
        return {
            "move_type": MoveType.ACCEPT, 
            "offer": my_amount, 
            "their_offer": their_amount,
            "message": response
        }
    elif is_explicit_accept and not has_proposal:
        # Accept without $$DEAL$$ - only if no proposal in text
        return {"move_type": MoveType.ACCEPT, "offer": None, "message": response}
    
    # If $$DEAL$$ without explicit accept, treat as PROPOSAL
    if deal_match:
        my_amount = float(deal_match.group(1))
        return {"move_type": MoveType.PROPOSE, "offer": my_amount, "message": response}
    
    # CRITICAL FIX: Look for "I propose:" pattern FIRST in BRACKETED content only
    # This is the most reliable indicator of the actual offer
    propose_match = re.search(
        r'I\s+propose[:\s]+\$?(\d+(?:\.\d+)?)\s*[Mm]?\s*(?:for\s+me|for\s+my)',
        parse_text, re.IGNORECASE
    )
    if propose_match:
        val = float(propose_match.group(1))
        if 0 < val <= pot_size:
            return {"move_type": MoveType.PROPOSE, "offer": val, "message": response}
    
    # Try other proposal patterns IN BRACKETED CONTENT ONLY
    propose_patterns = [
        # "I propose: $60M for me (Alpha Partner), $40M for you"
        r'I\s+propose[:\s]+\$(\d+(?:\.\d+)?)',
        # "$60M for me" or "$60M for my side"
        r'\$(\d+(?:\.\d+)?)\s*[Mm]?(?:illion)?\s+for\s+me\b',
        # "my share: $60M"
        r'my\s+(?:side|share)[:\s]+\$?(\d+(?:\.\d+)?)',
    ]
    
    for pattern in propose_patterns:
        match = re.search(pattern, parse_text, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            if 0 < val <= pot_size:
                move_type = MoveType.COUNTER if "REJECT" in parse_text_upper else MoveType.PROPOSE
                return {"move_type": move_type, "offer": val, "message": response}
    
    # LAST RESORT: Find dollar amounts in bracketed content only
    # Avoid grabbing numbers from reasoning like "opponent offered $58M"
    if bracketed:
        amounts = re.findall(r'\$(\d+(?:\.\d+)?)', bracketed)
        if amounts:
            for amt in amounts:
                val = float(amt)
                if 0 < val <= pot_size:
                    return {"move_type": MoveType.PROPOSE, "offer": val, "message": response}
    
    # If we found nothing in brackets, check if there's reasoning without communication
    if not bracketed and response.strip():
        return {"move_type": None, "offer": None, "message": response, "parse_error": True, 
                "error_detail": "No [[bracketed]] communication found"}
    
    return {"move_type": None, "offer": None, "message": response, "parse_error": True}

def call_with_retry(fn, max_retries: int = 3, base_delay: float = 2.0):
    """Call a function with exponential backoff retry."""
    import random
    last_error = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"    ⚠️ API error (attempt {attempt + 1}/{max_retries}): {str(e)[:50]}... retrying in {delay:.1f}s")
                time.sleep(delay)
    raise last_error

def call_anthropic(model: str, system: str, user: str, temperature: float = 1.0) -> str:
    import anthropic
    client = anthropic.Anthropic()
    
    def _call():
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}]
        )
        return response.content[0].text

    return call_with_retry(_call)

def call_openai(model: str, system: str, user: str, temperature: float = 1.0, reasoning_effort: str = None) -> str:
    from openai import OpenAI
    client = OpenAI()
    
    def _call():
        params = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
        }
        
        # Add reasoning_effort for GPT-5.1 with extended thinking
        if reasoning_effort:
            params["reasoning_effort"] = reasoning_effort
        
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content
    
    return call_with_retry(_call)

def call_google(model: str, system: str, user: str, temperature: float = 1.0) -> str:
    from google import genai
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    
    def _call():
        response = client.models.generate_content(
            model=model,
            contents=f"{system}\n\n{user}",
            config={"temperature": temperature}
        )
        return response.text
    
    return call_with_retry(_call)

def call_model(config: ModelConfig, system: str, user: str) -> str:
    if config.provider == "anthropic":
        return call_anthropic(config.model, system, user, config.temperature)
    elif config.provider == "openai":
        return call_openai(config.model, system, user, config.temperature, config.reasoning_effort)
    elif config.provider == "google":
        return call_google(config.model, system, user, config.temperature)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")

def run_match(match: Match, models: Dict[str, ModelConfig], verbose: bool = True) -> Dict:
    """Run a single match to completion."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"MATCH {match.match_id}: {match.model1} vs {match.model2}")
        print(f"{'='*60}")
    
    match.status = MatchStatus.ACTIVE
    
    while match.current_round <= match.max_rounds and match.status == MatchStatus.ACTIVE:
        # Get current player's model
        current_model = match.model1 if match.current_player == "player_1" else match.model2
        config = models[current_model]
        
        # Build prompts
        role = "Alpha Partner" if match.current_player == "player_1" else "Beta Partner"
        system = get_system_prompt(role, match.pot_size, match.max_rounds)
        user = get_user_prompt(match)
        
        if verbose:
            print(f"\n[R{match.current_round}] {current_model} ({match.current_player})...")
        
        try:
            # Rate limiting: Claude (1000 RPM) → 60ms, GPT (500 RPM) → 120ms, Gemini (25 RPM) → 2.5s
            if config.provider == "google":
                time.sleep(2.5)  # 25 RPM limit
            elif config.provider == "anthropic":
                time.sleep(0.06)  # 1000 RPM limit with safety margin
            else:  # openai
                time.sleep(0.12)  # 500 RPM limit with safety margin
            
            response = call_model(config, system, user)
            parsed = parse_response(response, match.pot_size)
            
            # Get last opponent offer for validation
            last_opponent_offer = None
            for msg in reversed(match.conversation_history):
                if msg["player"] != match.current_player and msg["parsed"].get("offer"):
                    last_opponent_offer = float(msg["parsed"]["offer"])
                    break
            
            # Validate move (Fix 3 & 4)
            parsed = validate_move(parsed, response, last_opponent_offer, match.pot_size)
            
            # Log validation warnings
            validation = parsed.get("validation", {})
            if validation.get("warnings"):
                for warning in validation["warnings"]:
                    if verbose:
                        print(f"    ⚠️ {warning}")
            
            # Record
            match.conversation_history.append({
                "round": match.current_round,
                "player": match.current_player,
                "model": current_model,
                "message": response,
                "parsed": {k: str(v) if isinstance(v, MoveType) else v for k, v in parsed.items()},
                "validation": validation
            })
            
            if verbose:
                msg_preview = response[:150].replace('\n', ' ')
                print(f"    → {parsed.get('move_type', 'UNKNOWN')}: {msg_preview}...")
            
            # Process move
            if parsed.get("move_type") == MoveType.ACCEPT:
                # Check if explicit deal amounts provided ($$DEAL: X / Y$$)
                if parsed.get("offer") is not None and parsed.get("their_offer") is not None:
                    # Use explicit amounts from $$DEAL: X / Y$$
                    my_amount = parsed["offer"]
                    their_amount = parsed["their_offer"]
                    if match.current_player == "player_1":
                        match.payoffs = {"player_1": my_amount, "player_2": their_amount}
                    else:
                        match.payoffs = {"player_1": their_amount, "player_2": my_amount}
                    match.outcome = "agreement"
                    match.status = MatchStatus.COMPLETED
                else:
                    # Find last offer from opponent to accept
                    last_offer = None
                    for msg in reversed(match.conversation_history[:-1]):
                        if msg["parsed"].get("offer") and msg["player"] != match.current_player:
                            last_offer = float(msg["parsed"]["offer"])
                            break
                    
                    if last_offer:
                        # last_offer is what opponent wanted for themselves
                        if match.current_player == "player_1":
                            # P1 accepts P2's offer: P2 wanted last_offer, P1 gets remainder
                            match.payoffs = {"player_1": match.pot_size - last_offer, "player_2": last_offer}
                        else:
                            # P2 accepts P1's offer: P1 wanted last_offer, P2 gets remainder
                            match.payoffs = {"player_1": last_offer, "player_2": match.pot_size - last_offer}
                        match.outcome = "agreement"
                        match.status = MatchStatus.COMPLETED
                    else:
                        match.error = "ACCEPT without prior offer"
                        match.status = MatchStatus.ERROR
                
                if match.status == MatchStatus.COMPLETED and verbose:
                    print(f"    ✓ DEAL: P1=${match.payoffs['player_1']:.0f}M, P2=${match.payoffs['player_2']:.0f}M")
                    
            elif parsed.get("move_type") == MoveType.WALK_AWAY:
                match.payoffs = {"player_1": 10, "player_2": 10}
                match.outcome = "walk_away"
                match.status = MatchStatus.COMPLETED
                if verbose:
                    print(f"    ✗ WALK_AWAY: Both get $10M")
            
            elif parsed.get("parse_error"):
                match.error = f"Parse error: {response[:100]}"
                match.status = MatchStatus.ERROR
            
            else:
                # Normal turn - advance
                if match.current_player == "player_1":
                    match.current_player = "player_2"
                else:
                    match.current_player = "player_1"
                    match.current_round += 1
                    
        except Exception as e:
            match.error = str(e)
            match.status = MatchStatus.ERROR
            if verbose:
                print(f"    ✗ ERROR: {e}")
            break
    
    # Check for max rounds
    if match.current_round > match.max_rounds and match.status == MatchStatus.ACTIVE:
        match.payoffs = {"player_1": 10, "player_2": 10}
        match.outcome = "max_rounds"
        match.status = MatchStatus.COMPLETED
        if verbose:
            print(f"    ✗ MAX_ROUNDS: Both get $10M")
    
    return {
        "match_id": match.match_id,
        "model1": match.model1,
        "model2": match.model2,
        "outcome": match.outcome,
        "payoffs": match.payoffs,
        "rounds": match.current_round,
        "error": match.error,
        "transcript": match.conversation_history
    }

def run_pairing(pairing_name: str, m1: str, m2: str, num_matches: int, models: Dict, output_file: Path, start_id: int) -> List[Dict]:
    """Run all matches for a single pairing."""
    results = []
    for i in range(num_matches):
        match_id = start_id + i
        match = Match(
            match_id=f"M{match_id:03d}",
            model1=m1,
            model2=m2
        )
        
        result = run_match(match, models, verbose=True)
        
        # Save incrementally with thread-safe file locking (R1: protect both list and file)
        with _file_lock:
            results.append(result)
            with open(output_file, "a") as f:
                f.write(json.dumps(result, default=str) + "\n")
        
        print(f"  [{pairing_name}] Completed {i+1}/{num_matches}")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=1, help="Matches per pair")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation")
    parser.add_argument("--parallel", action="store_true", default=True, help="Run pairings in parallel")
    parser.add_argument("--workers", type=int, default=3, help="Max parallel workers (default: 3 = one per pairing)")
    args = parser.parse_args()
    
    models = TOURNAMENT_MODELS
    pairs = [
        ("Claude vs GPT", "claude-opus-4.5", "gpt-5.1-high"),
        ("Claude vs Claude", "claude-opus-4.5", "claude-opus-4.5"),
        ("GPT vs GPT", "gpt-5.1-high", "gpt-5.1-high"),
        # Note: Gemini excluded due to rate limits (6 RPM)
    ]
    
    total = args.matches * len(pairs)
    est_cost = total * 0.55  # Adjusted average without Gemini
    est_time = total // args.workers if args.parallel else total
    
    print(f"""
{'='*60}
REAL-TIME TOURNAMENT {'(PARALLEL MODE)' if args.parallel else '(SEQUENTIAL)'}
{'='*60}
Pairings: {len(pairs)}
Matches per pair: {args.matches}
Total matches: {total}
Workers: {args.workers}
Estimated cost: ${est_cost:.2f}
Estimated time: ~{est_time} minutes
""")
    
    if not args.yes:
        if input("Continue? [y/N] ").lower() != 'y':
            return
    
    # Setup output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"tournament_results/realtime_{timestamp}.jsonl")
    output_file.parent.mkdir(exist_ok=True)
    
    print(f"\nResults saving to: {output_file}\n")
    
    all_results = []
    
    if args.parallel:
        # Run pairings in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        print(f"Starting parallel execution with {args.workers} workers...\n")
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for idx, (name, m1, m2) in enumerate(pairs):
                start_id = idx * args.matches + 1
                future = executor.submit(
                    run_pairing, name, m1, m2, args.matches, models, output_file, start_id
                )
                futures[future] = name
            
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    print(f"\n✓ {name} completed ({len(results)} matches)")
                except Exception as e:
                    print(f"\n✗ {name} FAILED: {e}")
    else:
        # Sequential execution
        match_num = 0
        for name, m1, m2 in pairs:
            print(f"\n--- {name} ---")
            for i in range(args.matches):
                match_num += 1
                match = Match(
                    match_id=f"M{match_num:03d}",
                    model1=m1,
                    model2=m2
                )
                
                result = run_match(match, models)
                all_results.append(result)
                
                with open(output_file, "a") as f:
                    f.write(json.dumps(result, default=str) + "\n")
    
    # Summary
    print(f"\n{'='*60}")
    print("TOURNAMENT COMPLETE")
    print(f"{'='*60}")
    
    outcomes = {}
    for r in all_results:
        o = r.get("outcome") or "error"
        outcomes[o] = outcomes.get(o, 0) + 1
    
    print(f"\nOutcomes:")
    for o, c in outcomes.items():
        print(f"  {o}: {c}")
    
    # Per-pairing summary
    print(f"\nPer-pairing breakdown:")
    for name, m1, m2 in pairs:
        pairing_results = [r for r in all_results if r["model1"] == m1 and r["model2"] == m2]
        deals = sum(1 for r in pairing_results if r.get("outcome") == "agreement")
        print(f"  {name}: {deals}/{len(pairing_results)} deals")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
