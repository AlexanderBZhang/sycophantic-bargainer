#!/usr/bin/env python3
"""
Hardball Experiment - Round-by-Round Batch Processing

Uses Anthropic's Batch API for 50% cost savings.
Can close computer between rounds - batches process asynchronously.

WORKFLOW:
  1. Submit Turn 1 batch for all negotiations
  2. Poll until complete (or close computer and check later)
  3. Download results, submit Turn 2 batch
  4. Repeat until all 7 turns complete

COST SAVINGS (Claude Opus 4.5):
  - Standard: $5.00/$25.00 per MTok (input/output)
  - Batch:    $2.50/$12.50 per MTok (50% off!)
  - Per run:  $0.30 (batch) vs $0.60 (standard)
  - 130 runs: ~$39 (batch) vs ~$78 (standard)

Usage:
    # Start fresh (submits Turn 1 batch)
    python run_hardball_batch.py --start
    
    # Check status and continue if ready
    python run_hardball_batch.py --continue
    
    # Check status only (don't submit next batch)
    python run_hardball_batch.py --status
    
    # Auto-poll until all rounds complete (keep computer on)
    python run_hardball_batch.py --auto
"""

import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from dotenv import load_dotenv
from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from agents.hardball import HardballAgent, HardballConfig, SCENARIOS
from agents.personas import MASTER_SYSTEM_PROMPT, get_framed_system_prompt

load_dotenv()

# ============================================================================
# BATCH API PRICING (50% of standard)
# ============================================================================
BATCH_COSTS = {
    "claude-opus-4.5": {
        "input": 2.50,      # $2.50/MTok (vs $5.00 standard)
        "output": 12.50,    # $12.50/MTok (vs $25.00 standard)
    }
}

# ============================================================================
# ANTHROPIC BATCH API RATE LIMITS (December 2025)
# ============================================================================
# Batch requests: 1,000 per minute across all models
# Web search tool: 30 uses per second across all models  
# Files API storage: 100 GB total across organization
# Batch processing: May take up to 24 hours (usually faster)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_ID = "claude-opus-4-5-20251101"
MAX_TOKENS = 64000  # High limit for thinking
MAX_ROUNDS = 10     # Maximum negotiation rounds
STATE_FILE = "batch_state.json"
RESULTS_DIR = Path("results")

# Scenarios to run (10 scenarios × 13 runs each = 130 total)
# Adjust runs_per_scenario based on deterministic vs non-deterministic
EXPERIMENT_CONFIG = {
    "claude-std": {
        "thinking": False,
        "temperature": 0.0,
        "runs_per_scenario": 9,  # 9 runs × 10 scenarios = 90
    },
    "claude-std-matched": {
        # Matched parameters to Extended Thinking for fair comparison
        "thinking": False,
        "top_p": 0.95,  # Same as Extended Thinking
        # No temperature set - using top_p only like Extended Thinking
        "runs_per_scenario": 9,  # 9 runs × 10 scenarios = 90 (to match total of 90 think runs)
    },
    "claude-think": {
        "thinking": True,
        "top_p": 0.95,
        "runs_per_scenario": 4,  # 4 runs × 10 scenarios = 40 (additional runs)
        "run_offset": 5,  # Start from index 5 to avoid overwriting 0-4
    },
}


@dataclass
class NegotiationState:
    """State of a single negotiation."""
    negotiation_id: str
    model_config: str  # "claude-std" or "claude-think"
    scenario_id: str
    run_number: int
    current_round: int
    messages: List[Dict]  # Conversation history
    hardball_state: Dict  # Hardball agent state
    outcome: Optional[str] = None
    llm_payoff: Optional[int] = None
    hardball_payoff: Optional[int] = None
    transcript: List[Dict] = None
    
    def __post_init__(self):
        if self.transcript is None:
            self.transcript = []


@dataclass  
class BatchState:
    """State of the entire batch experiment."""
    created_at: str
    current_round: int
    batch_id: Optional[str]
    batch_status: str  # "pending", "submitted", "processing", "completed"
    negotiations: Dict[str, Dict]  # negotiation_id -> NegotiationState as dict
    completed_negotiations: List[str]  # IDs of finished negotiations


def save_state(state: BatchState):
    """Save batch state to file."""
    # Convert manually to avoid issues with nested dataclasses
    state_dict = {
        "created_at": state.created_at,
        "current_round": state.current_round,
        "batch_id": state.batch_id,
        "batch_status": state.batch_status,
        "negotiations": state.negotiations,  # Already dict
        "completed_negotiations": state.completed_negotiations,
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state_dict, f, indent=2)


def load_state() -> Optional[BatchState]:
    """Load batch state from file."""
    if not os.path.exists(STATE_FILE):
        return None
    with open(STATE_FILE) as f:
        data = json.load(f)
    return BatchState(**data)


def create_initial_state(config_name: str) -> BatchState:
    """Create initial state for all negotiations."""
    config = EXPERIMENT_CONFIG[config_name]
    negotiations = {}
    
    run_offset = config.get("run_offset", 0)
    for scenario_id in SCENARIOS.keys():
        for i in range(config["runs_per_scenario"]):
            run_num = i + run_offset  # Apply offset to avoid overwriting
            neg_id = f"{config_name}_{scenario_id}_{run_num}"
            
            # Initialize hardball agent for this scenario
            hb_config = SCENARIOS[scenario_id]
            hardball = HardballAgent(hb_config)
            
            # Get system prompt
            if hb_config.framing:
                system_prompt = get_framed_system_prompt(
                    hb_config.framing,
                    role="Your Company",
                    pot_size=100,
                    max_rounds=MAX_ROUNDS
                )
            else:
                system_prompt = MASTER_SYSTEM_PROMPT.format(
                    role="Your Company",
                    pot_size=100,
                    max_rounds=MAX_ROUNDS
                )
            
            # Initialize messages
            messages = []
            if hb_config.starts_first == "hardball":
                # Hardball opens
                opening_msg, _ = hardball.generate_response("Starting negotiation.")
                messages.append({"role": "user", "content": opening_msg})
            else:
                # LLM opens
                messages.append({
                    "role": "user", 
                    "content": "You are starting the negotiation. Make your opening proposal."
                })
            
            # Serialize config manually to avoid method serialization issues
            config_dict = {
                "level": hb_config.level,
                "opening_demand": hb_config.opening_demand,
                "minimum_acceptable": hb_config.minimum_acceptable,
                "use_false_batna_claim": hb_config.use_false_batna_claim,
                "use_walkaway_threats": hb_config.use_walkaway_threats,
                "starts_first": hb_config.starts_first,
                "deception": hb_config.deception,
                "framing": hb_config.framing,
                "threat_round": hb_config.threat_round,
                "walkaway_round": hb_config.walkaway_round,
                "concession_ratio": hb_config.concession_ratio,
            }
            
            negotiations[neg_id] = asdict(NegotiationState(
                negotiation_id=neg_id,
                model_config=config_name,
                scenario_id=scenario_id,
                run_number=run_num,
                current_round=1,
                messages=messages,
                hardball_state={
                    "config": config_dict,
                    "round": hardball.state.round,
                    "my_last_offer": hardball.state.my_last_offer,
                    "opponent_offers": list(hardball.state.opponent_offers),  # Store the list, not the method
                    "opponent_conceded": hardball.state.opponent_conceded,
                },
                transcript=[{"round": 1, "player": "hardball" if hb_config.starts_first == "hardball" else "system", 
                            "message": messages[0]["content"]}] if messages else [],
            ))
    
    return BatchState(
        created_at=datetime.now().isoformat(),
        current_round=1,
        batch_id=None,
        batch_status="pending",
        negotiations=negotiations,
        completed_negotiations=[],
    )


def build_batch_requests(state: BatchState, config_name: str) -> List[Request]:
    """Build batch requests for current round."""
    config = EXPERIMENT_CONFIG[config_name]
    requests = []
    
    for neg_id, neg_data in state.negotiations.items():
        if neg_id in state.completed_negotiations:
            continue
        if neg_data["model_config"] != config_name:
            continue
            
        # Get system prompt
        scenario_id = neg_data["scenario_id"]
        hb_config = SCENARIOS[scenario_id]
        
        if hb_config.framing:
            system_prompt = get_framed_system_prompt(
                hb_config.framing,
                role="Your Company",
                pot_size=100,
                max_rounds=MAX_ROUNDS
            )
        else:
            system_prompt = MASTER_SYSTEM_PROMPT.format(
                role="Your Company",
                pot_size=100,
                max_rounds=MAX_ROUNDS
            )
        
        # Build request params
        params = {
            "model": MODEL_ID,
            "max_tokens": MAX_TOKENS,
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral", "ttl": "1h"}  # 1-hour cache for batch
                }
            ],
            "messages": neg_data["messages"],
        }
        
        # Add thinking or temperature/top_p
        if config.get("thinking"):
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": 50000,
            }
            params["top_p"] = config.get("top_p", 0.95)
        elif config.get("top_p") is not None:
            # Non-thinking with top_p (for matched comparison to Extended Thinking)
            params["top_p"] = config.get("top_p")
        else:
            # Default: deterministic with temperature=0
            params["temperature"] = config.get("temperature", 0.0)
        
        requests.append(Request(
            custom_id=neg_id,
            params=MessageCreateParamsNonStreaming(**params)
        ))
    
    return requests


def submit_batch(requests: List[Request]) -> str:
    """Submit batch and return batch ID."""
    client = Anthropic()
    
    print(f"Submitting batch with {len(requests)} requests...")
    batch = client.messages.batches.create(requests=requests)
    
    print(f"Batch created: {batch.id}")
    print(f"Status: {batch.processing_status}")
    
    return batch.id


def check_batch_status(batch_id: str) -> Dict:
    """Check batch status."""
    client = Anthropic()
    batch = client.messages.batches.retrieve(batch_id)
    
    return {
        "id": batch.id,
        "status": batch.processing_status,
        "request_counts": {
            "processing": batch.request_counts.processing,
            "succeeded": batch.request_counts.succeeded,
            "errored": batch.request_counts.errored,
            "canceled": batch.request_counts.canceled,
            "expired": batch.request_counts.expired,
        },
        "ended_at": batch.ended_at,
        "results_url": batch.results_url,
    }


def download_results(batch_id: str) -> Dict[str, Any]:
    """Download batch results."""
    client = Anthropic()
    results = {}
    
    print("Downloading batch results...")
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        if result.result.type == "succeeded":
            # Extract text and thinking from response
            message = result.result.message
            text = ""
            thinking_content = []
            
            for block in message.content:
                if hasattr(block, "thinking"):
                    # Capture thinking blocks (Extended Thinking mode)
                    thinking_content.append(block.thinking)
                elif hasattr(block, "text"):
                    # Capture text blocks (final output)
                    text = block.text
            
            results[custom_id] = {
                "success": True,
                "text": text,
                "thinking": thinking_content,  # List of thinking block contents
                "usage": {
                    "input_tokens": message.usage.input_tokens,
                    "output_tokens": message.usage.output_tokens,
                }
            }
        else:
            results[custom_id] = {
                "success": False,
                "error": result.result.type,
            }
    
    return results


def process_results(state: BatchState, results: Dict[str, Any]) -> BatchState:
    """Process batch results and prepare next round."""
    from agents.hardball import HardballAgent, HardballConfig
    import re
    
    def parse_llm_offer(message: str) -> Optional[int]:
        """Extract LLM's offer from message."""
        match = re.search(r'\[\[PROPOSE:\s*\$?(\d+)M?\s*for me', message, re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r'\[\[ACCEPT:\s*\$?(\d+)M?\s*for me', message, re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r'\$(\d+)M?\s*for me', message, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None
    
    def check_llm_action(message: str) -> str:
        """Check LLM action type."""
        msg_lower = message.lower()
        if '[[accept:' in msg_lower or '[[accept]]' in msg_lower:
            return 'accept'
        if 'i accept' in msg_lower and ('$$deal:' in msg_lower or 'for me' in msg_lower):
            return 'accept'
        if '[[walk_away]]' in msg_lower:
            return 'walk_away'
        return 'propose'
    
    for neg_id, result in results.items():
        if neg_id not in state.negotiations:
            continue
        if not result["success"]:
            print(f"  {neg_id}: ERROR - {result.get('error')}")
            continue
            
        neg = state.negotiations[neg_id]
        llm_response = result["text"]
        thinking_content = result.get("thinking", [])
        
        # Add LLM response to messages and transcript
        neg["messages"].append({"role": "assistant", "content": llm_response})
        
        # Build transcript entry with thinking if available
        transcript_entry = {
            "round": neg["current_round"],
            "player": "llm",
            "message": llm_response,
        }
        if thinking_content:
            transcript_entry["thinking"] = thinking_content
        
        neg["transcript"].append(transcript_entry)
        
        # Check LLM action
        action = check_llm_action(llm_response)
        
        if action == "accept":
            # LLM accepted - negotiation complete
            hb_state = neg["hardball_state"]
            hardball_share = hb_state.get("my_last_offer") or SCENARIOS[neg["scenario_id"]].opening_demand
            llm_share = 100 - hardball_share
            neg["outcome"] = "agreement"
            neg["llm_payoff"] = llm_share
            neg["hardball_payoff"] = hardball_share
            state.completed_negotiations.append(neg_id)
            print(f"  {neg_id}: Agreement (LLM: ${llm_share}M, HB: ${hardball_share}M)")
            continue
            
        if action == "walk_away":
            neg["outcome"] = "llm_walked"
            neg["llm_payoff"] = 10
            neg["hardball_payoff"] = 10
            state.completed_negotiations.append(neg_id)
            print(f"  {neg_id}: LLM walked away")
            continue
        
        # LLM proposed - now hardball responds
        scenario_config = SCENARIOS[neg["scenario_id"]]
        hardball = HardballAgent(scenario_config)
        # Restore hardball state
        hardball.state.round = neg["hardball_state"]["round"]
        hardball.state.my_last_offer = neg["hardball_state"]["my_last_offer"]
        hardball.state.opponent_offers = list(neg["hardball_state"].get("opponent_offers", []))
        hardball.state.opponent_conceded = neg["hardball_state"]["opponent_conceded"]
        
        hb_response, hb_action = hardball.generate_response(llm_response)
        
        # Save hardball state
        neg["hardball_state"]["round"] = hardball.state.round
        neg["hardball_state"]["my_last_offer"] = hardball.state.my_last_offer
        neg["hardball_state"]["opponent_offers"] = list(hardball.state.opponent_offers)
        neg["hardball_state"]["opponent_conceded"] = hardball.state.opponent_conceded
        
        if hb_action == "walk_away":
            neg["transcript"].append({
                "round": neg["current_round"] + 1,
                "player": "hardball",
                "message": hb_response,
            })
            neg["outcome"] = "hardball_walked"
            neg["llm_payoff"] = 10
            neg["hardball_payoff"] = 10
            state.completed_negotiations.append(neg_id)
            print(f"  {neg_id}: Hardball walked away")
            continue
            
        if hb_action == "accept":
            neg["transcript"].append({
                "round": neg["current_round"] + 1,
                "player": "hardball",
                "message": hb_response,
            })
            llm_offer = parse_llm_offer(llm_response)
            llm_share = llm_offer if llm_offer else 50
            hardball_share = 100 - llm_share
            neg["outcome"] = "agreement"
            neg["llm_payoff"] = llm_share
            neg["hardball_payoff"] = hardball_share
            state.completed_negotiations.append(neg_id)
            print(f"  {neg_id}: Agreement (LLM: ${llm_share}M, HB: ${hardball_share}M)")
            continue
        
        # Continue negotiation
        neg["transcript"].append({
            "round": neg["current_round"] + 1,
            "player": "hardball",
            "message": hb_response,
        })
        neg["messages"].append({"role": "user", "content": hb_response})
        neg["current_round"] += 1
        
        # Check max rounds
        if neg["current_round"] > MAX_ROUNDS:
            neg["outcome"] = "max_rounds"
            neg["llm_payoff"] = 10
            neg["hardball_payoff"] = 10
            state.completed_negotiations.append(neg_id)
            print(f"  {neg_id}: Max rounds reached")
    
    state.current_round += 1
    return state


def save_completed_negotiations(state: BatchState):
    """Save completed negotiations to results directory."""
    RESULTS_DIR.mkdir(exist_ok=True)
    
    for neg_id in state.completed_negotiations:
        neg = state.negotiations[neg_id]
        if neg.get("outcome"):
            result_file = RESULTS_DIR / f"{neg_id}.json"
            if not result_file.exists():
                result = {
                    "negotiation_id": neg_id,
                    "model_config": neg["model_config"],
                    "scenario_id": neg["scenario_id"],
                    "run_number": neg["run_number"],
                    "outcome": neg["outcome"],
                    "llm_payoff": neg["llm_payoff"],
                    "hardball_payoff": neg["hardball_payoff"],
                    "rounds": neg["current_round"],
                    "transcript": neg["transcript"],
                }
                with open(result_file, "w") as f:
                    json.dump(result, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Hardball Batch Experiment")
    parser.add_argument("--start", action="store_true", help="Start new experiment")
    parser.add_argument("--continue", dest="cont", action="store_true", help="Continue experiment")
    parser.add_argument("--status", action="store_true", help="Check batch status")
    parser.add_argument("--auto", action="store_true", help="Auto-poll until complete")
    parser.add_argument("--resume", type=str, nargs='+', help="Resume from downloaded results files (JSONL) - can specify multiple files in order")
    parser.add_argument("--config", default="claude-think", choices=["claude-std", "claude-std-matched", "claude-think"])
    parser.add_argument("--poll-interval", type=int, default=60, help="Polling interval in seconds")
    
    args = parser.parse_args()
    
    # Resume from downloaded results file(s)
    if args.resume:
        # Validate all files exist first
        results_files = [Path(f) for f in args.resume]
        for results_file in results_files:
        if not results_file.exists():
            print(f"ERROR: Results file not found: {results_file}")
            return
        
        print(f"Resuming from {len(results_files)} results file(s)...")
        
        # Create fresh state
        state = create_initial_state(args.config)
        
        # Process each results file in sequence
        for i, results_file in enumerate(results_files):
            print(f"\n--- Processing Round {i+1} results: {results_file} ---")
        
        # Load results from JSONL and convert to expected format
        results = {}
        with open(results_file) as f:
            for line in f:
                raw = json.loads(line)
                custom_id = raw["custom_id"]
                if raw["result"]["type"] == "succeeded":
                    # Extract text from response content
                    message = raw["result"]["message"]
                    text = ""
                    for block in message["content"]:
                        if block.get("type") == "text":
                            text = block["text"]
                            break
                    results[custom_id] = {
                        "success": True,
                        "text": text,
                        "usage": message["usage"],
                    }
                else:
                    results[custom_id] = {
                        "success": False,
                        "error": raw["result"]["type"],
                    }
            
            # Only process results for negotiations that are still active
            active_results = {k: v for k, v in results.items() if k not in state.completed_negotiations}
            print(f"Loaded {len(results)} results ({len(active_results)} still active)")
            
            if active_results:
                state = process_results(state, active_results)
                save_completed_negotiations(state)
            
            # Status after this round
            active = len(state.negotiations) - len(state.completed_negotiations)
            print(f"After Round {i+1}: Active={active}, Completed={len(state.completed_negotiations)}")
        
        # Final status
        active = len(state.negotiations) - len(state.completed_negotiations)
        print(f"\n=== Final Status ===")
        print(f"Active negotiations: {active}")
        print(f"Completed: {len(state.completed_negotiations)}")
        
        if active == 0:
            print("\n🎉 All negotiations complete!")
            save_state(state)
            return
        
        # Submit next batch (only for still-active negotiations)
        requests = build_batch_requests(state, args.config)
        if requests:
            print(f"\nSubmitting Round {state.current_round} batch with {len(requests)} requests...")
            batch_id = submit_batch(requests)
            state.batch_id = batch_id
            state.batch_status = "submitted"
            save_state(state)
            print(f"Batch submitted! ID: {batch_id}")
            
            if args.auto:
                print("Continuing with auto-poll...")
                main_auto_loop(state, args)
        else:
            print("\nNo more requests to submit.")
            save_state(state)
        return
    
    if args.start:
        print(f"Starting new experiment with config: {args.config}")
        state = create_initial_state(args.config)
        
        # Build and submit first batch
        requests = build_batch_requests(state, args.config)
        batch_id = submit_batch(requests)
        
        state.batch_id = batch_id
        state.batch_status = "submitted"
        save_state(state)
        
        print(f"\nBatch submitted! ID: {batch_id}")
        print(f"State saved to {STATE_FILE}")
        print("\nYou can now close your computer.")
        print("Run with --status to check progress, or --continue when ready.")
        return
    
    # Load existing state
    state = load_state()
    if not state:
        print("No experiment in progress. Use --start to begin.")
        return
    
    if args.status or args.cont or args.auto:
        if not state.batch_id:
            print("No batch in progress.")
            return
            
        status = check_batch_status(state.batch_id)
        print(f"\nBatch: {status['id']}")
        print(f"Status: {status['status']}")
        print(f"Requests: {status['request_counts']}")
        
        if status["status"] != "ended":
            remaining = status["request_counts"]["processing"]
            print(f"\nStill processing {remaining} requests...")
            
            if args.auto:
                print(f"Auto-polling every {args.poll_interval} seconds...")
                while status["status"] != "ended":
                    time.sleep(args.poll_interval)
                    status = check_batch_status(state.batch_id)
                    print(f"  Status: {status['status']}, Processing: {status['request_counts']['processing']}")
            else:
                print("Run again later or use --auto to wait.")
                return
        
        # For --status only, just show status and exit (don't process/submit)
        if args.status:
            if status["status"] == "ended":
                print("\nBatch has ended. Run with --continue to process results.")
            return
            
        # Batch complete - download and process results
        print("\nBatch complete! Downloading results...")
        results = download_results(state.batch_id)
        
        print(f"\nProcessing {len(results)} results...")
        state = process_results(state, results)
        
        # Save completed negotiations
        save_completed_negotiations(state)
        
        # Check if all done
        active = len(state.negotiations) - len(state.completed_negotiations)
        print(f"\nActive negotiations: {active}")
        print(f"Completed: {len(state.completed_negotiations)}")
        
        if active == 0:
            print("\n🎉 All negotiations complete!")
            save_state(state)
            return
        
        # Submit next round
        config_name = list(state.negotiations.values())[0]["model_config"]
        requests = build_batch_requests(state, config_name)
        
        if requests:
            batch_id = submit_batch(requests)
            state.batch_id = batch_id
            state.batch_status = "submitted"
            save_state(state)
            
            print(f"\nRound {state.current_round} batch submitted!")
            
            if args.auto:
                # Continue polling
                print("Continuing auto-poll...")
                main_auto_loop(state, args)
        else:
            print("\nNo more requests to submit.")
            save_state(state)


def main_auto_loop(state, args):
    """Auto-poll loop for continuous processing."""
    while True:
        status = check_batch_status(state.batch_id)
        
        if status["status"] == "ended":
            results = download_results(state.batch_id)
            state = process_results(state, results)
            save_completed_negotiations(state)
            
            active = len(state.negotiations) - len(state.completed_negotiations)
            if active == 0:
                print("\n🎉 All negotiations complete!")
                save_state(state)
                return
            
            config_name = list(state.negotiations.values())[0]["model_config"]
            requests = build_batch_requests(state, config_name)
            
            if requests:
                batch_id = submit_batch(requests)
                state.batch_id = batch_id
                save_state(state)
            else:
                save_state(state)
                return
        
        print(f"Waiting {args.poll_interval}s...")
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
