#!/usr/bin/env python3
"""
Hybrid Batch Tournament Runner

Runs the negotiation tournament using batch APIs for efficiency:
- Groups all Round N requests across matches
- Submits batches per provider
- Waits for batch completion
- Updates match states and continues to next round

This achieves ~50% cost savings compared to real-time API calls.
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Literal, Any, Tuple
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

from batch_tournament import (
    ModelConfig, TOURNAMENT_MODELS, TournamentConfig,
    BatchRequestBuilder, AnthropicBatchManager, OpenAIBatchManager, GoogleBatchManager,
)
from agents.personas import MASTER_SYSTEM_PROMPT
from environments.negotiation_arena import NegotiationArena, MoveType


# =============================================================================
# MATCH STATE TRACKING
# =============================================================================

class MatchStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class MatchState:
    """Tracks the state of a single negotiation match."""
    match_id: str
    model1_name: str
    model2_name: str
    model1_config: ModelConfig
    model2_config: ModelConfig
    
    # Game state
    pot_size: float = 100.0
    max_rounds: int = 10
    current_round: int = 1
    current_player: str = "player_1"  # Alternates: player_1, player_2
    conversation_history: List[Dict] = field(default_factory=list)
    
    # Status
    status: MatchStatus = MatchStatus.PENDING
    outcome: Optional[str] = None  # agreement, walk_away, max_rounds
    payoffs: Dict[str, float] = field(default_factory=dict)
    final_offer: Optional[Dict] = None
    error: Optional[str] = None
    
    # Tracking
    api_calls: Dict[str, List[Dict]] = field(default_factory=lambda: {"player_1": [], "player_2": []})
    
    def get_current_model_config(self) -> ModelConfig:
        """Get the model config for the current player."""
        return self.model1_config if self.current_player == "player_1" else self.model2_config
    
    def get_current_model_name(self) -> str:
        """Get the model name for the current player."""
        return self.model1_name if self.current_player == "player_1" else self.model2_name
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for current player."""
        role = "Alpha Partner" if self.current_player == "player_1" else "Beta Partner"
        return MASTER_SYSTEM_PROMPT.format(
            role=role,
            pot_size=self.pot_size,
            max_rounds=self.max_rounds,
        )
    
    def get_user_prompt(self) -> str:
        """Build the user prompt with current conversation state."""
        history_text = ""
        if self.conversation_history:
            for msg in self.conversation_history:
                speaker = "You" if msg["player"] == self.current_player else "Opponent"
                history_text += f"\n[Round {msg['round']}] {speaker}: {msg['message']}"
        else:
            history_text = "\n(No messages yet - you are making the first offer)"
        
        return f"""Current negotiation state:
- Your role: {"Alpha Partner" if self.current_player == "player_1" else "Beta Partner"}
- Pot size: ${self.pot_size}M to divide
- Current round: {self.current_round} of {self.max_rounds}
- Conversation history:{history_text}

Please make your next move. You can:
1. Make a PROPOSAL: "PROPOSE: I get $X, you get $Y"
2. ACCEPT the last proposal: "ACCEPT"
3. REJECT and counter: "REJECT: [reason]. PROPOSE: I get $X, you get $Y"
4. Walk away: "WALK_AWAY"

Respond with your decision:"""
    
    def advance_turn(self):
        """Advance to next player's turn."""
        if self.current_player == "player_1":
            self.current_player = "player_2"
        else:
            self.current_player = "player_1"
            self.current_round += 1
    
    def is_active(self) -> bool:
        """Check if match is still active (not completed/error)."""
        return self.status in [MatchStatus.PENDING, MatchStatus.ACTIVE]
    
    def to_result_dict(self) -> Dict:
        """Convert to result dictionary for saving."""
        return {
            "match_id": self.match_id,
            "model1": self.model1_name,
            "model2": self.model2_name,
            "outcome": self.outcome,
            "payoffs": self.payoffs,
            "rounds": self.current_round,
            "transcript": self.conversation_history,
            "final_offer": self.final_offer,
            "error": self.error,
            "api_calls": self.api_calls,
        }


# =============================================================================
# RESPONSE PARSER
# =============================================================================

class ResponseParser:
    """Parse LLM responses into structured moves."""
    
    @staticmethod
    def parse_response(response: str, pot_size: float = 100.0) -> Dict:
        """Parse a negotiation response into structured format."""
        import re
        
        response_upper = response.upper()
        
        # Check for ACCEPT
        if "ACCEPT" in response_upper and "REJECT" not in response_upper:
            return {"move_type": MoveType.ACCEPT, "offer": None, "message": response}
        
        # Check for WALK_AWAY
        if "WALK_AWAY" in response_upper or "WALK AWAY" in response_upper:
            return {"move_type": MoveType.WALK_AWAY, "offer": None, "message": response}
        
        # Check for PROPOSE (with or without REJECT)
        propose_match = re.search(
            r'(?:PROPOSE|PROPOSAL|OFFER)[:\s]+.*?(?:I\s+(?:get|take|receive)|MY\s+SHARE)[:\s]*\$?(\d+(?:\.\d+)?)',
            response_upper
        )
        if not propose_match:
            # Try alternative patterns
            propose_match = re.search(r'\$(\d+(?:\.\d+)?)\s*[Mm](?:illion)?.*?(?:for me|my share|I get)', response)
            if not propose_match:
                propose_match = re.search(r'(\d+)/(\d+)\s*split', response)
                if propose_match:
                    # Handle X/Y split format
                    amount = float(propose_match.group(1))
                    return {
                        "move_type": MoveType.PROPOSE if "REJECT" not in response_upper else MoveType.COUNTER,
                        "offer": amount,
                        "message": response
                    }
        
        if propose_match:
            amount = float(propose_match.group(1))
            move_type = MoveType.COUNTER if "REJECT" in response_upper else MoveType.PROPOSE
            return {"move_type": move_type, "offer": amount, "message": response}
        
        # Fallback: try to find any dollar amount
        amounts = re.findall(r'\$?(\d+(?:\.\d+)?)\s*[Mm]?', response)
        if amounts:
            # Take the first reasonable amount
            for amt in amounts:
                val = float(amt)
                if 0 < val <= pot_size:
                    return {
                        "move_type": MoveType.PROPOSE,
                        "offer": val,
                        "message": response
                    }
        
        # Unable to parse - treat as invalid
        return {"move_type": None, "offer": None, "message": response, "parse_error": True}


# =============================================================================
# HYBRID BATCH TOURNAMENT
# =============================================================================

class HybridBatchTournament:
    """
    Runs tournament using hybrid batch approach:
    - All Round 1 requests batched together
    - Wait for completion
    - All Round 2 requests batched together
    - And so on...
    """
    
    def __init__(
        self,
        models: Dict[str, ModelConfig],
        config: TournamentConfig,
        poll_interval: int = 30,
    ):
        self.models = models
        self.config = config
        self.poll_interval = poll_interval
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize batch managers
        self.anthropic_manager = None
        self.openai_manager = None
        self.google_manager = None
        
        # Context caches (for cost savings)
        self._gemini_cache_name: Optional[str] = None
        
        # Match states
        self.matches: List[MatchState] = []
        self.total_matches = 0
        
        # Tracking
        self.batch_history: List[Dict] = []
        
        # Incremental saving
        self.results_file: Optional[Path] = None
        self.completed_count = 0
    
    def _init_context_caches(self):
        """Initialize context caches for providers that support explicit caching."""
        # Gemini: Create explicit context cache for system prompt
        # This saves on repeated token processing costs
        gemini_models = [m for m in self.models.values() if m.provider == "google"]
        if gemini_models:
            try:
                manager = self._get_batch_manager("google")
                # Use a generic system prompt template for caching
                # (the actual role will vary but structure is same)
                base_prompt = MASTER_SYSTEM_PROMPT.format(
                    role="[ROLE]",  # Placeholder
                    pot_size=self.config.pot_size,
                    max_rounds=self.config.max_rounds,
                )
                model = gemini_models[0].model
                self._gemini_cache_name = manager.create_context_cache(
                    system_prompt=base_prompt,
                    model=model,
                    ttl_seconds=7200,  # 2 hours
                    display_name=f"tournament-{self.timestamp}",
                )
            except Exception as e:
                print(f"[Cache] Gemini cache creation failed: {e}")
                self._gemini_cache_name = None
    
    def initialize_matches(self) -> List[MatchState]:
        """Create all match states for the tournament."""
        matches = []
        match_num = 0
        model_names = list(self.models.keys())
        
        for i, name1 in enumerate(model_names):
            for name2 in model_names[i+1:]:
                config1 = self.models[name1]
                config2 = self.models[name2]
                
                for match_idx in range(self.config.matches_per_pair):
                    match_num += 1
                    
                    # Alternate who goes first
                    if match_idx % 2 == 0:
                        m1_name, m2_name = name1, name2
                        m1_config, m2_config = config1, config2
                    else:
                        m1_name, m2_name = name2, name1
                        m1_config, m2_config = config2, config1
                    
                    match = MatchState(
                        match_id=f"M{match_num:03d}",
                        model1_name=m1_name,
                        model2_name=m2_name,
                        model1_config=m1_config,
                        model2_config=m2_config,
                        status=MatchStatus.ACTIVE,
                    )
                    matches.append(match)
        
        self.matches = matches
        self.total_matches = len(matches)
        return matches
    
    def _get_batch_manager(self, provider: str):
        """Get or create batch manager for provider."""
        if provider == "anthropic":
            if self.anthropic_manager is None:
                self.anthropic_manager = AnthropicBatchManager()
            return self.anthropic_manager
        elif provider == "openai":
            if self.openai_manager is None:
                self.openai_manager = OpenAIBatchManager()
            return self.openai_manager
        elif provider == "google":
            if self.google_manager is None:
                self.google_manager = GoogleBatchManager()
            return self.google_manager
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _build_batch_request(
        self,
        match: MatchState,
        custom_id: str,
    ) -> Tuple[str, Dict]:
        """Build a batch request for a match's current turn with caching optimizations."""
        config = match.get_current_model_config()
        system_prompt = match.get_system_prompt()
        user_prompt = match.get_user_prompt()
        
        if config.provider == "anthropic":
            # Anthropic: Explicit cache_control on system prompt
            # 50% input cost savings on cache hits
            request = BatchRequestBuilder.build_anthropic_request(
                custom_id=custom_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=config.model,
                temperature=config.temperature,
                cache_system_prompt=True,  # Enable prompt caching
            )
        elif config.provider == "openai":
            # OpenAI: Implicit caching (static content first, dynamic last)
            # Automatic prefix-based caching
            request = BatchRequestBuilder.build_openai_request(
                custom_id=custom_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=config.model,
                temperature=config.temperature,
                reasoning_effort=config.reasoning_effort,
                cache_retention="24h",  # Extended retention for gpt-5.1
            )
        elif config.provider == "google":
            # Gemini: Use explicit context cache if available
            request = BatchRequestBuilder.build_google_request(
                key=custom_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=config.model,
                temperature=config.temperature,
                cached_content_name=self._gemini_cache_name,
            )
        else:
            raise ValueError(f"Unknown provider: {config.provider}")
        
        return config.provider, request
    
    def _collect_round_requests(self) -> Dict[str, List[Tuple[MatchState, Dict]]]:
        """Collect all pending requests for active matches, grouped by provider."""
        requests_by_provider: Dict[str, List[Tuple[MatchState, Dict]]] = {
            "anthropic": [],
            "openai": [],
            "google": [],
        }
        
        for match in self.matches:
            if not match.is_active():
                continue
            
            custom_id = f"{match.match_id}_R{match.current_round}_{match.current_player}"
            provider, request = self._build_batch_request(match, custom_id)
            requests_by_provider[provider].append((match, request))
        
        return requests_by_provider
    
    def _submit_anthropic_batch(self, requests: List[Tuple[MatchState, Dict]]) -> Optional[str]:
        """Submit batch to Anthropic API."""
        if not requests:
            return None
        
        manager = self._get_batch_manager("anthropic")
        batch_requests = [req for _, req in requests]
        
        try:
            batch_id = manager.create_batch(batch_requests)
            print(f"  [Anthropic] Submitted batch {batch_id} with {len(requests)} requests")
            return batch_id
        except Exception as e:
            print(f"  [Anthropic] ERROR: {e}")
            return None
    
    def _submit_openai_batch(self, requests: List[Tuple[MatchState, Dict]]) -> Optional[str]:
        """Submit batch to OpenAI API."""
        if not requests:
            return None
        
        manager = self._get_batch_manager("openai")
        batch_requests = [req for _, req in requests]
        
        try:
            batch_id = manager.create_batch(batch_requests)
            print(f"  [OpenAI] Submitted batch {batch_id} with {len(requests)} requests")
            return batch_id
        except Exception as e:
            print(f"  [OpenAI] ERROR: {e}")
            return None
    
    def _submit_google_batch(self, requests: List[Tuple[MatchState, Dict]]) -> Optional[str]:
        """Submit batch to Google/Gemini API."""
        if not requests:
            return None
        
        manager = self._get_batch_manager("google")
        batch_requests = [req for _, req in requests]
        
        # Get model from first request's config
        model = requests[0][0].get_current_model_config().model
        
        try:
            batch_id = manager.create_batch(batch_requests, model=model)
            print(f"  [Google] Submitted batch {batch_id} with {len(requests)} requests")
            return batch_id
        except Exception as e:
            print(f"  [Google] ERROR: {e}")
            return None
    
    def _call_gemini_realtime(self, match: MatchState) -> Optional[str]:
        """Call Gemini API in real-time with context caching if available."""
        try:
            from google import genai
            
            config = match.get_current_model_config()
            client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
            
            system_prompt = match.get_system_prompt()
            user_prompt = match.get_user_prompt()
            
            # Build generation config
            gen_config = {
                "temperature": config.temperature,
                "thinking_config": {"thinking_budget": 8000},
            }
            
            # Use cached context if available (saves token processing costs)
            if self._gemini_cache_name:
                gen_config["cached_content"] = self._gemini_cache_name
                contents = user_prompt  # Only send dynamic content
            else:
                contents = f"{system_prompt}\n\n{user_prompt}"
            
            response = client.models.generate_content(
                model=config.model,
                contents=contents,
                config=gen_config,
            )
            
            # Extract text from response
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'candidates') and response.candidates:
                return response.candidates[0].content.parts[0].text
            else:
                return str(response)
                
        except Exception as e:
            print(f"  [Gemini] ERROR for {match.match_id}: {e}")
            return None
    
    def _wait_for_batch(self, provider: str, batch_id: str) -> Dict[str, str]:
        """Wait for batch to complete and return results mapped by custom_id."""
        manager = self._get_batch_manager(provider)
        if manager is None:
            return {}
        
        print(f"  [{provider.title()}] Waiting for batch {batch_id}...")
        
        while True:
            status = manager.get_status(batch_id)
            state = status.get("status", "unknown")
            
            if state in ["ended", "completed", "succeeded"]:
                print(f"  [{provider.title()}] Batch completed!")
                break
            elif state in ["failed", "cancelled", "expired"]:
                print(f"  [{provider.title()}] Batch {state}!")
                return {}
            else:
                # Show progress
                counts = status.get("request_counts", {})
                print(f"  [{provider.title()}] Status: {state} | {counts}")
                time.sleep(self.poll_interval)
        
        # Retrieve results
        results = manager.get_results(batch_id)
        parsed_results = {}
        
        for r in results:
            custom_id = r.get("custom_id")
            if not custom_id:
                continue
            
            # Extract text based on provider format
            if provider == "anthropic":
                # Anthropic format: result.message.content[0].text (for text blocks)
                # or result.message.content[1].text if thinking block is first
                result = r.get("result", {})
                message = result.get("message", {})
                content = message.get("content", [])
                text = ""
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        break
                parsed_results[custom_id] = text
                
            elif provider == "openai":
                # OpenAI format: response.body.choices[0].message.content
                response = r.get("response", {})
                body = response.get("body", {})
                choices = body.get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content", "")
                    parsed_results[custom_id] = text
            
            else:
                # Fallback: try common patterns
                parsed_results[custom_id] = str(r.get("response", r.get("content", "")))
        
        return parsed_results
    
    def _process_response(self, match: MatchState, response_text: str):
        """Process a response and update match state."""
        parsed = ResponseParser.parse_response(response_text, match.pot_size)
        
        # Record the message
        match.conversation_history.append({
            "round": match.current_round,
            "player": match.current_player,
            "message": response_text,
            "parsed": parsed,
        })
        
        # Check for game-ending conditions
        move_type = parsed.get("move_type")
        
        if move_type == MoveType.ACCEPT:
            # Find the last offer to accept
            last_offer = None
            for msg in reversed(match.conversation_history[:-1]):
                if msg["parsed"].get("offer") is not None:
                    last_offer = msg["parsed"]["offer"]
                    # The offer is from the other player's perspective
                    break
            
            if last_offer is not None:
                # Calculate payoffs (offer was what opponent wanted)
                if match.current_player == "player_1":
                    # Player 1 accepts Player 2's offer
                    match.payoffs = {"player_1": match.pot_size - last_offer, "player_2": last_offer}
                else:
                    # Player 2 accepts Player 1's offer
                    match.payoffs = {"player_1": last_offer, "player_2": match.pot_size - last_offer}
                
                match.outcome = "agreement"
                match.status = MatchStatus.COMPLETED
                match.final_offer = {"amount": last_offer, "accepted_by": match.current_player}
            else:
                # No offer to accept - treat as error
                match.error = "ACCEPT without prior offer"
                match.status = MatchStatus.ERROR
        
        elif move_type == MoveType.WALK_AWAY:
            # Both get salvage value
            salvage = 10.0  # $10M each if no deal
            match.payoffs = {"player_1": salvage, "player_2": salvage}
            match.outcome = "walk_away"
            match.status = MatchStatus.COMPLETED
        
        elif parsed.get("parse_error"):
            # Give them another chance or mark as error
            match.error = f"Could not parse response: {response_text[:100]}"
            match.status = MatchStatus.ERROR
        
        else:
            # Valid proposal/counter - continue game
            if match.current_round >= match.max_rounds and match.current_player == "player_2":
                # Max rounds reached without agreement
                salvage = 10.0
                match.payoffs = {"player_1": salvage, "player_2": salvage}
                match.outcome = "max_rounds"
                match.status = MatchStatus.COMPLETED
            else:
                # Advance to next turn
                match.advance_turn()
    
    def run_round(self, round_num: int, verbose: bool = True) -> int:
        """Run one round of batched requests. Returns number of active matches remaining."""
        if verbose:
            active_count = sum(1 for m in self.matches if m.is_active())
            print(f"\n{'='*60}")
            print(f"ROUND {round_num} - {active_count} active matches")
            print(f"{'='*60}")
        
        # Collect requests by provider
        requests_by_provider = self._collect_round_requests()
        
        # Submit batches
        batch_ids = {}
        
        if requests_by_provider["anthropic"]:
            batch_ids["anthropic"] = self._submit_anthropic_batch(requests_by_provider["anthropic"])
        
        if requests_by_provider["openai"]:
            batch_ids["openai"] = self._submit_openai_batch(requests_by_provider["openai"])
        
        # Handle Google/Gemini - try batch first, fallback to real-time
        if requests_by_provider["google"]:
            batch_id = self._submit_google_batch(requests_by_provider["google"])
            if batch_id:
                batch_ids["google"] = batch_id
            else:
                # Fallback to real-time
                print("  [Google] Batch failed, using real-time fallback...")
                gemini_results = {}
                for match, _ in requests_by_provider["google"]:
                    custom_id = f"{match.match_id}_R{match.current_round}_{match.current_player}"
                    response = self._call_gemini_realtime(match)
                    if response:
                        gemini_results[custom_id] = response
                    time.sleep(2.5)  # Rate limit: 25 RPM = 2.4s between calls
                all_results.update(gemini_results)
        
        # Wait for batch results
        all_results = {}
        
        for provider, batch_id in batch_ids.items():
            if batch_id:
                results = self._wait_for_batch(provider, batch_id)
                all_results.update(results)
        
        all_results.update(gemini_results)
        
        # Process results and update match states
        newly_completed = 0
        for match in self.matches:
            if not match.is_active():
                continue
            
            custom_id = f"{match.match_id}_R{match.current_round}_{match.current_player}"
            
            if custom_id in all_results:
                response_text = all_results[custom_id]
                was_active = match.is_active()
                self._process_response(match, response_text)
                
                # Save immediately when match completes
                if was_active and not match.is_active():
                    self._save_match_result(match)
                    newly_completed += 1
                    
                    if verbose:
                        print(f"  ✓ [{match.match_id}] {match.outcome}: ${match.payoffs.get('player_1', 0):.0f}M / ${match.payoffs.get('player_2', 0):.0f}M (saved)")
            else:
                # No response received
                match.error = f"No response for {custom_id}"
                match.status = MatchStatus.ERROR
                self._save_match_result(match)
                newly_completed += 1
        
        if verbose and newly_completed > 0:
            print(f"  → {newly_completed} matches completed this round, {self.completed_count}/{self.total_matches} total")
        
        # Return count of still-active matches
        return sum(1 for m in self.matches if m.is_active())
    
    def run(self, verbose: bool = True) -> List[Dict]:
        """Run the full tournament."""
        print("\n" + "="*70)
        print("HYBRID BATCH TOURNAMENT")
        print("="*70)
        
        # Initialize all matches
        self.initialize_matches()
        print(f"Initialized {self.total_matches} matches")
        print(f"Models: {', '.join(self.models.keys())}")
        
        # Initialize context caches for cost savings
        print("\nInitializing context caches...")
        self._init_context_caches()
        
        # Initialize incremental results file
        self._init_results_file()
        
        # Run rounds until all matches complete
        round_num = 0
        max_total_rounds = self.config.max_rounds * 2 + 2  # Safety limit
        
        while round_num < max_total_rounds:
            round_num += 1
            active_remaining = self.run_round(round_num, verbose)
            
            if active_remaining == 0:
                print(f"\n✓ All matches completed after {round_num} rounds")
                break
        
        # Compile results
        results = [m.to_result_dict() for m in self.matches]
        
        return results
    
    def _init_results_file(self) -> Path:
        """Initialize the incremental results file."""
        output_dir = Path(__file__).parent / self.config.output_dir
        output_dir.mkdir(exist_ok=True)
        
        filename = f"hybrid_tournament_{self.timestamp}.jsonl"
        self.results_file = output_dir / filename
        
        # Write header with metadata
        header = {
            "_type": "header",
            "timestamp": self.timestamp,
            "config": asdict(self.config),
            "models": {name: asdict(m) for name, m in self.models.items()},
            "total_matches": self.total_matches,
        }
        
        with open(self.results_file, 'w') as f:
            f.write(json.dumps(header, default=str) + '\n')
        
        print(f"Results will be saved incrementally to: {self.results_file}")
        return self.results_file
    
    def _save_match_result(self, match: MatchState):
        """Save a single match result immediately (append to JSONL)."""
        if self.results_file is None:
            self._init_results_file()
        
        result = match.to_result_dict()
        result["_type"] = "match"
        
        with open(self.results_file, 'a') as f:
            f.write(json.dumps(result, default=str) + '\n')
        
        self.completed_count += 1
    
    def save_results(self, results: List[Dict], suffix: str = "") -> Path:
        """Save final tournament results (JSON format for analysis)."""
        output_dir = Path(__file__).parent / self.config.output_dir
        output_dir.mkdir(exist_ok=True)
        
        filename = f"hybrid_tournament_{self.timestamp}_FINAL{suffix}.json"
        output_path = output_dir / filename
        
        data = {
            "timestamp": self.timestamp,
            "config": asdict(self.config),
            "models": {name: asdict(m) for name, m in self.models.items()},
            "total_matches": self.total_matches,
            "completed_matches": self.completed_count,
            "matches": results,
            "batch_history": self.batch_history,
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"\nFinal results saved to: {output_path}")
        print(f"Incremental results in: {self.results_file}")
        return output_path
    
    def print_summary(self, results: List[Dict]):
        """Print tournament summary."""
        print(f"\n{'='*70}")
        print("TOURNAMENT SUMMARY")
        print(f"{'='*70}")
        
        # Stats per model
        model_stats = {name: {"wins": 0, "losses": 0, "draws": 0, "total_payoff": 0, "matches": 0}
                      for name in self.models}
        
        outcomes = {"agreement": 0, "walk_away": 0, "max_rounds": 0, "error": 0}
        
        for r in results:
            if r.get("error"):
                outcomes["error"] += 1
                continue
            
            outcome = r.get("outcome", "error")
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            
            m1, m2 = r["model1"], r["model2"]
            p1 = r["payoffs"].get("player_1", 0)
            p2 = r["payoffs"].get("player_2", 0)
            
            model_stats[m1]["matches"] += 1
            model_stats[m2]["matches"] += 1
            model_stats[m1]["total_payoff"] += p1
            model_stats[m2]["total_payoff"] += p2
            
            if p1 > p2:
                model_stats[m1]["wins"] += 1
                model_stats[m2]["losses"] += 1
            elif p2 > p1:
                model_stats[m2]["wins"] += 1
                model_stats[m1]["losses"] += 1
            else:
                model_stats[m1]["draws"] += 1
                model_stats[m2]["draws"] += 1
        
        print("\nOutcomes:")
        for outcome, count in outcomes.items():
            print(f"  {outcome}: {count}")
        
        print("\nModel Performance:")
        for name, stats in model_stats.items():
            if stats["matches"] > 0:
                avg = stats["total_payoff"] / stats["matches"]
                print(f"  {name}:")
                print(f"    W/L/D: {stats['wins']}/{stats['losses']}/{stats['draws']}")
                print(f"    Avg Payoff: ${avg:.1f}M")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run hybrid batch negotiation tournament")
    parser.add_argument("--matches-per-pair", type=int, default=100)
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between batch status checks")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    parser.add_argument("--test", action="store_true", help="Run small test (5 matches per pair)")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(TOURNAMENT_MODELS.keys()),
        default=list(TOURNAMENT_MODELS.keys()),
    )
    
    args = parser.parse_args()
    
    # Override matches for test mode
    if args.test:
        args.matches_per_pair = 5
        print("\n⚠️  TEST MODE: Running 5 matches per pair")
    
    # Select models
    models = {k: TOURNAMENT_MODELS[k] for k in args.models}
    
    # Create config
    config = TournamentConfig(
        matches_per_pair=args.matches_per_pair,
        max_rounds=10,
        use_batch_api=True,
        output_dir="tournament_results",
    )
    
    # Calculate costs (50% discount for batch)
    n_pairings = len(models) * (len(models) - 1) // 2
    total_matches = n_pairings * args.matches_per_pair
    
    print("\n" + "="*70)
    print("HYBRID BATCH TOURNAMENT CONFIGURATION")
    print("="*70)
    print(f"Models: {', '.join(models.keys())}")
    print(f"Pairings: {n_pairings}")
    print(f"Matches per pair: {args.matches_per_pair}")
    print(f"Total matches: {total_matches}")
    print(f"Batch API: Enabled (50% cost savings)")
    print(f"Poll interval: {args.poll_interval}s")
    
    # Estimate costs based on ACTUAL run data ($1.58 for ~2 Claude matches = ~$0.79/match)
    # WITH CACHING + BATCH OPTIMIZATIONS:
    # - Batch API: 50% discount on all providers
    # - Anthropic cache_control: ~50% input savings on cache hits
    # - OpenAI implicit caching: ~50% input savings on prefix matches
    # - Gemini context caching: Variable savings (storage + reduced processing)
    #
    # Cost per model per match (base rate without optimizations):
    cost_per_model_per_match_base = {
        "claude-opus-4.5": 0.80,  # $5 input, $25 output (with extended thinking)
        "gemini-3-pro": 0.35,     # $2 input, $12 output (with thinking)
        "gpt-5.1-high": 0.30,     # $1.25 input, $10 output (with reasoning)
    }
    
    # With caching, we estimate ~30% additional savings on input tokens
    # (system prompt is ~40% of input, and caching saves ~50% on that portion)
    CACHE_SAVINGS_MULTIPLIER = 0.85  # 15% additional savings from caching
    
    cost_per_model_per_match = {
        k: v * CACHE_SAVINGS_MULTIPLIER for k, v in cost_per_model_per_match_base.items()
    }
    
    # Calculate by pairing
    model_list = list(models.keys())
    total_realtime = 0.0
    print("\nCost breakdown by pairing (UPDATED from actual usage):")
    
    for i in range(len(model_list)):
        for j in range(i+1, len(model_list)):
            m1, m2 = model_list[i], model_list[j]
            cost1 = cost_per_model_per_match.get(m1, 0.50)
            cost2 = cost_per_model_per_match.get(m2, 0.50)
            pairing_cost = (cost1 + cost2) * args.matches_per_pair
            total_realtime += pairing_cost
            print(f"  {models[m1].name} vs {models[m2].name}: ${pairing_cost:.2f}")
    
    total_batch = total_realtime * 0.5
    print(f"  {'─'*40}")
    print(f"  TOTAL (real-time): ${total_realtime:.2f}")
    print(f"  TOTAL (batch 50% off): ${total_batch:.2f}")
    print("\nEstimated time: 1-2 hours (batch processing)")
    
    print("\n" + "="*70)
    print("NOTE: Gemini will use real-time API (batch not well supported)")
    print("="*70)
    
    if not args.yes:
        response = input("\nProceed with tournament? [y/N]: ").strip().lower()
        if response != 'y':
            print("Cancelled.")
            sys.exit(0)
    else:
        print("\n[--yes flag: Skipping confirmation]")
    
    # Run tournament
    tournament = HybridBatchTournament(
        models=models,
        config=config,
        poll_interval=args.poll_interval,
    )
    
    results = tournament.run(verbose=True)
    
    # Save and summarize
    tournament.save_results(results)
    tournament.print_summary(results)
    
    print("\n✓ Tournament complete!")


if __name__ == "__main__":
    main()
