#!/usr/bin/env python3
"""
Batch Tournament System: Cost-Effective Large-Scale LLM Negotiations

Uses batch APIs from all three providers for 50% cost reduction:
- Anthropic: client.messages.batches.create()
- Google: client.batches.create()
- OpenAI: client.batches.create()

Designed for head-to-head pairings where each model plays every other model N times.
"""

import os
import sys
import json
import time
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from itertools import combinations
import threading

# Load environment variables
load_dotenv(Path(__file__).parent / ".env")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.personas import get_system_prompt, MASTER_SYSTEM_PROMPT


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a tournament model."""
    name: str
    provider: Literal["anthropic", "google", "openai"]
    model: str
    temperature: float = 0.7
    reasoning_effort: Optional[str] = None  # For OpenAI reasoning models
    
    # Pricing per million tokens (Dec 2025)
    input_price: float = 0.0
    output_price: float = 0.0
    thinking_price: float = 0.0  # For models with separate thinking tokens
    
    # Rate limits
    rpm: int = 100
    tpm: int = 100_000
    
    # External metadata for cross-lab analysis
    lab: str = ""  # anthropic, google, openai, etc.
    lm_arena_score: Optional[int] = None  # LM Arena ELO (Dec 2025)
    lm_arena_rank: Optional[int] = None  # LM Arena rank
    
    def to_dict(self) -> Dict:
        return asdict(self)


# Default tournament models (SOTA Dec 2025)
# External scores from LM Arena (Dec 4, 2025)
TOURNAMENT_MODELS = {
    "claude-opus-4.5": ModelConfig(
        name="Claude-Opus-4.5",
        provider="anthropic",
        model="claude-opus-4-5-20251101",
        temperature=1.0,  # Required for extended thinking
        input_price=5.00,
        output_price=25.00,
        rpm=1000,  # Tier 2: 1,000 RPM
        tpm=450_000,  # Tier 2: 450K input TPM (90K output TPM)
        lab="anthropic",
        lm_arena_score=1535,  # Rank #1 (Dec 4, 2025)
        lm_arena_rank=1,
    ),
    "gemini-3-pro": ModelConfig(
        name="Gemini-3-Pro",
        provider="google",
        model="gemini-3-pro-preview",
        temperature=1.0,  # Gemini 3 recommends temperature=1.0 for best reasoning
        input_price=2.00,  # prompts <= 200k
        output_price=12.00,  # includes thinking tokens
        rpm=25,  # Paid tier: 25 RPM (free tier: 6 RPM)
        tpm=1_000_000,  # Paid tier: 1M TPM
        lab="google",
        lm_arena_score=1499,  # Rank #4 (Dec 4, 2025)
        lm_arena_rank=4,
    ),
    "gpt-5.1": ModelConfig(
        name="GPT-5.1",
        provider="openai",
        model="gpt-5.1",
        temperature=1.0,
        reasoning_effort=None,  # Standard mode (no extended thinking)
        input_price=1.25,
        output_price=10.00,
        rpm=5000,  # Updated: 5,000 RPM
        tpm=1_000_000,  # Updated: 1M TPM
        lab="openai",
        lm_arena_score=1479,
        lm_arena_rank=7,
    ),
    "gpt-5.1-high": ModelConfig(
        name="GPT-5.1-High",
        provider="openai",
        model="gpt-5.1",
        temperature=1.0,
        reasoning_effort="high",  # Extended thinking enabled
        input_price=1.25,
        output_price=10.00,  # includes reasoning
        rpm=5000,  # Updated: 5,000 RPM
        tpm=1_000_000,  # Updated: 1M TPM
        lab="openai",
        lm_arena_score=1479,
        lm_arena_rank=7,
    ),
}

# Batch API pricing discount
BATCH_DISCOUNT = 0.50  # 50% off standard pricing


@dataclass
class TournamentConfig:
    """Configuration for a tournament."""
    matches_per_pair: int = 100  # Each pair plays N times
    pot_size: float = 100.0
    max_rounds: int = 10
    reservation_value: float = 10.0
    use_batch_api: bool = True  # Use batch APIs for 50% cost reduction
    poll_interval: int = 30  # Seconds between batch status checks
    max_wait_time: int = 86400  # 24 hours max wait
    output_dir: str = "tournament_results"


# =============================================================================
# BATCH REQUEST BUILDERS
# =============================================================================

def build_negotiation_prompt(
    agent_role: str,
    round_num: int,
    conversation_history: List[Dict],
    pot_size: float = 100.0,
) -> str:
    """Build a negotiation prompt for a single turn."""
    # Format conversation history
    history_text = ""
    if conversation_history:
        for msg in conversation_history:
            history_text += f"\n[Round {msg['round']}] {msg['player']}: {msg['message']}"
    else:
        history_text = "\n(No messages yet - you are making the first offer)"
    
    prompt = f"""Current negotiation state:
- Your role: {agent_role}
- Pot size: ${pot_size}M to divide
- Current round: {round_num}
- Conversation history:{history_text}

Please make your next move. You can:
1. Make a PROPOSAL: "PROPOSE: I get $X, you get $Y"
2. ACCEPT the last proposal: "ACCEPT"
3. REJECT and counter: "REJECT: [reason]. PROPOSE: I get $X, you get $Y"
4. Walk away: "WALK_AWAY"

Respond with your decision:"""
    
    return prompt


class BatchRequestBuilder:
    """Builds batch requests for each provider's API format."""
    
    @staticmethod
    def build_anthropic_request(
        custom_id: str,
        system_prompt: str,
        user_prompt: str,
        model: str = "claude-opus-4-5-20251101",
        temperature: float = 1.0,
        cache_system_prompt: bool = True,
    ) -> Dict:
        """Build Anthropic batch request format with prompt caching.
        
        Caching spec:
        - cache_control must be at end of content block
        - TTL: "ephemeral" (5 min) or extend with "ttl": "1h"
        - Min tokens: 1024 for Sonnet/Opus
        - Max 4 cache breakpoints per request
        - Hierarchy: Tools → System → Messages (changes invalidate downstream)
        """
        # System prompt with cache_control for 50% input cost savings
        # Our system prompt is ~800 tokens, but with history it exceeds 1024
        system_block = {
            "type": "text",
            "text": system_prompt,
        }
        if cache_system_prompt:
            # Cache the static system prompt (same across all turns)
            system_block["cache_control"] = {"type": "ephemeral"}
        
        return {
            "custom_id": custom_id,
            "params": {
                "model": model,
                "max_tokens": 16000,
                "temperature": temperature,
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 10000,
                },
                "system": [system_block],
                "messages": [
                    {"role": "user", "content": user_prompt}
                ],
            }
        }
    
    @staticmethod
    def build_google_request(
        key: str,
        system_prompt: str,
        user_prompt: str,
        model: str = "gemini-3-pro-preview",
        temperature: float = 0.7,
        cached_content_name: Optional[str] = None,
    ) -> Tuple[str, Dict]:
        """Build Google/Gemini batch request format.
        
        Returns (key, request) tuple. The format depends on batching mode:
        - Inline: List of GenerateContentRequest objects (no key wrapper)
        - File: JSONL with {"key": ..., "request": ...} per line
        
        Caching spec:
        - Implicit (Gemini 2.5+): Auto for prompts >1024 tokens
        - Explicit: Use cached_content_name from client.caches.create()
        
        Batch spec:
        - Inline: <20MB, list of raw requests
        - File: JSONL via File API (>20MB, up to 2GB)
        - NOT idempotent: re-running creates duplicate jobs!
        """
        # Build the GenerateContentRequest for inline batching
        # Per docs: system_instruction goes in 'config' field for batch requests
        request = {
                "contents": [
                {
                    "parts": [{"text": user_prompt}],
                    "role": "user"  # Required for inline batch
                }
                ],
            "config": {
                    "temperature": temperature,
                },
        }
        
        # If we have a cached context (system prompt), reference it
        if cached_content_name:
            request["cached_content"] = cached_content_name
        else:
            # No cache - include system prompt in config
            # Per docs: 'config': {'system_instruction': {'parts': [...]}}
            request["config"]["system_instruction"] = {
                "parts": [{"text": system_prompt}]
            }
        
        # Return tuple - caller decides how to use based on batching mode
        return (key, request)
    
    @staticmethod
    def build_openai_request(
        custom_id: str,
        system_prompt: str,
        user_prompt: str,
        model: str = "gpt-5.1",
        temperature: float = 1.0,
        reasoning_effort: Optional[str] = None,
        cache_retention: str = "24h",
    ) -> Dict:
        """Build OpenAI batch request format (JSONL line).
        
        Caching spec (IMPLICIT - automatic):
        - Static content MUST be at the very start (system prompt)
        - Dynamic variables go at the end (user prompt with history)
        - Min prefix: 1024 tokens, hits in 128-token increments
        - Retention: "24h" for gpt-5.1/gpt-4.1 families
        - Do NOT use stream=true in batch
        """
        body = {
            "model": model,
            "temperature": temperature,
            # STATIC FIRST: System prompt cached automatically by prefix matching
            "messages": [
                {"role": "system", "content": system_prompt},
                # DYNAMIC LAST: User prompt with conversation history
                {"role": "user", "content": user_prompt},
            ],
            # NOTE: prompt_cache_retention is only valid for real-time API, not batch API
            # Batch API does not support cache retention parameter
            # stream=false is implicit (batch doesn't support streaming)
        }
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
            
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }


# =============================================================================
# BATCH JOB MANAGERS
# =============================================================================

class AnthropicBatchManager:
    """Manages Anthropic batch API jobs."""
    
    def __init__(self, api_key: Optional[str] = None):
        import anthropic
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY required")
        self.client = anthropic.Anthropic(api_key=key)
    
    def create_batch(self, requests: List[Dict], display_name: str = None) -> str:
        """Submit a batch job. Returns batch_id."""
        batch = self.client.messages.batches.create(requests=requests)
        print(f"[Anthropic] Created batch: {batch.id}")
        return batch.id
    
    def get_status(self, batch_id: str) -> Dict:
        """Get batch job status."""
        batch = self.client.messages.batches.retrieve(batch_id)
        return {
            "id": batch.id,
            "status": batch.processing_status,
            "created_at": str(batch.created_at),
            "ended_at": str(batch.ended_at) if batch.ended_at else None,
            "request_counts": {
                "total": batch.request_counts.processing + batch.request_counts.succeeded + batch.request_counts.errored + batch.request_counts.canceled + batch.request_counts.expired,
                "succeeded": batch.request_counts.succeeded,
                "errored": batch.request_counts.errored,
            },
            "results_url": batch.results_url,
        }
    
    def is_complete(self, batch_id: str) -> bool:
        """Check if batch is complete."""
        status = self.get_status(batch_id)
        return status["status"] == "ended"
    
    def get_results(self, batch_id: str) -> List[Dict]:
        """Get batch results using SDK (handles auth automatically).
        
        Returns list of dicts with structure:
        {
            "custom_id": str,
            "result": {
                "type": "succeeded" | "errored" | ...,
                "message": {...}  # if succeeded
            }
        }
        """
        results = []
        for result in self.client.messages.batches.results(batch_id):
            # Convert SDK object to dict for consistent processing
            result_dict = {
                "custom_id": result.custom_id,
                "result": {
                    "type": result.result.type,
                }
            }
            
            if result.result.type == "succeeded":
                # Extract message content
                message = result.result.message
                result_dict["result"]["message"] = {
                    "role": message.role,
                    "content": [
                        {"type": block.type, "text": getattr(block, "text", "")}
                        for block in message.content
                    ]
                }
            elif result.result.type == "errored":
                result_dict["result"]["error"] = str(result.result.error)
            
            results.append(result_dict)
        
        return results


class GoogleBatchManager:
    """Manages Google/Gemini batch API jobs with context caching."""
    
    def __init__(self, api_key: Optional[str] = None):
        from google import genai
        from google.genai import types
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY required")
        self.client = genai.Client(api_key=key)
        self.types = types
        self._context_cache: Optional[str] = None  # Cached system prompt name
        self._cache_model: Optional[str] = None
    
    def create_context_cache(
        self,
        system_prompt: str,
        model: str,
        ttl_seconds: int = 3600,  # 1 hour default
        display_name: str = "negotiation-system-prompt",
    ) -> str:
        """Create an explicit context cache for the system prompt.
        
        Spec:
        - Use client.caches.create() to upload content
        - Caches are IMMUTABLE (can't edit content, only update TTL)
        - Billing: token processing (write) + storage duration
        """
        if self._context_cache and self._cache_model == model:
            print(f"[Google] Reusing existing cache: {self._context_cache}")
            return self._context_cache
        
        try:
            # Contents must have 'role' field for caching
            # Use 'user' role since this is context/instructions
            cache = self.client.caches.create(
                model=model,
                config={
                    "contents": [
                        {
                            "parts": [{"text": system_prompt}],
                            "role": "user"  # Required for caching
                        }
                    ],
                    "display_name": display_name,
                    "ttl": f"{ttl_seconds}s",
                }
            )
            self._context_cache = cache.name
            self._cache_model = model
            print(f"[Google] Created context cache: {cache.name} (TTL: {ttl_seconds}s)")
            return cache.name
        except Exception as e:
            print(f"[Google] Failed to create cache: {e}")
            return None
    
    def get_cached_content_name(self) -> Optional[str]:
        """Get the current context cache name."""
        return self._context_cache
    
    def create_batch(self, requests: List[Tuple[str, Dict]], model: str, display_name: str = None) -> str:
        """Submit a batch job. Returns batch job name.
        
        Args:
            requests: List of (key, GenerateContentRequest) tuples
            model: Model name (e.g., "gemini-3-pro-preview")
            display_name: Optional job display name
        
        Spec:
        - Inline batching: <20MB (list of raw GenerateContentRequest objects)
        - File batching: >20MB, up to 2GB (JSONL with {key, request} per line)
        - WARNING: Not idempotent! Re-running creates duplicate jobs
        """
        # Extract just the request objects for size estimation
        raw_requests = [req for _, req in requests]
        request_size = len(json.dumps(raw_requests))
        
        if request_size > 20 * 1024 * 1024:  # >20MB
            print(f"[Google] Large batch ({request_size / 1024 / 1024:.1f}MB), using file upload...")
            return self._create_batch_from_file(requests, model, display_name)
        
        # Inline batching: pass raw GenerateContentRequest objects (NOT {key, request})
        batch_job = self.client.batches.create(
            model=model,
            src=raw_requests,  # List of GenerateContentRequest objects
            config={"display_name": display_name or f"tournament-{datetime.now().isoformat()}"},
        )
        print(f"[Google] Created inline batch: {batch_job.name}")
        return batch_job.name
    
    def _create_batch_from_file(self, requests: List[Tuple[str, Dict]], model: str, display_name: str = None) -> str:
        """Create batch from JSONL file for large requests."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for key, req in requests:
                # File format uses {key, request} wrapper
                f.write(json.dumps({"key": key, "request": req}) + '\n')
            temp_path = f.name
        
        try:
            uploaded = self.client.files.upload(
                file=temp_path,
                config=self.types.UploadFileConfig(
                    display_name=display_name or "batch-requests",
                    mime_type="application/jsonl"
                )
            )
            
            batch_job = self.client.batches.create(
                model=model,
                src=uploaded.name,
                config={"display_name": display_name or f"tournament-{datetime.now().isoformat()}"},
            )
            print(f"[Google] Created file batch: {batch_job.name}")
            return batch_job.name
        finally:
            os.unlink(temp_path)
    
    def create_batch_from_file(self, file_path: str, model: str, display_name: str = None) -> str:
        """Submit a batch job from JSONL file. Returns batch job name."""
        # Upload file first
        uploaded = self.client.files.upload(
            file=file_path,
            config=self.types.UploadFileConfig(
                display_name=display_name or "batch-requests",
                mime_type="jsonl"
            )
        )
        
        # Create batch from file
        batch_job = self.client.batches.create(
            model=model,
            src=uploaded.name,
            config={"display_name": display_name or f"tournament-{datetime.now().isoformat()}"},
        )
        print(f"[Google] Created batch from file: {batch_job.name}")
        return batch_job.name
    
    def get_status(self, batch_name: str) -> Dict:
        """Get batch job status."""
        batch = self.client.batches.get(name=batch_name)
        state = batch.state.name if hasattr(batch.state, 'name') else str(batch.state)
        
        # Normalize state to 'status' field expected by _wait_for_batch
        # Map Google states to common states
        status_map = {
            'JOB_STATE_PENDING': 'pending',
            'JOB_STATE_RUNNING': 'in_progress',
            'JOB_STATE_SUCCEEDED': 'succeeded',
            'JOB_STATE_FAILED': 'failed',
            'JOB_STATE_CANCELLED': 'cancelled',
            'JOB_STATE_EXPIRED': 'expired',
        }
        normalized_status = status_map.get(state, state.lower())
        
        return {
            "name": batch.name,
            "status": normalized_status,  # Use 'status' to match other providers
            "state": state,  # Keep original state too
            "display_name": batch.display_name if hasattr(batch, 'display_name') else None,
        }
    
    def is_complete(self, batch_name: str) -> bool:
        """Check if batch is complete."""
        status = self.get_status(batch_name)
        completed_states = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}
        return status["state"] in completed_states
    
    def get_results(self, batch_name: str) -> List[Dict]:
        """Get batch results."""
        batch = self.client.batches.get(name=batch_name)
        
        results = []
        # Check for inline responses
        if batch.dest and hasattr(batch.dest, 'inlined_responses') and batch.dest.inlined_responses:
            for resp in batch.dest.inlined_responses:
                results.append({
                    "response": resp.response if hasattr(resp, 'response') else None,
                    "error": resp.error if hasattr(resp, 'error') else None,
                })
        # Check for file output
        elif batch.dest and hasattr(batch.dest, 'file_name') and batch.dest.file_name:
            content = self.client.files.download(file=batch.dest.file_name)
            for line in content.decode('utf-8').strip().split("\n"):
                if line:
                    results.append(json.loads(line))
        
        return results


class OpenAIBatchManager:
    """Manages OpenAI batch API jobs."""
    
    def __init__(self, api_key: Optional[str] = None):
        import openai
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY required")
        self.client = openai.OpenAI(api_key=key)
    
    def create_batch(self, requests: List[Dict], display_name: str = None) -> str:
        """Submit a batch job from requests. Returns batch_id."""
        import tempfile
        
        # Write requests to temp JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for req in requests:
                f.write(json.dumps(req) + '\n')
            temp_path = f.name
        
        try:
            # Upload file
            with open(temp_path, 'rb') as f:
                batch_file = self.client.files.create(file=f, purpose="batch")
            
            # Create batch
            batch = self.client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            print(f"[OpenAI] Created batch: {batch.id}")
            return batch.id
        finally:
            os.unlink(temp_path)
    
    def get_status(self, batch_id: str) -> Dict:
        """Get batch job status."""
        batch = self.client.batches.retrieve(batch_id)
        return {
            "id": batch.id,
            "status": batch.status,
            "created_at": batch.created_at,
            "completed_at": batch.completed_at,
            "request_counts": {
                "total": batch.request_counts.total,
                "completed": batch.request_counts.completed,
                "failed": batch.request_counts.failed,
            },
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
        }
    
    def is_complete(self, batch_id: str) -> bool:
        """Check if batch is complete."""
        status = self.get_status(batch_id)
        return status["status"] in ("completed", "failed", "expired", "cancelled")
    
    def get_results(self, batch_id: str) -> List[Dict]:
        """Download and parse batch results."""
        status = self.get_status(batch_id)
        if not status["output_file_id"]:
            return []
        
        content = self.client.files.content(status["output_file_id"]).content
        results = []
        for line in content.decode('utf-8').strip().split("\n"):
            if line:
                results.append(json.loads(line))
        return results


# =============================================================================
# TOURNAMENT ORCHESTRATOR
# =============================================================================

class HeadToHeadTournament:
    """
    Orchestrates a head-to-head tournament between models.
    
    Each model plays every other model N times (matches_per_pair).
    Uses batch APIs for 50% cost reduction when enabled.
    """
    
    def __init__(
        self,
        models: List[ModelConfig],
        config: TournamentConfig = None,
    ):
        self.models = {m.name: m for m in models}
        self.config = config or TournamentConfig()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize batch managers
        self.batch_managers = {}
        
        # Generate pairings
        self.pairings = list(combinations(models, 2))
        
        # Calculate totals
        self.total_matches = len(self.pairings) * self.config.matches_per_pair
        
        print(f"\n{'='*70}")
        print("HEAD-TO-HEAD TOURNAMENT CONFIGURATION")
        print(f"{'='*70}")
        print(f"Models: {', '.join(m.name for m in models)}")
        print(f"Pairings: {len(self.pairings)}")
        print(f"Matches per pair: {self.config.matches_per_pair}")
        print(f"Total matches: {self.total_matches}")
        print(f"Batch API: {'Enabled (50% discount)' if self.config.use_batch_api else 'Disabled'}")
    
    def estimate_costs(self) -> Dict[str, float]:
        """Estimate tournament costs based on historical token usage."""
        # Token estimates per match (UPDATED based on actual run: $1.58 for ~2 Claude matches)
        # Claude with extended thinking uses MUCH more tokens than expected
        # Each negotiation has ~3-5 rounds, each round has system prompt + history + response
        TOKEN_ESTIMATES = {
            # Claude Opus 4.5 with extended thinking:
            # - Input: ~2000 tokens/turn (system prompt ~800 + history grows each turn)
            # - Output: ~500 text + ~5000 thinking tokens per turn
            # - Average 4 turns per match = ~8000 input, ~22000 output
            "Claude-Opus-4.5": {"input": 8000, "output": 22000, "calls": 4},
            # Gemini 3 Pro with thinking:
            # - Similar pattern but thinking charged at output rate
            "Gemini-3-Pro": {"input": 8000, "output": 8000, "thinking": 15000, "calls": 4},
            # GPT 5.1 with reasoning:
            # - Reasoning tokens included in output
            "GPT-5.1-High": {"input": 8000, "output": 25000, "calls": 4},
        }
        
        costs = {}
        total_cost = 0.0
        
        for model in self.models.values():
            est = TOKEN_ESTIMATES.get(model.name, {"input": 3000, "output": 2000, "calls": 2})
            
            # Each model plays in (n-1) pairings, each with matches_per_pair matches
            matches_for_model = (len(self.models) - 1) * self.config.matches_per_pair
            
            input_tokens = est["input"] * matches_for_model
            output_tokens = est["output"] * matches_for_model
            thinking_tokens = est.get("thinking", 0) * matches_for_model
            
            # Calculate cost
            if model.provider == "google":
                # Gemini charges thinking at output rate
                cost = (input_tokens * model.input_price / 1_000_000 +
                       (output_tokens + thinking_tokens) * model.output_price / 1_000_000)
            else:
                cost = (input_tokens * model.input_price / 1_000_000 +
                       output_tokens * model.output_price / 1_000_000)
            
            # Apply batch discount if enabled
            if self.config.use_batch_api:
                cost *= BATCH_DISCOUNT
            
            costs[model.name] = cost
            total_cost += cost
        
        costs["total"] = total_cost
        return costs
    
    def estimate_time(self) -> Dict[str, float]:
        """Estimate tournament time based on rate limits."""
        # Token estimates per match
        TOKEN_ESTIMATES = {
            "Claude-Opus-4.5": {"input": 1900, "output": 531, "calls": 1.5},
            "Gemini-3-Pro": {"input": 4031, "output": 1019, "thinking": 12061, "calls": 3.0},
            "GPT-5.1-High": {"input": 2351, "output": 6239, "calls": 2.0},
        }
        
        times = {}
        
        for model in self.models.values():
            est = TOKEN_ESTIMATES.get(model.name, {"input": 3000, "output": 2000, "calls": 2})
            matches_for_model = (len(self.models) - 1) * self.config.matches_per_pair
            
            total_calls = est["calls"] * matches_for_model
            time_minutes = total_calls / model.rpm
            times[model.name] = time_minutes
        
        # Bottleneck is the slowest model
        times["bottleneck"] = max(times.values())
        
        return times
    
    def print_estimates(self):
        """Print cost and time estimates."""
        costs = self.estimate_costs()
        times = self.estimate_time()
        
        print(f"\n{'='*70}")
        print("COST & TIME ESTIMATES")
        print(f"{'='*70}")
        
        print("\nCost breakdown (with batch API 50% discount):" if self.config.use_batch_api else "\nCost breakdown (standard pricing):")
        for model in self.models.values():
            print(f"  {model.name}: ${costs[model.name]:.2f}")
        print(f"  {'─'*30}")
        print(f"  TOTAL: ${costs['total']:.2f}")
        
        print(f"\nTime estimates (sequential per model):")
        for model in self.models.values():
            print(f"  {model.name}: {times[model.name]:.1f} minutes")
        print(f"\nWith batch API: ~1 hour (batches process in parallel)")
        print(f"Real-time parallel: ~{times['bottleneck']:.1f} minutes (bottleneck)")
    
    def run_real_time(self, verbose: bool = True) -> List[Dict]:
        """Run tournament in real-time (not batch). Returns results."""
        import time as time_module
        from environments.negotiation_arena import NegotiationArena
        from agents.llm_agent import LLMAgent
        
        results = []
        match_num = 0
        
        # Track API calls for rate limiting
        api_call_times = {name: [] for name in self.models}
        
        for model1, model2 in self.pairings:
            print(f"\n{'='*60}")
            print(f"PAIRING: {model1.name} vs {model2.name}")
            print(f"{'='*60}")
            
            for i in range(self.config.matches_per_pair):
                match_num += 1
                match_id = f"M{match_num:03d}"
                
                if verbose:
                    print(f"\n[{match_id}] {model1.name} vs {model2.name} (match {i+1}/{self.config.matches_per_pair})")
                
                try:
                    # Create agents
                    agent1 = LLMAgent(
                        agent_id="player_1",
                        persona_type="default",
                        model=model1.model,
                        provider=model1.provider,
                        temperature=model1.temperature,
                        reasoning_effort=model1.reasoning_effort,
                    )
                    agent2 = LLMAgent(
                        agent_id="player_2",
                        persona_type="default",
                        model=model2.model,
                        provider=model2.provider,
                        temperature=model2.temperature,
                        reasoning_effort=model2.reasoning_effort,
                    )
                    
                    # Create arena
                    arena = NegotiationArena(
                        pot_size=self.config.pot_size,
                        agent_ids=("player_1", "player_2"),
                        max_rounds=self.config.max_rounds,
                        shrinking_pot=False,
                        random_first_player=True,
                        reservation_value=self.config.reservation_value,
                    )
                    
                    # Run match with rate limiting
                    result = self._run_single_match_rate_limited(
                        agent1, agent2, arena, match_id, 
                        {model1.name: model1, model2.name: model2},
                        api_call_times,
                        verbose
                    )
                    result["model1"] = model1.name
                    result["model2"] = model2.name
                    result["match_id"] = match_id
                    result["pair_match_num"] = i + 1
                    results.append(result)
                    
                except Exception as e:
                    print(f"[{match_id}] ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    results.append({
                        "model1": model1.name,
                        "model2": model2.name,
                        "match_id": match_id,
                        "error": str(e),
                    })
        
        return results
    
    def _check_rate_limit(self, model_name: str, model_config: ModelConfig, 
                          api_call_times: Dict[str, List[float]]) -> float:
        """Check if we need to wait for rate limit. Returns seconds to wait."""
        import time as time_module
        
        now = time_module.time()
        window = 60.0  # 1 minute window
        
        # Clean old calls
        api_call_times[model_name] = [t for t in api_call_times[model_name] if now - t < window]
        
        # Check if at limit
        if len(api_call_times[model_name]) >= model_config.rpm:
            oldest = min(api_call_times[model_name])
            wait_time = window - (now - oldest) + 0.1
            return max(0, wait_time)
        
        return 0
    
    def _record_api_call(self, model_name: str, api_call_times: Dict[str, List[float]]):
        """Record an API call for rate limiting."""
        import time as time_module
        api_call_times[model_name].append(time_module.time())
    
    def _run_single_match_rate_limited(
        self,
        agent1,
        agent2,
        arena,
        match_id: str,
        model_configs: Dict[str, ModelConfig],
        api_call_times: Dict[str, List[float]],
        verbose: bool = True
    ) -> Dict:
        """Run a single negotiation match with rate limiting."""
        import time as time_module
        
        state = arena.reset()
        
        # Map agents to their model configs
        agent_models = {
            arena.agent_A: (agent1, list(model_configs.values())[0]),
            arena.agent_B: (agent2, list(model_configs.values())[1]),
        }
        agents = {arena.agent_A: agent1, arena.agent_B: agent2}
        rewards = {arena.agent_A: 0.0, arena.agent_B: 0.0}
        
        done = False
        turn = 0
        max_turns = arena.max_rounds * 2 + 5
        
        while not done and turn < max_turns:
            current_agent, current_model = agent_models[arena.current_player_id]
            obs = arena.get_observation(arena.current_player_id)
            
            try:
                # Check rate limit
                wait_time = self._check_rate_limit(
                    current_model.name, current_model, api_call_times
                )
                if wait_time > 0:
                    if verbose:
                        print(f"  ⏳ Rate limit: waiting {wait_time:.1f}s for {current_model.name}")
                    time_module.sleep(wait_time)
                
                # Make API call
                response = current_agent.choose_action(obs)
                self._record_api_call(current_model.name, api_call_times)
                
                state, rewards, done = arena.step({arena.current_player_id: response})
                
                if verbose and len(arena.conversation_history) > 0:
                    last_msg = arena.conversation_history[-1]
                    print(f"  [R{last_msg.round}] {last_msg.player_id}: {last_msg.message[:60]}...")
                    
            except Exception as e:
                if verbose:
                    print(f"  ERROR: {e}")
                break
            
            turn += 1
        
        metrics = arena.get_metrics()
        
        if verbose:
            print(f"  Outcome: {metrics['outcome']}")
            p1 = rewards.get(arena.agent_A, 0)
            p2 = rewards.get(arena.agent_B, 0)
            u = metrics.get('utilities', {})
            print(f"  Payoffs: ${p1:.0f}M/${p2:.0f}M | Utilities: {u.get('player_1', 0):.0f}/{u.get('player_2', 0):.0f}")
        
        return {
            "payoffs": rewards,
            "metrics": metrics,
            "transcript": arena.get_conversation_transcript(),
            "rounds": arena.current_round,
            "api_calls": {
                "player_1": agent1.api_call_history,
                "player_2": agent2.api_call_history,
            }
        }
    
    def save_results(self, results: List[Dict], suffix: str = ""):
        """Save tournament results to JSON."""
        output_dir = Path(__file__).parent / self.config.output_dir
        output_dir.mkdir(exist_ok=True)
        
        filename = f"h2h_tournament_{self.timestamp}{suffix}.json"
        output_path = output_dir / filename
        
        # Make results JSON serializable
        def serialize(obj):
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)
        
        data = {
            "timestamp": self.timestamp,
            "config": asdict(self.config),
            "models": {name: m.to_dict() for name, m in self.models.items()},
            "total_matches": self.total_matches,
            "matches": results,
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=serialize)
        
        print(f"\nResults saved to: {output_path}")
        return output_path
    
    def print_summary(self, results: List[Dict]):
        """Print tournament summary statistics."""
        print(f"\n{'='*70}")
        print("TOURNAMENT SUMMARY")
        print(f"{'='*70}")
        
        # Calculate stats per model
        model_stats = {name: {"wins": 0, "losses": 0, "draws": 0, "total_utility": 0, "matches": 0}
                      for name in self.models}
        
        for r in results:
            if "error" in r:
                continue
            
            m1, m2 = r["model1"], r["model2"]
            p1 = r["payoffs"].get("player_1", 0)
            p2 = r["payoffs"].get("player_2", 0)
            u = r["metrics"].get("utilities", {})
            u1 = u.get("player_1", 0)
            u2 = u.get("player_2", 0)
            
            model_stats[m1]["matches"] += 1
            model_stats[m2]["matches"] += 1
            model_stats[m1]["total_utility"] += u1
            model_stats[m2]["total_utility"] += u2
            
            if u1 > u2:
                model_stats[m1]["wins"] += 1
                model_stats[m2]["losses"] += 1
            elif u2 > u1:
                model_stats[m2]["wins"] += 1
                model_stats[m1]["losses"] += 1
            else:
                model_stats[m1]["draws"] += 1
                model_stats[m2]["draws"] += 1
        
        print(f"\n{'Model':<20} {'W-L-D':<12} {'Win%':<8} {'Avg Utility':<12}")
        print("-" * 52)
        
        for name, stats in sorted(model_stats.items(), key=lambda x: -x[1]["wins"]):
            wld = f"{stats['wins']}-{stats['losses']}-{stats['draws']}"
            win_pct = stats['wins'] / max(stats['matches'], 1) * 100
            avg_u = stats['total_utility'] / max(stats['matches'], 1)
            print(f"{name:<20} {wld:<12} {win_pct:>5.1f}%   {avg_u:>8.1f}")
        
        # Head-to-head matrix
        print(f"\n{'='*70}")
        print("HEAD-TO-HEAD RESULTS")
        print(f"{'='*70}")
        
        for model1, model2 in self.pairings:
            pair_results = [r for r in results 
                          if r.get("model1") == model1.name and r.get("model2") == model2.name
                          and "error" not in r]
            
            if pair_results:
                m1_wins = sum(1 for r in pair_results 
                            if r["metrics"]["utilities"].get("player_1", 0) > r["metrics"]["utilities"].get("player_2", 0))
                m2_wins = sum(1 for r in pair_results 
                            if r["metrics"]["utilities"].get("player_2", 0) > r["metrics"]["utilities"].get("player_1", 0))
                draws = len(pair_results) - m1_wins - m2_wins
                
                print(f"\n{model1.name} vs {model2.name}: {len(pair_results)} matches")
                print(f"  {model1.name}: {m1_wins} wins")
                print(f"  {model2.name}: {m2_wins} wins")
                print(f"  Draws: {draws}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Run a head-to-head tournament."""
    print("\n" + "="*70)
    print("HEAD-TO-HEAD LLM NEGOTIATION TOURNAMENT")
    print("="*70)
    
    # Check API keys
    required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]
    missing = [k for k in required_keys if not os.getenv(k)]
    if missing:
        print(f"ERROR: Missing API keys: {missing}")
        sys.exit(1)
    print("✓ All API keys present")
    
    # Configure tournament
    models = [
        TOURNAMENT_MODELS["claude-opus-4.5"],
        TOURNAMENT_MODELS["gemini-3-pro"],
        TOURNAMENT_MODELS["gpt-5.1-high"],
    ]
    
    config = TournamentConfig(
        matches_per_pair=100,  # Each pair plays 100 times
        use_batch_api=False,  # Start with real-time, batch support TBD
    )
    
    tournament = HeadToHeadTournament(models, config)
    tournament.print_estimates()
    
    # Confirm
    print("\n" + "="*70)
    response = input("Proceed with tournament? [y/N]: ").strip().lower()
    if response != 'y':
        print("Tournament cancelled.")
        sys.exit(0)
    
    # Run tournament
    print("\n" + "="*70)
    print("STARTING TOURNAMENT")
    print("="*70)
    
    results = tournament.run_real_time(verbose=True)
    
    # Save and summarize
    tournament.save_results(results)
    tournament.print_summary(results)


if __name__ == "__main__":
    main()
