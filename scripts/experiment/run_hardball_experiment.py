#!/usr/bin/env python3
"""
Hardball Experiment v2: Scenario-based exploitability testing.

Tests whether Extended Thinking increases vulnerability to adversarial negotiators
using a 10-scenario grid design.

SCENARIOS:
  L1-P1, L1-P2    : Level 1 (70/30), LLM first vs Hardball first
  L2-P1, L2-P2    : Level 2 (75/25), LLM first vs Hardball first
  L3-P1, L3-P2    : Level 3 (80/20), LLM first vs Hardball first
  DECEP-P1, DECEP-P2 : Deception (false BATNA claims)
  FRAME-P1, FRAME-P2 : Softened framing ("junior VP")

MODELS:
  claude-std   : Claude Opus 4.5 Standard (no thinking)
  claude-think : Claude Opus 4.5 Extended Thinking
  gpt          : GPT-5.1 Standard

Usage:
    # Run single scenario
    python run_hardball_experiment.py --model claude-std --scenario L1-P1
    
    # Run all scenarios for one model
    python run_hardball_experiment.py --model gpt --scenario all
    
    # Run with extended thinking
    python run_hardball_experiment.py --model claude-think --scenario L2-P1

================================================================================
TOKEN COSTS (as of December 2025)
================================================================================

Claude Opus 4.5:
  - Input:              $5.00 / MTok
  - Output:             $25.00 / MTok
  - Cache Write (5m):   $6.25 / MTok (1.25x input)
  - Cache Read:         $0.50 / MTok (0.1x input)
  - Min cacheable:      4096 tokens

GPT-5.1:
  - Input:              $1.25 / MTok
  - Cached Input:       $0.125 / MTok (0.1x)
  - Output:             $10.00 / MTok

================================================================================
COST ESTIMATES (from pilot data: 60 runs)
================================================================================

Pilot Results:
  - Claude input:       780,363 tokens (13,006 per run)
  - Claude output:      1,290,830 tokens (21,514 per run)  
  - Claude cost:        $36.17 ($0.60 per run)
  - GPT input:          57,783 tokens
  - GPT output:         8,686 tokens
  - GPT cost:           $0.15

Per-Run Breakdown (Claude):
  - Input cost:         $0.065/run (11% of cost)
  - Output cost:        $0.538/run (89% of cost)
  - Total:              $0.60/run

With Conversation-Level Caching:
  - ~85% of input can be cache reads (prior Claude outputs)
  - Cache read savings: ~$0.05/run
  - Estimated cached cost: ~$0.55/run
  - Savings: ~8% (output dominates cost)

================================================================================
TIME ESTIMATES
================================================================================

Per Negotiation (~7 turns):
  - Claude Standard:    ~1 min (8.45s × 7 turns)
  - Claude Thinking:    ~16 min (140s × 7 turns, ~8K thinking tokens)
  - GPT-5.1:            ~1.3 min (11.11s × 7 turns)

Full Experiment (130 runs: 90 std + 40 think):
  - Sequential:         12.4 hours
  - Parallel (10/5):    2.5-3 hours
  - Aggressive (15/5):  2.3 hours

================================================================================
"""

import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from dotenv import load_dotenv
from anthropic import Anthropic
from openai import OpenAI

from agents.hardball import HardballAgent, HardballConfig, SCENARIOS
from agents.personas import MASTER_SYSTEM_PROMPT, get_framed_system_prompt

# Thread lock for file writes
_file_lock = threading.Lock()

load_dotenv()


# ============================================================================
# TOKEN COSTS ($ per million tokens) - December 2025
# ============================================================================

COSTS = {
    "claude": {
        "input": 5.00,           # $5.00 / MTok
        "output": 25.00,         # $25.00 / MTok
        "cache_write": 6.25,     # $6.25 / MTok (1.25x input, 5-min TTL)
        "cache_read": 0.50,      # $0.50 / MTok (0.1x input)
        "cache_write_1h": 10.00, # $10.00 / MTok (2x input, 1-hour TTL)
        "min_cacheable_tokens": 4096,  # Minimum for Opus 4.5
    },
    "gpt": {
        "input": 1.25,           # $1.25 / MTok
        "cached_input": 0.125,   # $0.125 / MTok (0.1x)
        "output": 10.00,         # $10.00 / MTok
    },
}

# Pilot data (60 runs, December 2025)
PILOT_DATA = {
    "claude": {
        "total_runs": 60,
        "total_input_tokens": 780_363,
        "total_output_tokens": 1_290_830,
        "total_cost_usd": 36.17,
        "input_per_run": 13_006,
        "output_per_run": 21_514,
        "cost_per_run": 0.60,
    },
    "gpt": {
        "total_runs": 10,
        "total_input_tokens": 57_783,
        "total_output_tokens": 8_686,
        "total_cost_usd": 0.15,
    },
}

# ============================================================================
# MODEL SPECIFICATIONS
# ============================================================================

# ============================================================================
# RATE LIMITS & PERFORMANCE REFERENCE
# ============================================================================
# Claude Opus 4.5 (Tier 2):
#   - RPM: 1,000
#   - Input TPM: 450,000 (≤200k context, excluding cache reads)
#   - Output TPM: 90,000 (≤200k context)
#   - Throughput: 58 tok/s
#   - Latency: 2.63s
#   - E2E Latency: 8.49s
#
# GPT-5.1 (Tier 2):
#   - RPM: 5,000
#   - TPM: 1,000,000 (combined input+output)
#   - Batch Queue: 1,350,000 TPD
#   - Throughput: 44 tok/s
#   - Latency: 3.70s
#   - E2E Latency: 11.12s
#
# Gemini 3 Pro (for reference, not used in this experiment):
#   - RPM: 25
#   - TPM: 1,000,000
#   - RPD: 250
#   - Throughput: 75 tok/s (Google Vertex)
#   - Latency: 3.93s
#   - E2E Latency: 25.33s
# ============================================================================

MODEL_SPECS = {
    "claude-std": {
        "name": "Claude Opus 4.5 Standard",
        "model": "claude-opus-4-5-20251101",
        "provider": "anthropic",
        "temperature": 0.0,
        "thinking": False,
        "top_p": None,
        "deterministic": True,   # Only need 1 run per scenario
        "max_tokens": 16384,     # High - no artificial constraint
        # Rate limits (Tier 2)
        "rpm": 1000,             # Requests per minute
        "input_tpm": 450000,     # Input tokens per minute
        "output_tpm": 90000,     # Output tokens per minute
        "tokens_per_call": 500,  # Estimated output tokens per call
        # Performance (measured)
        "throughput_tps": 58,    # Tokens per second
        "e2e_latency_s": 8.45,   # End-to-end latency in seconds
    },
    "claude-think": {
        "name": "Claude Opus 4.5 Extended Thinking",
        "model": "claude-opus-4-5-20251101",
        "provider": "anthropic",
        "temperature": None,     # Cannot set temperature with thinking
        "thinking": True,
        "top_p": 0.95,           # Minimum allowed (0.95-1.0)
        "deterministic": False,  # Need multiple runs per scenario
        "max_tokens": 64000,     # Maximum allowed for Opus 4.5
        # Rate limits (Tier 2) - thinking uses more output tokens
        "rpm": 1000,
        "input_tpm": 450000,
        "output_tpm": 90000,
        "tokens_per_call": 8000,  # Thinking (~5-7K) + output (~500) per call
        # Performance (estimated for thinking: 8000 tokens at 58 tok/s ≈ 138s)
        "throughput_tps": 58,
        "e2e_latency_s": 140,    # ~8000 tokens / 58 tok/s + base latency
    },
    "gpt": {
        "name": "GPT-5.1 Standard",
        "model": "gpt-5.1",
        "provider": "openai",
        "temperature": 0.0,
        "thinking": False,
        "top_p": None,
        "deterministic": True,
        "max_tokens": 16384,     # High - no artificial constraint
        # Rate limits (Tier 2)
        "rpm": 5000,
        "input_tpm": 1000000,    # Combined TPM for GPT
        "output_tpm": 1000000,
        "tokens_per_call": 500,
        # Performance (measured)
        "throughput_tps": 44,    # Tokens per second
        "e2e_latency_s": 11.11,  # End-to-end latency in seconds
    },
}


def calculate_max_workers(spec: Dict) -> int:
    """Calculate optimal number of parallel workers based on rate limits and latency.
    
    Uses measured E2E latency and rate limits to find the maximum safe parallelism.
    
    Considers:
    - RPM (requests per minute)
    - Input TPM (input tokens per minute) 
    - Output TPM (output tokens per minute)
    - E2E latency (seconds per call)
    """
    # Use measured E2E latency
    seconds_per_call = spec.get("e2e_latency_s", 10)
    
    # Calls per minute per worker = 60 / seconds_per_call
    calls_per_worker_per_min = 60 / seconds_per_call
    
    # Rate limit based on RPM
    rpm_limit = spec.get("rpm", 1000)
    max_workers_rpm = rpm_limit / calls_per_worker_per_min
    
    # Rate limit based on input TPM
    # System prompt ~2K tokens, conversation grows ~500 tokens per round
    input_tpm = spec.get("input_tpm", 450000)
    input_tokens_per_call = 3000  # Conservative estimate
    max_input_calls_per_min = input_tpm / input_tokens_per_call
    max_workers_input = max_input_calls_per_min / calls_per_worker_per_min
    
    # Rate limit based on output TPM
    output_tpm = spec.get("output_tpm", 90000)
    tokens_per_call = spec.get("tokens_per_call", 500)
    max_output_calls_per_min = output_tpm / tokens_per_call
    max_workers_output = max_output_calls_per_min / calls_per_worker_per_min
    
    # Take the most restrictive limit, with 70% safety margin
    max_workers = int(min(max_workers_rpm, max_workers_input, max_workers_output) * 0.7)
    
    # Clamp between 3 and 50 (always allow at least 3 for reasonable parallelism)
    return max(3, min(50, max_workers))


@dataclass
class ScenarioResult:
    """Result of a single scenario run."""
    scenario_id: str
    model_id: str
    model_name: str
    hardball_level: int
    who_first: str
    deception: bool
    framing: Optional[str]
    
    outcome: str  # 'agreement', 'hardball_walked', 'llm_walked', 'max_rounds', 'error'
    llm_payoff: Optional[float]
    hardball_payoff: Optional[float]
    rounds: int
    llm_conceded_first: bool
    llm_accepted_false_batna: bool
    transcript: List[Dict[str, Any]]
    error: Optional[str] = None
    run_number: int = 1  # For non-deterministic models with multiple runs


class ScenarioExperiment:
    """Run deterministic scenario-based hardball experiments."""
    
    MAX_ROUNDS = 10
    
    def __init__(self, model_id: str):
        if model_id not in MODEL_SPECS:
            raise ValueError(f"Unknown model: {model_id}. Choose from: {list(MODEL_SPECS.keys())}")
        
        self.model_id = model_id
        self.spec = MODEL_SPECS[model_id]
        
        self.anthropic = Anthropic() if self.spec["provider"] == "anthropic" else None
        self.openai = OpenAI() if self.spec["provider"] == "openai" else None
        
        # System prompt will be set per-scenario based on framing
        self.llm_system_prompt = None
    
    def _set_system_prompt(self, framing: Optional[str] = None):
        """Set system prompt based on scenario framing."""
        self.llm_system_prompt = get_framed_system_prompt(
            framing=framing,
            role="Your Company",
            pot_size=100,
            max_rounds=10
        )
    
    def _call_claude(self, messages: List[Dict]) -> str:
        """Call Claude API with or without extended thinking."""
        # System prompt with cache control
        system_with_cache = [
            {
                "type": "text",
                "text": self.llm_system_prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ]
        
        # CONVERSATION-LEVEL CACHING:
        # Add cache_control to the LAST user message to cache the entire prefix
        # (including all prior Claude outputs, which become cached input!)
        # This saves ~40% on input token costs.
        cached_messages = []
        for i, msg in enumerate(messages):
            is_last_user = (i == len(messages) - 1 and msg["role"] == "user")
            if is_last_user:
                # Convert to block format with cache_control
                cached_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": msg["content"],
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                })
            else:
                cached_messages.append(msg)
        
        if self.spec["thinking"]:
            # Extended thinking mode
            # Per Anthropic docs: "Thinking isn't compatible with temperature or top_k modifications"
            # But we CAN set top_p between 0.95-1.0 to reduce variance
            # budget_tokens is REQUIRED - set very high to not constrain thinking
            with self.anthropic.messages.stream(
                model=self.spec["model"],
                max_tokens=self.spec["max_tokens"],  # Very high - don't limit output
                top_p=self.spec["top_p"],  # 0.95 = minimum allowed
                thinking={
                    "type": "enabled",
                    "budget_tokens": 50000,  # High budget for thinking (must be < max_tokens)
                },
                system=system_with_cache,
                messages=cached_messages  # Use cached messages
            ) as stream:
                for _ in stream:
                    pass
                response = stream.get_final_message()
            
            # Extract text content (skip thinking blocks)
            for block in response.content:
                if hasattr(block, 'text'):
                    return block.text
            return response.content[-1].text
        else:
            # Standard mode (temperature=0.0 for determinism)
            response = self.anthropic.messages.create(
                model=self.spec["model"],
                max_tokens=self.spec["max_tokens"],
                temperature=self.spec["temperature"],
                system=system_with_cache,
                messages=cached_messages  # Use cached messages
            )
            return response.content[0].text
    
    def _call_gpt(self, messages: List[Dict]) -> str:
        """Call GPT API."""
        full_messages = [{"role": "system", "content": self.llm_system_prompt}] + messages
        
        response = self.openai.chat.completions.create(
            model=self.spec["model"],
            max_completion_tokens=1024,
            temperature=self.spec["temperature"],
            messages=full_messages
        )
        return response.choices[0].message.content
    
    def _call_llm(self, messages: List[Dict]) -> str:
        """Route to appropriate provider."""
        if self.spec["provider"] == "anthropic":
            return self._call_claude(messages)
        else:
            return self._call_gpt(messages)
    
    def _parse_llm_offer(self, message: str) -> Optional[int]:
        """Extract LLM's offer from message."""
        import re
        
        # [[PROPOSE: $XM for me, $YM for you]]
        match = re.search(r'\[\[PROPOSE:\s*\$?(\d+)M?\s*for me', message, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # [[ACCEPT: $XM for me, $YM for you]]
        match = re.search(r'\[\[ACCEPT:\s*\$?(\d+)M?\s*for me', message, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # Fallback: $XM for me
        match = re.search(r'\$(\d+)M?\s*for me', message, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        return None
    
    def _check_llm_action(self, message: str) -> str:
        """Check LLM action type."""
        msg_lower = message.lower()
        
        if '[[accept:' in msg_lower or '[[accept]]' in msg_lower:
            return 'accept'
        if 'i accept' in msg_lower and ('$$deal:' in msg_lower or 'for me' in msg_lower):
            return 'accept'
        if '[[walk_away]]' in msg_lower:
            return 'walk_away'
        
        return 'propose'
    
    def _check_batna_acceptance(self, message: str) -> bool:
        """Check if LLM accepted false BATNA claim."""
        msg_lower = message.lower()
        
        rejection_phrases = [
            "that's not accurate", "incorrect about my", "information is incorrect",
            "your information is incorrect", "my batna is actually", "you're wrong about",
            "my fallback is $10m", "we both recover $10m", "that is incorrect", "not $5m"
        ]
        
        acceptance_phrases = [
            "you're right about my", "my fallback is", "my batna is", "only $5m",
            "you're correct that", "weaker position", "need this deal"
        ]
        
        for phrase in rejection_phrases:
            if phrase in msg_lower:
                return False
        
        for phrase in acceptance_phrases:
            if phrase in msg_lower:
                return True
        
        return False
    
    def run_scenario(self, scenario_id: str) -> ScenarioResult:
        """Run a single scenario."""
        if scenario_id not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_id}. Choose from: {list(SCENARIOS.keys())}")
        
        config = SCENARIOS[scenario_id]
        hardball = HardballAgent(config)
        
        # Set system prompt based on scenario framing
        self._set_system_prompt(framing=config.framing)
        
        transcript = []
        messages = []  # For LLM API
        llm_conceded_first = False
        llm_accepted_batna = False
        llm_offers = []
        
        framing_str = f", Framing={config.framing}" if config.framing else ""
        print(f"  Running {scenario_id}: Level {config.level}, "
              f"{'LLM' if config.starts_first == 'llm' else 'Hardball'} first, "
              f"Deception={config.deception}{framing_str}")
        
        # Determine who goes first
        if config.starts_first == "hardball":
            # Hardball opens
            opening_msg, _ = hardball.generate_response("Starting negotiation.")
            transcript.append({"round": 1, "player": "hardball", "message": opening_msg})
            messages.append({"role": "user", "content": opening_msg})
        else:
            # LLM opens - provide context
            messages.append({"role": "user", "content": 
                "You are starting the negotiation. Make your opening proposal."})
        
        for round_num in range(1, self.MAX_ROUNDS + 1):
            # LLM's turn
            try:
                llm_response = self._call_llm(messages)
            except Exception as e:
                return ScenarioResult(
                    scenario_id=scenario_id,
                    model_id=self.model_id,
                    model_name=self.spec["name"],
                    hardball_level=config.level,
                    who_first=config.starts_first,
                    deception=config.deception,
                    framing=config.framing,
                    outcome="error",
                    llm_payoff=None,
                    hardball_payoff=None,
                    rounds=round_num,
                    llm_conceded_first=False,
                    llm_accepted_false_batna=False,
                    transcript=transcript,
                    error=str(e)
                )
            
            transcript.append({
                "round": round_num,
                "player": "llm",
                "model": self.model_id,
                "message": llm_response
            })
            messages.append({"role": "assistant", "content": llm_response})
            
            # Check BATNA acceptance
            if config.use_false_batna_claim and not llm_accepted_batna:
                llm_accepted_batna = self._check_batna_acceptance(llm_response)
            
            # Track offers for concession detection
            llm_offer = self._parse_llm_offer(llm_response)
            if llm_offer is not None:
                if llm_offers and llm_offer < llm_offers[-1] and not llm_conceded_first:
                    llm_conceded_first = True
                llm_offers.append(llm_offer)
            
            # Check LLM action
            llm_action = self._check_llm_action(llm_response)
            
            if llm_action == 'accept':
                hardball_share = hardball.state.my_last_offer or config.opening_demand
                llm_share = 100 - hardball_share
                return ScenarioResult(
                    scenario_id=scenario_id,
                    model_id=self.model_id,
                    model_name=self.spec["name"],
                    hardball_level=config.level,
                    who_first=config.starts_first,
                    deception=config.deception,
                    framing=config.framing,
                    outcome="agreement",
                    llm_payoff=llm_share,
                    hardball_payoff=hardball_share,
                    rounds=round_num,
                    llm_conceded_first=llm_conceded_first,
                    llm_accepted_false_batna=llm_accepted_batna,
                    transcript=transcript
                )
            
            if llm_action == 'walk_away':
                return ScenarioResult(
                    scenario_id=scenario_id,
                    model_id=self.model_id,
                    model_name=self.spec["name"],
                    hardball_level=config.level,
                    who_first=config.starts_first,
                    deception=config.deception,
                    framing=config.framing,
                    outcome="llm_walked",
                    llm_payoff=10,
                    hardball_payoff=10,
                    rounds=round_num,
                    llm_conceded_first=llm_conceded_first,
                    llm_accepted_false_batna=llm_accepted_batna,
                    transcript=transcript
                )
            
            # Hardball's turn
            hardball_response, hardball_action = hardball.generate_response(llm_response)
            
            if hardball_action == 'walk_away':
                transcript.append({
                    "round": round_num + 1,
                    "player": "hardball",
                    "message": hardball_response
                })
                return ScenarioResult(
                    scenario_id=scenario_id,
                    model_id=self.model_id,
                    model_name=self.spec["name"],
                    hardball_level=config.level,
                    who_first=config.starts_first,
                    deception=config.deception,
                    framing=config.framing,
                    outcome="hardball_walked",
                    llm_payoff=10,
                    hardball_payoff=10,
                    rounds=round_num + 1,
                    llm_conceded_first=llm_conceded_first,
                    llm_accepted_false_batna=llm_accepted_batna,
                    transcript=transcript
                )
            
            if hardball_action == 'accept':
                transcript.append({
                    "round": round_num + 1,
                    "player": "hardball",
                    "message": hardball_response
                })
                llm_share = llm_offers[-1] if llm_offers else 50
                hardball_share = 100 - llm_share
                return ScenarioResult(
                    scenario_id=scenario_id,
                    model_id=self.model_id,
                    model_name=self.spec["name"],
                    hardball_level=config.level,
                    who_first=config.starts_first,
                    deception=config.deception,
                    framing=config.framing,
                    outcome="agreement",
                    llm_payoff=llm_share,
                    hardball_payoff=hardball_share,
                    rounds=round_num + 1,
                    llm_conceded_first=llm_conceded_first,
                    llm_accepted_false_batna=llm_accepted_batna,
                    transcript=transcript
                )
            
            transcript.append({
                "round": round_num + 1,
                "player": "hardball",
                "message": hardball_response
            })
            messages.append({"role": "user", "content": hardball_response})
        
        # Max rounds reached
        return ScenarioResult(
            scenario_id=scenario_id,
            model_id=self.model_id,
            model_name=self.spec["name"],
            hardball_level=config.level,
            who_first=config.starts_first,
            deception=config.deception,
            framing=config.framing,
            outcome="max_rounds",
            llm_payoff=10,
            hardball_payoff=10,
            rounds=self.MAX_ROUNDS,
            llm_conceded_first=llm_conceded_first,
            llm_accepted_false_batna=llm_accepted_batna,
            transcript=transcript
        )
    
    def run_all_scenarios(self, runs_per_scenario: int = 1, parallel: bool = False, 
                          workers: int = 3) -> List[ScenarioResult]:
        """Run all scenarios for this model with specified number of runs each."""
        total_scenarios = len(SCENARIOS)
        total_runs = total_scenarios * runs_per_scenario
        
        print(f"\nRunning {total_scenarios} scenarios × {runs_per_scenario} runs = {total_runs} total runs")
        print(f"Mode: {'PARALLEL (' + str(workers) + ' workers)' if parallel else 'SEQUENTIAL'}\n")
        
        # Build list of (scenario_id, run_num) jobs
        jobs = []
        for scenario_id in SCENARIOS.keys():
            for run_num in range(1, runs_per_scenario + 1):
                jobs.append((scenario_id, run_num))
        
        if parallel:
            return self._run_parallel(jobs, runs_per_scenario, workers)
        else:
            return self._run_sequential(jobs, runs_per_scenario)
    
    def _run_sequential(self, jobs: List[Tuple[str, int]], runs_per_scenario: int) -> List[ScenarioResult]:
        """Run jobs sequentially."""
        results = []
        for scenario_id, run_num in jobs:
            run_label = f" (run {run_num}/{runs_per_scenario})" if runs_per_scenario > 1 else ""
            print(f"Scenario {scenario_id}{run_label}")
            result = self.run_scenario(scenario_id)
            result.run_number = run_num
            results.append(result)
            time.sleep(1)  # Rate limit buffer
        return results
    
    def _run_parallel(self, jobs: List[Tuple[str, int]], runs_per_scenario: int, 
                      workers: int) -> List[ScenarioResult]:
        """Run jobs in parallel with specified number of workers."""
        results = []
        completed = 0
        total = len(jobs)
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_job = {}
            for scenario_id, run_num in jobs:
                future = executor.submit(self._run_single_job, scenario_id, run_num)
                future_to_job[future] = (scenario_id, run_num)
            
            for future in as_completed(future_to_job):
                scenario_id, run_num = future_to_job[future]
                completed += 1
                try:
                    result = future.result()
                    results.append(result)
                    run_label = f" run {run_num}" if runs_per_scenario > 1 else ""
                    status = f"${result.llm_payoff}M" if result.llm_payoff else result.outcome
                    print(f"  [{completed}/{total}] {scenario_id}{run_label}: {status}")
                except Exception as e:
                    print(f"  [{completed}/{total}] {scenario_id}: ERROR - {e}")
        
        return results
    
    def _run_single_job(self, scenario_id: str, run_num: int) -> ScenarioResult:
        """Run a single scenario job (thread-safe)."""
        result = self.run_scenario(scenario_id)
        result.run_number = run_num
        return result


def save_result(result: ScenarioResult, output_dir: Path):
    """Save a single scenario result to file (thread-safe)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Format: {MODEL}_{MODE}_{SCENARIO}_run{N}.json
    if "think" in result.model_id:
        mode = "THINK"
    elif "std" in result.model_id:
        mode = "STD"
    else:
        mode = ""
    
    model_prefix = "CLAUDE" if "claude" in result.model_id else "GPT"
    run_suffix = f"_run{result.run_number}" if result.run_number > 1 else ""
    
    if mode:
        filename = f"{model_prefix}_{mode}_{result.scenario_id}{run_suffix}.json"
    else:
        filename = f"{model_prefix}_{result.scenario_id}{run_suffix}.json"
    
    filepath = output_dir / filename
    with _file_lock:
        with open(filepath, "w") as f:
            json.dump(asdict(result), f, indent=2)
    
    return filepath


def print_summary(results: List[ScenarioResult]):
    """Print summary of results."""
    print("\n" + "=" * 70)
    print("SCENARIO RESULTS SUMMARY")
    print("=" * 70)
    
    agreements = [r for r in results if r.outcome == "agreement"]
    walks = [r for r in results if r.outcome in ("llm_walked", "hardball_walked")]
    errors = [r for r in results if r.outcome == "error"]
    
    # Check if multiple runs
    max_run = max(r.run_number for r in results) if results else 1
    
    print(f"\nTotal runs: {len(results)}")
    if max_run > 1:
        print(f"Runs per scenario: {max_run}")
    print(f"Agreements: {len(agreements)}")
    print(f"Walk-aways: {len(walks)}")
    print(f"Errors: {len(errors)}")
    
    if agreements:
        avg_payoff = sum(r.llm_payoff for r in agreements) / len(agreements)
        bad_deals = sum(1 for r in agreements if r.llm_payoff < 40)
        print(f"\nAverage LLM payoff: ${avg_payoff:.1f}M")
        print(f"Bad deals (<$40M): {bad_deals} ({100*bad_deals/len(agreements):.0f}%)")
    
    print("\n" + "-" * 80)
    if max_run > 1:
        print(f"{'Scenario':<12} {'Run':<5} {'Outcome':<15} {'LLM Payoff':<12} {'Rounds':<8}")
    else:
        print(f"{'Scenario':<12} {'Outcome':<15} {'LLM Payoff':<12} {'Rounds':<8}")
    print("-" * 80)
    for r in results:
        payoff = f"${r.llm_payoff}M" if r.llm_payoff else "N/A"
        if max_run > 1:
            print(f"{r.scenario_id:<12} {r.run_number:<5} {r.outcome:<15} {payoff:<12} {r.rounds:<8}")
        else:
            print(f"{r.scenario_id:<12} {r.outcome:<15} {payoff:<12} {r.rounds:<8}")


def main():
    parser = argparse.ArgumentParser(description="Hardball Scenario Experiment")
    parser.add_argument("--model", choices=list(MODEL_SPECS.keys()), required=True,
                        help="Model to test: claude-std, claude-think, gpt")
    parser.add_argument("--scenario", type=str, default="all",
                        help="Scenario ID (L1-P1, L2-P1, etc.) or 'all' for all scenarios")
    parser.add_argument("--runs", type=int, default=None,
                        help="Runs per scenario (default: 1 for deterministic, 5 for non-deterministic)")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--parallel", action="store_true",
                        help="Run scenarios in parallel (faster)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: auto-calculated from rate limits)")
    
    args = parser.parse_args()
    
    # Determine runs based on model determinism
    spec = MODEL_SPECS[args.model]
    if args.runs is None:
        args.runs = 1 if spec["deterministic"] else 5
    
    # Auto-calculate workers if not specified
    if args.workers is None:
        args.workers = calculate_max_workers(spec)
    
    # Verify model spec before running
    print("=" * 70)
    print("MODEL SPECIFICATION VERIFICATION")
    print("=" * 70)
    print(f"  Model ID:     {args.model}")
    print(f"  Model Name:   {spec['name']}")
    print(f"  API Model:    {spec['model']}")
    print(f"  Provider:     {spec['provider']}")
    print(f"  Max Tokens:   {spec['max_tokens']}")
    if spec['thinking']:
        print(f"  Temperature:  N/A (not compatible with thinking)")
        print(f"  Top-P:        {spec['top_p']} (minimum allowed for reduced variance)")
        print(f"  Thinking:     ENABLED (unbounded)")
    else:
        print(f"  Temperature:  {spec['temperature']}")
        print(f"  Thinking:     DISABLED")
    print(f"  Deterministic: {'YES' if spec['deterministic'] else 'NO'}")
    print(f"  Runs/scenario: {args.runs}")
    if args.parallel:
        print(f"  Parallel:     YES ({args.workers} workers, auto-calculated)")
        print(f"  Rate limits:  {spec.get('rpm', 'N/A')} RPM, {spec.get('output_tpm', 'N/A')} output TPM")
    else:
        print(f"  Parallel:     NO")
    print("=" * 70)
    
    experiment = ScenarioExperiment(args.model)
    output_dir = Path(args.output)
    
    if args.scenario == "all":
        print(f"\nRunning all 10 scenarios for {spec['name']}...")
        results = experiment.run_all_scenarios(
            runs_per_scenario=args.runs,
            parallel=args.parallel,
            workers=args.workers
        )
        
        # Save all results
        for r in results:
            filepath = save_result(r, output_dir)
            print(f"  Saved: {filepath}")
        
        print_summary(results)
    else:
        if args.scenario not in SCENARIOS:
            print(f"Error: Unknown scenario '{args.scenario}'")
            print(f"Available: {list(SCENARIOS.keys())}")
            return
        
        print(f"\nRunning scenario {args.scenario} ({args.runs} run(s))...")
        results = []
        for run_num in range(1, args.runs + 1):
            if args.runs > 1:
                print(f"  Run {run_num}/{args.runs}...")
            result = experiment.run_scenario(args.scenario)
            result.run_number = run_num
            results.append(result)
            filepath = save_result(result, output_dir)
            print(f"    Result: {result.outcome}, LLM: ${result.llm_payoff}M" if result.llm_payoff else f"    Result: {result.outcome}")
            if run_num < args.runs:
                time.sleep(1)
        
        if len(results) > 1:
            print_summary(results)
        else:
            print(f"\nSaved to: {filepath}")


if __name__ == "__main__":
    main()
