#!/usr/bin/env python3
"""Submit CaSiNo dataset to Anthropic Batch API for semantic analysis.

This script:
1. Loads preprocessed CaSiNo participant entries
2. Creates analysis prompts for each participant's negotiation messages  
3. Submits to Anthropic Batch API for 50% cost savings
4. Saves batch state for download later

For stratification, we keep personality metadata (SVO, agreeableness) attached.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from run_semantic_analysis import (
    SYCOPHANCY_RUBRIC,
    MODEL_ID,
    RESULTS_DIR,
    sanitize_custom_id,
    save_batch_state,
    BatchJobState,
)

# Prompt for CaSiNo (human negotiation - matches Enron schema for cross-comparison)
CASINO_ANALYSIS_PROMPT = """You are analyzing human negotiation transcripts for communication patterns, specifically looking at sycophantic language markers.

{rubric}

CONTEXT: This is from the CaSiNo corpus (human-human resource allocation negotiations). 
Participants negotiate over camping supplies (food, water, firewood) with DIFFERENT private 
valuations - this is NOT a zero-sum game. Trading based on differing preferences is smart 
strategy, not deference. Analyze for natural human communication patterns.

=== PARTICIPANT MESSAGES ===
{messages}

=== FULL DIALOGUE CONTEXT (for reference) ===
{full_dialogue}

Analyze ONLY the participant's messages shown above, not the full dialogue.

Return this JSON structure (same format as Enron analysis for cross-comparison):
{{
  "sycophancy_score": <float 0.0-10.0, where 5.0=balanced professional communication>,
  "key_indicators": [<list ALL relevant exact quotes/phrases that support your score - include as many as needed, can be empty if none>],
  "categories": {{
    "apology_count": <int>,
    "deference_count": <int>,
    "hedging_count": <int>,
    "flattery_count": <int>,
    "validation_count": <int>
  }},
  "context": "<negotiation|request|update|directive|social|other>",
  "explanation": "<1-2 sentence reasoning for the score>"
}}

Notes:
- Score 5.0 = normal balanced negotiation communication
- Not all politeness is sycophancy - social norms require diplomacy
- Consider that this is a structured negotiation with competing interests
- Return ONLY valid JSON.
"""


def build_casino_requests(sample_size: int = None):
    """Build batch requests for CaSiNo entries."""
    casino_path = RESULTS_DIR / 'CASINO_PREPROCESSED.jsonl'
    
    entries = []
    with open(casino_path, 'r', encoding='utf-8') as f:
        for line in f:
            entries.append(json.loads(line))
    
    print(f"Loaded {len(entries)} total CaSiNo entries")
    
    # Optionally sample
    if sample_size and sample_size < len(entries):
        import random
        random.seed(42)  # Reproducible sampling
        entries = random.sample(entries, sample_size)
        print(f"Sampled {len(entries)} entries for analysis")
    
    requests = []
    metadata = []
    
    for i, entry in enumerate(entries):
        messages = entry.get('messages_combined', '')
        full_dialogue = entry.get('full_dialogue', '')
        entry_id = entry.get('id', f'casino_{i}')
        
        prompt = CASINO_ANALYSIS_PROMPT.format(
            rubric=SYCOPHANCY_RUBRIC,
            messages=messages,
            full_dialogue=full_dialogue
        )
        
        sanitized_id = sanitize_custom_id(f"casino_{i}_{entry_id}")
        
        request = Request(
            custom_id=sanitized_id,
            params=MessageCreateParamsNonStreaming(
                model=MODEL_ID,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
        )
        requests.append(request)
        
        # Keep personality metadata for stratification
        metadata.append({
            "sanitized_id": sanitized_id,
            "original_id": entry_id,
            "dialogue_id": entry.get('dialogue_id'),
            "participant_id": entry.get('participant_id'),
            "svo": entry.get('svo'),
            "agreeableness": entry.get('agreeableness'),
            "points_scored": entry.get('points_scored'),
            "satisfaction": entry.get('satisfaction'),
            "word_count": entry.get('word_count'),
        })
    
    return requests, metadata


def estimate_cost(requests, metadata):
    """Estimate batch cost based on token counts."""
    # Approximate: 4 chars per token
    total_input_chars = 0
    for req in requests:
        prompt = req['params']['messages'][0]['content']
        total_input_chars += len(prompt)
    
    # Estimate tokens
    input_tokens = total_input_chars / 4
    output_tokens = len(requests) * 500  # ~500 tokens per response
    
    # Claude Opus 4.5 batch pricing: $2.50 input, $12.50 output (50% off regular)
    input_cost = (input_tokens / 1_000_000) * 2.50
    output_cost = (output_tokens / 1_000_000) * 12.50
    total_cost = input_cost + output_cost
    
    print(f"\nCost estimate:")
    print(f"  Input tokens: ~{input_tokens:,.0f}")
    print(f"  Output tokens: ~{output_tokens:,.0f}")
    print(f"  Input cost: ${input_cost:.2f}")
    print(f"  Output cost: ${output_cost:.2f}")
    print(f"  TOTAL: ${total_cost:.2f}")
    
    return total_cost


def submit_batch(requests, name):
    """Submit batch to Anthropic API."""
    client = Anthropic()
    
    print(f"\nSubmitting {name} batch with {len(requests)} requests...")
    batch = client.messages.batches.create(requests=requests)
    
    print(f"  Batch created: {batch.id}")
    print(f"  Status: {batch.processing_status}")
    
    return batch.id


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=None, 
                        help='Sample size (default: all 2060 entries)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Build requests but do not submit')
    args = parser.parse_args()
    
    print("=" * 70)
    print("CASINO SEMANTIC ANALYSIS BATCH SUBMISSION")
    print("=" * 70)
    
    # Build requests
    print("\n--- Building CaSiNo requests ---")
    requests, metadata = build_casino_requests(args.sample)
    print(f"  Built {len(requests)} requests")
    
    # Show SVO distribution in sample
    svo_counts = {}
    for m in metadata:
        svo = m.get('svo', 'unknown')
        svo_counts[svo] = svo_counts.get(svo, 0) + 1
    print(f"\n  SVO distribution in batch:")
    for svo, count in sorted(svo_counts.items(), key=lambda x: -x[1]):
        print(f"    {svo}: {count}")
    
    # Estimate cost
    estimate_cost(requests, metadata)
    
    if args.dry_run:
        print("\n[DRY RUN] Would submit batch, but --dry-run was specified")
        return
    
    # Submit batch
    batch_id = submit_batch(requests, "CaSiNo")
    
    # Save state
    state = BatchJobState(
        batch_id=batch_id,
        dataset="casino",
        status="submitted",
        entries_metadata=metadata,
        created_at=datetime.now().isoformat()
    )
    state_file = RESULTS_DIR / "semantic_batch_casino_state.json"
    save_batch_state(state, state_file)
    print(f"\nSaved batch state to {state_file}")
    
    print("\n" + "=" * 70)
    print("BATCH SUBMITTED!")
    print("=" * 70)
    print(f"\nCaSiNo Batch ID: {batch_id}")
    print(f"\nCheck status at: https://console.anthropic.com/workspaces/default/batches")


if __name__ == "__main__":
    main()

