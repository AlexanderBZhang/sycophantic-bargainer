#!/usr/bin/env python3
"""Submit remaining CaSiNo entries (excluding the 50 already analyzed).

This script:
1. Loads the 50 already-analyzed entries from the first batch
2. Loads all 2060 preprocessed entries
3. Filters out the 50 already done
4. Submits the remaining 2010 as a new batch
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import random

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

from submit_casino_batch import CASINO_ANALYSIS_PROMPT


def get_already_analyzed_ids():
    """Get IDs of entries already analyzed in the first batch."""
    results_file = RESULTS_DIR / "semantic_analysis_casino_results.json"
    
    if not results_file.exists():
        return set()
    
    with open(results_file, encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract original IDs from the results
    already_done = set()
    for r in data.get('results', []):
        original_id = r.get('_original_id')
        if original_id:
            already_done.add(original_id)
    
    return already_done


def get_first_batch_sampled_ids():
    """Recreate the exact same random sample from the first batch."""
    # Load all entries
    with open(RESULTS_DIR / 'CASINO_PREPROCESSED.jsonl', encoding='utf-8') as f:
        all_entries = [json.loads(line) for line in f]
    
    # Use the same seed as the first batch
    random.seed(42)
    sampled = random.sample(all_entries, 50)
    
    return {e['id'] for e in sampled}


def main():
    print("=" * 70)
    print("SUBMITTING REMAINING CASINO ENTRIES")
    print("=" * 70)
    
    # Get IDs from first batch (using same random seed)
    first_batch_ids = get_first_batch_sampled_ids()
    print(f"\nFirst batch contained {len(first_batch_ids)} entries")
    
    # Load all entries
    print("\nLoading all preprocessed entries...")
    with open(RESULTS_DIR / 'CASINO_PREPROCESSED.jsonl', encoding='utf-8') as f:
        all_entries = [json.loads(line) for line in f]
    print(f"Total entries: {len(all_entries)}")
    
    # Filter out already-done entries
    remaining_entries = [e for e in all_entries if e['id'] not in first_batch_ids]
    print(f"Remaining to analyze: {len(remaining_entries)}")
    
    # Build requests
    print("\nBuilding batch requests...")
    requests = []
    metadata = []
    
    for i, entry in enumerate(remaining_entries):
        messages = entry.get('messages_combined', '')
        full_dialogue = entry.get('full_dialogue', '')
        entry_id = entry.get('id', f'casino_{i}')
        
        prompt = CASINO_ANALYSIS_PROMPT.format(
            rubric=SYCOPHANCY_RUBRIC,
            messages=messages,
            full_dialogue=full_dialogue
        )
        
        # Use different prefix to avoid ID collision
        sanitized_id = sanitize_custom_id(f"cas2_{i}_{entry_id}")
        
        request = Request(
            custom_id=sanitized_id,
            params=MessageCreateParamsNonStreaming(
                model=MODEL_ID,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
        )
        requests.append(request)
        
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
    
    print(f"Built {len(requests)} requests")
    
    # Show SVO distribution
    svo_counts = {}
    for m in metadata:
        svo = m.get('svo', 'unknown')
        svo_counts[svo] = svo_counts.get(svo, 0) + 1
    print(f"\nSVO distribution:")
    for svo, count in sorted(svo_counts.items(), key=lambda x: -x[1]):
        print(f"  {svo}: {count}")
    
    # Estimate cost
    total_input_chars = sum(len(r['params']['messages'][0]['content']) for r in requests)
    input_tokens = total_input_chars / 4
    output_tokens = len(requests) * 500
    input_cost = (input_tokens / 1_000_000) * 2.50
    output_cost = (output_tokens / 1_000_000) * 12.50
    total_cost = input_cost + output_cost
    
    print(f"\nCost estimate:")
    print(f"  Input tokens: ~{input_tokens:,.0f}")
    print(f"  Output tokens: ~{output_tokens:,.0f}")
    print(f"  TOTAL: ${total_cost:.2f}")
    
    # Submit batch
    print("\n" + "=" * 70)
    client = Anthropic()
    print(f"Submitting batch with {len(requests)} requests...")
    batch = client.messages.batches.create(requests=requests)
    
    print(f"  Batch created: {batch.id}")
    print(f"  Status: {batch.processing_status}")
    
    # Save state
    state = BatchJobState(
        batch_id=batch.id,
        dataset="casino_remaining",
        status="submitted",
        entries_metadata=metadata,
        created_at=datetime.now().isoformat()
    )
    state_file = RESULTS_DIR / "semantic_batch_casino_remaining_state.json"
    save_batch_state(state, state_file)
    
    print(f"\nState saved to: {state_file}")
    print("\n" + "=" * 70)
    print("BATCH SUBMITTED!")
    print("=" * 70)
    print(f"\nBatch ID: {batch.id}")
    print(f"\nAfter completion, run:")
    print(f"  python download_and_merge_casino.py --batch-id {batch.id}")


if __name__ == "__main__":
    main()

