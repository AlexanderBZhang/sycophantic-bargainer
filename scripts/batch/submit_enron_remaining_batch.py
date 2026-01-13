#!/usr/bin/env python3
"""Submit batch for 580 additional emails (excluding the 20 already processed)."""

import json
import random
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from run_semantic_analysis import (
    load_enron_data, 
    build_enron_batch_requests, 
    submit_batch, 
    save_batch_state, 
    BatchJobState,
    sanitize_custom_id,
    RESULTS_DIR
)

def main():
    # Load existing batch state to get the 20 already-processed email IDs
    batch_state_file = RESULTS_DIR / "semantic_batch_enron_state.json"
    
    if batch_state_file.exists():
        with open(batch_state_file) as f:
            batch_state = json.load(f)
        
        processed_email_ids = set()
        for entry in batch_state.get("entries_metadata", []):
            processed_email_ids.add(entry.get("email_id", ""))
        
        print(f"Already processed: {len(processed_email_ids)} emails")
    else:
        processed_email_ids = set()
        print("No existing batch state found")
    
    # Load all Enron emails
    all_emails = load_enron_data()
    print(f"Total available emails: {len(all_emails)}")
    
    # Filter out already processed
    remaining = [e for e in all_emails if e["email_id"] not in processed_email_ids]
    print(f"Remaining (not yet processed): {len(remaining)}")
    
    # Sample 580 new emails
    random.seed(43)  # Different seed to get different emails
    sample_580 = random.sample(remaining, min(580, len(remaining)))
    print(f"Sampled {len(sample_580)} new emails")
    
    # Build and submit batch
    print(f"\n{'='*60}")
    print(f"SUBMITTING BATCH: {len(sample_580)} new emails")
    print("=" * 60)
    
    requests, id_mapping = build_enron_batch_requests(sample_580)
    batch_id = submit_batch(requests)
    
    # Save state with metadata
    entries_metadata = []
    for i, e in enumerate(sample_580):
        sanitized_id = sanitize_custom_id(f"{i}_{e['email_id']}")
        entries_metadata.append({
            "sanitized_id": sanitized_id,
            "email_id": e["email_id"],
            "content": e["content"],
            "from_email": e.get("from_email", ""),
            "from_name": e.get("from_name", ""),
            "to_emails": e.get("to_emails", []),
            "subject": e.get("subject", ""),
            "_from_title_info": e.get("_from_title_info", {}),
            "_to_title_infos": e.get("_to_title_infos", []),
        })
    
    # Save to a NEW state file (don't overwrite the original 20)
    new_state_file = RESULTS_DIR / "semantic_batch_enron_580_state.json"
    
    state = BatchJobState(
        batch_id=batch_id,
        dataset="enron",
        status="submitted",
        entries_metadata=entries_metadata,
        created_at=datetime.now().isoformat()
    )
    save_batch_state(state, new_state_file)
    
    print(f"\nBatch submitted! Check status with:")
    print(f"  python -c \"from run_semantic_analysis import check_batch_status; check_batch_status('{batch_id}')\"")
    print(f"\nState saved to: {new_state_file}")


if __name__ == "__main__":
    main()

