#!/usr/bin/env python3
"""Download 580 batch results and merge with existing 20 to create 600 total."""

import json
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from run_semantic_analysis import (
    check_batch_status,
    download_batch_results,
    calculate_statistics,
    print_report,
    save_batch_state,
    BatchJobState,
    RESULTS_DIR
)

def main():
    # Load the 580 batch state
    batch_580_state_file = RESULTS_DIR / "semantic_batch_enron_580_state.json"
    
    if not batch_580_state_file.exists():
        print(f"Batch state not found: {batch_580_state_file}")
        return
    
    with open(batch_580_state_file) as f:
        batch_state = json.load(f)
    
    batch_id = batch_state["batch_id"]
    print(f"Batch ID: {batch_id}")
    
    # Check status
    status = check_batch_status(batch_id)
    
    if status["status"] != "ended":
        print(f"\nBatch not complete yet. Status: {status['status']}")
        print("Run this script again when the batch is complete.")
        return
    
    # Download 580 results
    entries_metadata = [BatchJobState.from_dict(batch_state).entries_metadata 
                        if hasattr(BatchJobState.from_dict(batch_state), 'entries_metadata')
                        else batch_state.get("entries_metadata", [])]
    
    # Fix: entries_metadata should be the list directly
    entries_metadata = batch_state.get("entries_metadata", [])
    
    results_580, errors_580 = download_batch_results(batch_id, entries_metadata)
    print(f"\nDownloaded {len(results_580)} results from 580 batch, {len(errors_580)} errors")
    
    # Load existing 20 results
    existing_results_file = RESULTS_DIR / "semantic_analysis_enron_results.json"
    if existing_results_file.exists():
        with open(existing_results_file) as f:
            existing_data = json.load(f)
        results_20 = existing_data.get("results", [])
        print(f"Loaded {len(results_20)} existing results")
    else:
        results_20 = []
        print("No existing results found")
    
    # Merge results
    all_results = results_20 + results_580
    print(f"\nTotal merged results: {len(all_results)}")
    
    # Calculate combined statistics
    stats = calculate_statistics(all_results, "enron")
    
    # Save merged results
    output = {
        "metadata": {
            "dataset": "enron",
            "model": "claude-opus-4-5-20251101",
            "mode": "batch",
            "total_entries": len(all_results),
            "batch_1_entries": len(results_20),
            "batch_2_entries": len(results_580),
            "timestamp": datetime.now().isoformat()
        },
        "statistics": stats,
        "results": all_results,
        "errors": errors_580
    }
    
    output_file = RESULTS_DIR / "semantic_analysis_enron_600_merged.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nMerged results saved to: {output_file}")
    
    # Print report
    print_report(stats, "enron")
    
    # Update batch state
    batch_state["status"] = "completed"
    batch_state["completed_at"] = datetime.now().isoformat()
    with open(batch_580_state_file, 'w') as f:
        json.dump(batch_state, f, indent=2)


if __name__ == "__main__":
    main()

