#!/usr/bin/env python3
"""Download remaining CaSiNo batch and merge with existing 50 results.

This script:
1. Downloads the remaining batch results
2. Loads the existing 50 results
3. Merges them into one complete file
4. Provides stratified summary statistics
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic

from run_semantic_analysis import RESULTS_DIR


def download_batch_results(batch_id: str, state_file: Path) -> tuple:
    """Download batch results with metadata preservation."""
    client = Anthropic()
    
    with open(state_file) as f:
        state = json.load(f)
    
    metadata_lookup = {e["sanitized_id"]: e for e in state.get("entries_metadata", [])}
    
    results = []
    errors = []
    
    print(f"Downloading results for {batch_id}...")
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        
        if result.result.type == "succeeded":
            message = result.result.message
            content = message.content[0].text if message.content else ""
            
            try:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    parsed["_entry_id"] = custom_id
                    parsed["_model"] = "claude-opus-4-5-20251101"
                    
                    if custom_id in metadata_lookup:
                        meta = metadata_lookup[custom_id]
                        for key, value in meta.items():
                            if key != "sanitized_id":
                                parsed[f"_{key}"] = value
                    
                    results.append(parsed)
                else:
                    errors.append({"custom_id": custom_id, "error": "No JSON in response"})
            except json.JSONDecodeError as e:
                errors.append({"custom_id": custom_id, "error": f"JSON parse error: {e}"})
        else:
            errors.append({
                "custom_id": custom_id,
                "error": f"Request failed: {result.result.type}"
            })
    
    return results, errors


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-id', type=str, required=True,
                        help='Batch ID for remaining entries')
    args = parser.parse_args()
    
    print("=" * 70)
    print("DOWNLOADING AND MERGING CASINO RESULTS")
    print("=" * 70)
    
    # Download remaining batch
    state_file = RESULTS_DIR / "semantic_batch_casino_remaining_state.json"
    
    if not state_file.exists():
        print(f"Error: State file not found: {state_file}")
        return
    
    print(f"\n--- Downloading Remaining Batch ---")
    new_results, new_errors = download_batch_results(args.batch_id, state_file)
    print(f"  Downloaded: {len(new_results)} results, {len(new_errors)} errors")
    
    # Load existing 50 results
    existing_file = RESULTS_DIR / "semantic_analysis_casino_results.json"
    print(f"\n--- Loading Existing Results ---")
    
    with open(existing_file, encoding='utf-8') as f:
        existing_data = json.load(f)
    
    existing_results = existing_data.get('results', [])
    print(f"  Existing: {len(existing_results)} results")
    
    # Merge
    all_results = existing_results + new_results
    all_errors = existing_data.get('errors', []) + new_errors
    
    print(f"\n--- Merged ---")
    print(f"  Total results: {len(all_results)}")
    print(f"  Total errors: {len(all_errors)}")
    
    # Save merged results
    merged_output = {
        "metadata": {
            "dataset": "casino",
            "model": "claude-opus-4-5-20251101",
            "mode": "batch",
            "total_entries": len(all_results),
            "errors": len(all_errors),
            "timestamp": datetime.now().isoformat(),
            "note": "CaSiNo prompts included non-zero-sum game context. Claude prompts did not have equivalent game-theoretic context as those batches were already completed.",
            "batches_merged": [
                {"batch": "first_50", "count": len(existing_results)},
                {"batch": args.batch_id, "count": len(new_results)}
            ]
        },
        "results": all_results,
        "errors": all_errors
    }
    
    # Save to the same file (overwrite with merged data)
    with open(existing_file, 'w', encoding='utf-8') as f:
        json.dump(merged_output, f, indent=2, ensure_ascii=False)
    print(f"\nMerged results saved to: {existing_file}")
    
    # Print summary with stratification
    print("\n" + "=" * 70)
    print("MERGED SUMMARY")
    print("=" * 70)
    
    all_scores = [r.get("sycophancy_score", 5.0) for r in all_results]
    print(f"\nOverall ({len(all_results)} entries):")
    print(f"  Mean sycophancy score: {sum(all_scores)/len(all_scores):.2f}")
    print(f"  Min: {min(all_scores):.2f}, Max: {max(all_scores):.2f}")
    
    # Stratify by SVO
    print("\n--- Stratified by SVO ---")
    svo_groups = {}
    for r in all_results:
        svo = r.get("_svo", "unknown")
        if svo not in svo_groups:
            svo_groups[svo] = []
        svo_groups[svo].append(r.get("sycophancy_score", 5.0))
    
    for svo, scores in sorted(svo_groups.items()):
        avg = sum(scores) / len(scores)
        std = (sum((s - avg)**2 for s in scores) / len(scores))**0.5
        print(f"  {svo}: n={len(scores)}, mean={avg:.2f}, std={std:.2f}")
    
    # Stratify by Agreeableness buckets
    print("\n--- Stratified by Agreeableness ---")
    agree_groups = {"low (1-3.5)": [], "medium (3.5-5.5)": [], "high (5.5-7)": []}
    for r in all_results:
        agree = r.get("_agreeableness")
        score = r.get("sycophancy_score", 5.0)
        if agree is not None:
            if agree <= 3.5:
                agree_groups["low (1-3.5)"].append(score)
            elif agree <= 5.5:
                agree_groups["medium (3.5-5.5)"].append(score)
            else:
                agree_groups["high (5.5-7)"].append(score)
    
    for bucket, scores in agree_groups.items():
        if scores:
            avg = sum(scores) / len(scores)
            std = (sum((s - avg)**2 for s in scores) / len(scores))**0.5
            print(f"  {bucket}: n={len(scores)}, mean={avg:.2f}, std={std:.2f}")
    
    print("\n" + "=" * 70)
    print("MERGE COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()

