#!/usr/bin/env python3
"""Download CaSiNo semantic analysis batch results.

Downloads results and preserves personality metadata (SVO, agreeableness) for stratification.
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
    
    # Load metadata
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
                    
                    # Add metadata (including SVO, agreeableness for stratification)
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
                        help='Batch ID to download')
    args = parser.parse_args()
    
    print("=" * 70)
    print("DOWNLOADING CASINO SEMANTIC ANALYSIS RESULTS")
    print("=" * 70)
    
    state_file = RESULTS_DIR / "semantic_batch_casino_state.json"
    
    if not state_file.exists():
        print(f"Error: State file not found: {state_file}")
        print("Make sure you have submitted a CaSiNo batch first.")
        return
    
    print(f"\n--- CaSiNo Human Negotiations ---")
    results, errors = download_batch_results(args.batch_id, state_file)
    print(f"  Downloaded: {len(results)} results, {len(errors)} errors")
    
    if errors:
        print(f"  First 3 errors: {errors[:3]}")
    
    # Save results
    output = {
        "metadata": {
            "dataset": "casino",
            "model": "claude-opus-4-5-20251101",
            "mode": "batch",
            "total_entries": len(results),
            "errors": len(errors),
            "timestamp": datetime.now().isoformat(),
            "note": "CaSiNo prompts included non-zero-sum game context. Claude prompts did not have equivalent game-theoretic context as those batches were already completed."
        },
        "results": results,
        "errors": errors
    }
    
    output_file = RESULTS_DIR / "semantic_analysis_casino_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")
    
    # Print summary with stratification
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if results:
        all_scores = [r.get("sycophancy_score", 5.0) for r in results]
        avg_score = sum(all_scores) / len(all_scores)
        print(f"\nOverall ({len(results)} entries):")
        print(f"  Mean sycophancy score: {avg_score:.2f}")
        
        # Stratify by SVO
        print("\n--- Stratified by SVO ---")
        svo_groups = {}
        for r in results:
            svo = r.get("_svo", "unknown")
            if svo not in svo_groups:
                svo_groups[svo] = []
            svo_groups[svo].append(r.get("sycophancy_score", 5.0))
        
        for svo, scores in sorted(svo_groups.items()):
            avg = sum(scores) / len(scores)
            print(f"  {svo}: n={len(scores)}, mean={avg:.2f}")
        
        # Stratify by Agreeableness buckets
        print("\n--- Stratified by Agreeableness ---")
        agree_groups = {"low (1-3.5)": [], "medium (3.5-5.5)": [], "high (5.5-7)": []}
        for r in results:
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
                print(f"  {bucket}: n={len(scores)}, mean={avg:.2f}")
    
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

