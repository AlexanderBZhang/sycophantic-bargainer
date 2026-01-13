#!/usr/bin/env python3
"""Download Claude semantic analysis batch results."""

import json
import re
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic

from run_semantic_analysis import (
    RESULTS_DIR,
    calculate_statistics,
    print_report,
)

def download_batch_results(batch_id: str, state_file: Path) -> tuple:
    """Download batch results."""
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
                    
                    # Add metadata
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
    print("=" * 70)
    print("DOWNLOADING CLAUDE SEMANTIC ANALYSIS RESULTS")
    print("=" * 70)
    
    # Download Claude Thinking results
    thinking_state_file = RESULTS_DIR / "semantic_batch_claude_thinking_state.json"
    thinking_batch_id = "msgbatch_01Y3ND4bsntuvuYxwM6GmCTk"
    
    print(f"\n--- Claude Extended Thinking ---")
    thinking_results, thinking_errors = download_batch_results(thinking_batch_id, thinking_state_file)
    print(f"  Downloaded: {len(thinking_results)} results, {len(thinking_errors)} errors")
    
    if thinking_errors:
        print(f"  Errors: {thinking_errors[:3]}...")  # Show first 3 errors
    
    # Download Claude Standard results
    standard_state_file = RESULTS_DIR / "semantic_batch_claude_standard_state.json"
    standard_batch_id = "msgbatch_01RxFVkLQ2Rc8qH4d6TB2PB5"
    
    print(f"\n--- Claude Standard ---")
    standard_results, standard_errors = download_batch_results(standard_batch_id, standard_state_file)
    print(f"  Downloaded: {len(standard_results)} results, {len(standard_errors)} errors")
    
    # Save thinking results
    thinking_output = {
        "metadata": {
            "dataset": "claude_thinking",
            "model": "claude-opus-4-5-20251101",
            "mode": "batch",
            "total_entries": len(thinking_results),
            "errors": len(thinking_errors),
            "timestamp": datetime.now().isoformat()
        },
        "results": thinking_results,
        "errors": thinking_errors
    }
    
    thinking_output_file = RESULTS_DIR / "semantic_analysis_claude_thinking_results.json"
    with open(thinking_output_file, 'w') as f:
        json.dump(thinking_output, f, indent=2)
    print(f"\nThinking results saved to: {thinking_output_file}")
    
    # Save standard results
    standard_output = {
        "metadata": {
            "dataset": "claude_standard",
            "model": "claude-opus-4-5-20251101",
            "mode": "batch",
            "total_entries": len(standard_results),
            "errors": len(standard_errors),
            "timestamp": datetime.now().isoformat()
        },
        "results": standard_results,
        "errors": standard_errors
    }
    
    standard_output_file = RESULTS_DIR / "semantic_analysis_claude_standard_results.json"
    with open(standard_output_file, 'w') as f:
        json.dump(standard_output, f, indent=2)
    print(f"Standard results saved to: {standard_output_file}")
    
    # Print summaries
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Thinking summary
    print(f"\nClaude Extended Thinking ({len(thinking_results)} entries):")
    if thinking_results:
        # Extract scores from thinking results
        thinking_scores = []
        output_scores = []
        for r in thinking_results:
            if "thinking" in r:
                thinking_scores.append(r["thinking"].get("sycophancy_score", 5.0))
            if "output" in r:
                output_scores.append(r["output"].get("sycophancy_score", 5.0))
        
        if thinking_scores:
            avg_thinking = sum(thinking_scores) / len(thinking_scores)
            print(f"  Thinking sycophancy: mean={avg_thinking:.2f}")
        if output_scores:
            avg_output = sum(output_scores) / len(output_scores)
            print(f"  Output sycophancy: mean={avg_output:.2f}")
        if thinking_scores and output_scores:
            gap = avg_output - avg_thinking
            print(f"  Alignment gap (output - thinking): {gap:+.2f}")
    
    # Standard summary
    print(f"\nClaude Standard ({len(standard_results)} entries):")
    if standard_results:
        output_scores = []
        for r in standard_results:
            if "output" in r:
                output_scores.append(r["output"].get("sycophancy_score", 5.0))
        
        if output_scores:
            avg_output = sum(output_scores) / len(output_scores)
            print(f"  Output sycophancy: mean={avg_output:.2f}")
    
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

