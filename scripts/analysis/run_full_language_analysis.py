"""
Full Language Analysis - All Experiments

Runs canonical marker analysis on ALL experiments:
1. 240 Tournament (Claude Standard vs GPT-5.1)
2. 180 Hardball (Claude Standard vs Claude Thinking)

Uses the same canonical markers throughout for consistency.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

# Import canonical markers (single source of truth)
from analysis.canonical_markers import (
    CATEGORY_1_APOLOGIES,
    CATEGORY_2_DEFERENCE,
    CATEGORY_3_HEDGES,
    CATEGORY_4_FLATTERY,
    ENRON_BASELINE,
    count_markers,
)


def count_words(text):
    """Count words in text."""
    return len(text.split())


def analyze_messages(messages, player_filter=None):
    """Analyze messages for language markers."""
    combined_text = ""
    word_count = 0
    
    for msg in messages:
        if isinstance(msg, dict):
            # Handle different message formats
            player = msg.get("player", msg.get("role", ""))
            content = msg.get("message", msg.get("content", ""))
            
            if player_filter is None or player == player_filter:
                combined_text += content + " "
                word_count += count_words(content)
    
    apologies = count_markers(combined_text, CATEGORY_1_APOLOGIES)
    deference = count_markers(combined_text, CATEGORY_2_DEFERENCE)
    hedges = count_markers(combined_text, CATEGORY_3_HEDGES)
    flattery = count_markers(combined_text, CATEGORY_4_FLATTERY)
    
    return {
        "words": word_count,
        "apologies": apologies,
        "deference": deference,
        "hedges": hedges,
        "flattery": flattery,
        "total": apologies + deference + hedges + flattery,
    }


def load_tournament_data(filepath):
    """Load 240 tournament JSONL data."""
    matches = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                matches.append(json.loads(line))
    return matches


def load_hardball_results(results_dir):
    """Load all Hardball JSON results."""
    results = {
        "claude_standard": [],
        "claude_thinking": [],
        "gpt": [],
    }
    
    for filepath in Path(results_dir).glob("*.json"):
        fname = filepath.name
        
        # Skip aggregate files
        if "all_results" in fname:
            continue
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if "transcript" not in data:
                continue
            
            # Determine model
            if fname.startswith("GPT_"):
                results["gpt"].append(data)
            elif fname.startswith("CLAUDE_STD_") or fname.startswith("claude-std-matched_"):
                results["claude_standard"].append(data)
            elif fname.startswith("CLAUDE_THINK_") or fname.startswith("claude-think_"):
                results["claude_thinking"].append(data)
                
        except Exception as e:
            print(f"Error loading {fname}: {e}")
    
    return results


def main():
    print("=" * 80)
    print("COMPREHENSIVE LANGUAGE MARKER ANALYSIS")
    print("Using Canonical Markers from analysis/canonical_markers.py")
    print("=" * 80)
    
    # Print canonical markers for reference
    print(f"\nCanonical Marker Counts:")
    print(f"  Apologies: {len(CATEGORY_1_APOLOGIES)} phrases")
    print(f"  Deference: {len(CATEGORY_2_DEFERENCE)} phrases")
    print(f"  Hedges:    {len(CATEGORY_3_HEDGES)} phrases")
    print(f"  Flattery:  {len(CATEGORY_4_FLATTERY)} phrases")
    
    # Initialize aggregate stats
    stats = defaultdict(lambda: {
        "words": 0,
        "apologies": 0,
        "deference": 0,
        "hedges": 0,
        "flattery": 0,
        "total": 0,
        "count": 0,
    })
    
    # =========================================================================
    # PART 1: 240 TOURNAMENT (Claude Standard vs GPT-5.1)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: 240 TOURNAMENT (Claude Standard vs GPT-5.1)")
    print("=" * 80)
    
    tournament_file = Path("results/FINAL_240_TOURNAMENT.jsonl")
    if tournament_file.exists():
        matches = load_tournament_data(tournament_file)
        print(f"Loaded {len(matches)} tournament matches")
        
        for match in matches:
            # Get transcript (different field name than Hardball)
            transcript = match.get("transcript", [])
            if not transcript:
                continue
            
            # Group messages by model
            claude_messages = []
            gpt_messages = []
            
            for msg in transcript:
                model = msg.get("model", "")
                message_text = msg.get("message", "")
                
                if "claude" in model.lower():
                    claude_messages.append({"message": message_text})
                elif "gpt" in model.lower():
                    gpt_messages.append({"message": message_text})
            
            # Analyze Claude messages
            if claude_messages:
                analysis = analyze_messages(claude_messages)
                for key in ["words", "apologies", "deference", "hedges", "flattery", "total"]:
                    stats["tournament_claude"][key] += analysis[key]
                stats["tournament_claude"]["count"] += 1
            
            # Analyze GPT messages
            if gpt_messages:
                analysis = analyze_messages(gpt_messages)
                for key in ["words", "apologies", "deference", "hedges", "flattery", "total"]:
                    stats["tournament_gpt"][key] += analysis[key]
                stats["tournament_gpt"]["count"] += 1
        
        print(f"  Claude appearances: {stats['tournament_claude']['count']}")
        print(f"  GPT appearances: {stats['tournament_gpt']['count']}")
    else:
        print(f"Tournament file not found: {tournament_file}")
    
    # =========================================================================
    # PART 2: 180 HARDBALL (Claude Standard vs Claude Thinking)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: 180 HARDBALL (Claude Standard vs Claude Thinking)")
    print("=" * 80)
    
    hardball_results = load_hardball_results("results")
    
    for model_key, results in hardball_results.items():
        if not results:
            continue
            
        for data in results:
            transcript = data.get("transcript", [])
            
            # Analyze LLM messages only (not hardball agent)
            llm_messages = [
                {"message": msg.get("message", "")}
                for msg in transcript
                if msg.get("player") == "llm"
            ]
            
            if not llm_messages:
                continue
            
            analysis = analyze_messages(llm_messages)
            
            # Map to stats key
            stats_key = f"hardball_{model_key}"
            for key in ["words", "apologies", "deference", "hedges", "flattery", "total"]:
                stats[stats_key][key] += analysis[key]
            stats[stats_key]["count"] += 1
    
    print(f"  Claude Standard: {stats['hardball_claude_standard']['count']} negotiations")
    print(f"  Claude Thinking: {stats['hardball_claude_thinking']['count']} negotiations")
    print(f"  GPT: {stats['hardball_gpt']['count']} negotiations")
    
    # =========================================================================
    # RESULTS TABLE
    # =========================================================================
    print("\n" + "=" * 80)
    print("RESULTS: LANGUAGE MARKERS (per 1,000 words)")
    print("=" * 80)
    
    # Header
    print(f"\n{'Source':<25} {'N':<8} {'Words':<12} {'Apol':<8} {'Defer':<8} {'Hedge':<8} {'Flat':<8} {'TOTAL':<8}")
    print("-" * 95)
    
    # Enron baseline
    enron = ENRON_BASELINE["rates_per_1k"]
    print(f"{'Enron Emails':<25} {'156K':<8} {'46.4M':<12} {enron['apologies']:<8.3f} {enron['deference']:<8.3f} {enron['hedges']:<8.3f} {enron['flattery']:<8.3f} {ENRON_BASELINE['total_per_1k']:<8.3f}")
    print("-" * 95)
    
    # Print each model's stats
    display_order = [
        ("tournament_claude", "Tournament: Claude Std"),
        ("tournament_gpt", "Tournament: GPT-5.1"),
        ("hardball_claude_standard", "Hardball: Claude Std"),
        ("hardball_claude_thinking", "Hardball: Claude Think"),
        ("hardball_gpt", "Hardball: GPT"),
    ]
    
    for key, label in display_order:
        s = stats[key]
        if s["count"] == 0:
            continue
        
        words = max(s["words"], 1)
        
        # Calculate per 1K rates
        apol = (s["apologies"] / words) * 1000
        defer = (s["deference"] / words) * 1000
        hedge = (s["hedges"] / words) * 1000
        flat = (s["flattery"] / words) * 1000
        total = (s["total"] / words) * 1000
        
        words_display = f"{words:,}"
        print(f"{label:<25} {s['count']:<8} {words_display:<12} {apol:<8.3f} {defer:<8.3f} {hedge:<8.3f} {flat:<8.3f} {total:<8.3f}")
    
    # =========================================================================
    # COMPARISON TO ENRON
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON TO ENRON BASELINE (multiplier)")
    print("=" * 80)
    
    print(f"\n{'Source':<25} {'Apol':<10} {'Defer':<10} {'Hedge':<10} {'Flat':<10} {'TOTAL':<10}")
    print("-" * 75)
    
    for key, label in display_order:
        s = stats[key]
        if s["count"] == 0:
            continue
        
        words = max(s["words"], 1)
        
        # Calculate per 1K rates
        apol = (s["apologies"] / words) * 1000
        defer = (s["deference"] / words) * 1000
        hedge = (s["hedges"] / words) * 1000
        flat = (s["flattery"] / words) * 1000
        total = (s["total"] / words) * 1000
        
        # Calculate multipliers
        apol_mult = apol / enron["apologies"] if enron["apologies"] > 0 else 0
        defer_mult = defer / enron["deference"] if enron["deference"] > 0 else 0
        hedge_mult = hedge / enron["hedges"] if enron["hedges"] > 0 else 0
        flat_mult = flat / enron["flattery"] if enron["flattery"] > 0 else 0
        total_mult = total / ENRON_BASELINE["total_per_1k"] if ENRON_BASELINE["total_per_1k"] > 0 else 0
        
        print(f"{label:<25} {apol_mult:<10.1f}x {defer_mult:<10.1f}x {hedge_mult:<10.1f}x {flat_mult:<10.1f}x {total_mult:<10.1f}x")
    
    # =========================================================================
    # RAW COUNTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("RAW COUNTS (for verification)")
    print("=" * 80)
    
    print(f"\n{'Source':<25} {'N':<8} {'Words':<12} {'Apol':<8} {'Defer':<8} {'Hedge':<8} {'Flat':<8} {'Total':<8}")
    print("-" * 95)
    
    for key, label in display_order:
        s = stats[key]
        if s["count"] == 0:
            continue
        
        words_display = f"{s['words']:,}"
        print(f"{label:<25} {s['count']:<8} {words_display:<12} {s['apologies']:<8} {s['deference']:<8} {s['hedges']:<8} {s['flattery']:<8} {s['total']:<8}")
    
    # =========================================================================
    # METHODOLOGY
    # =========================================================================
    print("\n" + "=" * 80)
    print("METHODOLOGY")
    print("=" * 80)
    print("""
Canonical Markers (from analysis/canonical_markers.py):
  - Apologies: "sorry", "apologize", "forgive", "my mistake", etc. (11 phrases)
  - Deference: "you're right", "i understand", "good point", etc. (14 phrases)
  - Hedges: "i think", "maybe", "perhaps", "probably", etc. (14 phrases)
  - Flattery: "great idea", "brilliant", "perfect", etc. (11 phrases)

Counting Method:
  - Exact string matching (case-insensitive)
  - Same method used for Enron baseline (156,109 emails, 46.4M words)
  - Ensures comparability across all experiments

Data Sources:
  - Tournament: results/FINAL_240_TOURNAMENT.jsonl (240 matches)
  - Hardball: results/*.json (180 negotiations)
""")


if __name__ == "__main__":
    main()

