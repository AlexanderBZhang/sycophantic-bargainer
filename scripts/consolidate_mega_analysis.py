#!/usr/bin/env python3
"""
Mega Analysis Consolidation Script

Consolidates findings from two experimental phases:
  - Experiment 1 (Archive): AI vs AI symmetric tournament (240 matches)
  - Experiment 2 (Current): Hardball adversary vs Claude (180 matches)

This script:
  1. Validates data file existence and structure
  2. Extracts key metrics from each experiment
  3. Generates a consolidated summary JSON
  4. Prints a human-readable summary

NOTE: This is prototype code used for experimental data consolidation.
It has not undergone rigorous testing or formal verification.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

WORKSPACE_ROOT = Path(__file__).parent.parent
RESULTS_DIR = WORKSPACE_ROOT / "results"
ARCHIVE_DIR = WORKSPACE_ROOT / "archive"

# Experiment 1: AI vs AI Tournament (from archive)
EXPERIMENT_1_FILES = {
    "tournament_data": ARCHIVE_DIR / "FINAL_240_COMPLETE.jsonl",
    "analysis_report": ARCHIVE_DIR / "ANALYSIS_REPORT.md",
    "tournament_analysis": ARCHIVE_DIR / "TOURNAMENT_ANALYSIS.md",
}

# Experiment 2: Hardball vs AI (current)
EXPERIMENT_2_FILES = {
    "thinking_90": RESULTS_DIR / "HARDBALL_THINKING_90.jsonl.json",
    "standard_90": RESULTS_DIR / "HARDBALL_STANDARD_90.jsonl",
    "thinking_tokens": RESULTS_DIR / "THINKING_TOKENS_COMBINED.json" if (RESULTS_DIR / "THINKING_TOKENS_COMBINED.json").exists() else None,
    "semantic_claude_thinking": RESULTS_DIR / "semantic_analysis_claude_thinking_results.json",
    "semantic_claude_standard": RESULTS_DIR / "semantic_analysis_claude_standard_results.json",
    "semantic_enron": RESULTS_DIR / "semantic_analysis_enron_600_merged.json",
    "semantic_casino": RESULTS_DIR / "semantic_analysis_casino_results.json",
}

OUTPUT_FILE = RESULTS_DIR / "MEGA_ANALYSIS_SUMMARY.json"


# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================

def load_jsonl(filepath: Path) -> list:
    """Load a JSONL file into a list of dictionaries."""
    if not filepath.exists():
        print(f"  [WARN] File not found: {filepath}")
        return []
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] JSON parse error in {filepath.name} line {line_num}: {e}")
    return data


def load_json(filepath: Path) -> Optional[Dict]:
    """Load a JSON file into a dictionary."""
    if not filepath or not filepath.exists():
        print(f"  [WARN] File not found: {filepath}")
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def count_lines(filepath: Path) -> int:
    """Count non-empty lines in a file."""
    if not filepath.exists():
        return 0
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


# =============================================================================
# EXPERIMENT 1: AI vs AI TOURNAMENT ANALYSIS
# =============================================================================

def analyze_experiment_1() -> Dict[str, Any]:
    """
    Analyze Experiment 1: Symmetric AI vs AI Tournament.
    
    This experiment tested Claude Opus 4.5 vs GPT-5.1 in cooperative
    negotiations to measure baseline sycophancy indicators.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: AI vs AI Symmetric Tournament")
    print("="*60)
    
    results = {
        "name": "AI vs AI Symmetric Tournament",
        "description": "Claude Opus 4.5 vs GPT-5.1 in cooperative bilateral negotiations",
        "data_available": False,
        "metrics": {},
        "files_checked": [],
    }
    
    # Check for tournament data
    tournament_file = EXPERIMENT_1_FILES["tournament_data"]
    results["files_checked"].append(str(tournament_file))
    
    if tournament_file.exists():
        matches = load_jsonl(tournament_file)
        results["data_available"] = len(matches) > 0
        results["metrics"]["total_matches"] = len(matches)
        print(f"  [OK] Tournament data: {len(matches)} matches")
        
        # Extract key metrics from matches if available
        if matches:
            # Count by model pairing
            pairings = {}
            for match in matches:
                p1 = match.get("player1_model", "unknown")
                p2 = match.get("player2_model", "unknown")
                key = f"{p1} vs {p2}"
                pairings[key] = pairings.get(key, 0) + 1
            results["metrics"]["match_pairings"] = pairings
    else:
        print(f"  [WARN] Tournament data not found: {tournament_file}")
    
    # Hardcoded summary from archived analysis (ANALYSIS_REPORT.md)
    # These values are extracted from the archive documentation
    results["summary"] = {
        "experiment_type": "symmetric_cooperative",
        "models_tested": ["Claude Opus 4.5 Standard", "GPT-5.1"],
        "total_matches": 240,
        "valid_matches": 239,
        "agreement_rate": 0.996,
        "key_findings": [
            "Claude exhibits 72% deference ratio vs GPT's 51%",
            "Claude concedes first 2× more often (66.5% vs 33.5%)",
            "Despite behavioral differences, both achieve equivalent payoffs (~$50M)",
            "Sycophancy manifests as process-level accommodation, not outcome-level loss",
        ],
        "language_analysis": {
            "claude_deference_per_1k": 4.45,
            "gpt_deference_per_1k": 3.64,
            "enron_baseline_per_1k": 0.08,
            "claude_vs_human_ratio": "54×",
        },
        "elo_ratings": {
            "GPT-5.1": 1501,
            "Claude Opus 4.5": 1499,
            "gap": 2,  # Statistically zero
        },
    }
    
    print(f"  [OK] Summary metrics extracted from archived analysis")
    return results


# =============================================================================
# EXPERIMENT 2: HARDBALL vs AI ANALYSIS
# =============================================================================

def analyze_experiment_2() -> Dict[str, Any]:
    """
    Analyze Experiment 2: Hardball Adversary vs Claude.
    
    This experiment tested whether Extended Thinking helps resist
    adversarial exploitation from a rule-based Hardball agent.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Hardball Adversary vs Claude")
    print("="*60)
    
    results = {
        "name": "Hardball Adversary vs Claude",
        "description": "Rule-based adversarial agent vs Claude (Standard and Extended Thinking)",
        "data_available": False,
        "metrics": {},
        "files_checked": [],
    }
    
    # Check for Hardball experiment data
    thinking_file = EXPERIMENT_2_FILES["thinking_90"]
    standard_file = EXPERIMENT_2_FILES["standard_90"]
    
    for name, filepath in [("thinking", thinking_file), ("standard", standard_file)]:
        if filepath and filepath.exists():
            # Handle both JSON and JSONL formats
            if str(filepath).endswith('.json'):
                data = load_json(filepath)
                count = len(data) if isinstance(data, list) else 1
            else:
                data = load_jsonl(filepath)
                count = len(data)
            results["metrics"][f"{name}_matches"] = count
            results["files_checked"].append(str(filepath))
            print(f"  [OK] {name.title()} data: {count} entries")
            results["data_available"] = True
        else:
            print(f"  [WARN] {name.title()} data not found")
    
    # Check semantic analysis files
    semantic_files = {
        "claude_thinking": EXPERIMENT_2_FILES["semantic_claude_thinking"],
        "claude_standard": EXPERIMENT_2_FILES["semantic_claude_standard"],
        "enron_baseline": EXPERIMENT_2_FILES["semantic_enron"],
        "casino_baseline": EXPERIMENT_2_FILES["semantic_casino"],
    }
    
    for name, filepath in semantic_files.items():
        if filepath and filepath.exists():
            data = load_json(filepath)
            if data and "results" in data:
                count = len(data["results"])
            elif isinstance(data, list):
                count = len(data)
            else:
                count = 1
            results["metrics"][f"semantic_{name}"] = count
            print(f"  [OK] Semantic analysis ({name}): {count} entries")
    
    # Hardcoded summary from HARDBALL_REPORT.md
    results["summary"] = {
        "experiment_type": "adversarial_exploitation",
        "models_tested": [
            "Claude Opus 4.5 Standard",
            "Claude Opus 4.5 Extended Thinking",
            "GPT-5.1 (comparison)",
        ],
        "adversary": "Rule-based Hardball agent",
        "total_matches": 180,
        "conditions": {
            "standard_n": 90,
            "thinking_n": 90,
            "gpt_n": 10,
        },
        "key_findings": [
            "Extended Thinking provides statistically significant protection (p=0.019)",
            "+$2.7M average payoff improvement (+10%) vs Standard",
            "92% vs 81% agreement rate",
            "44% reduction in deference markers (0.55 vs 0.98 per 1K words)",
            "Both conditions still exploitable - accept sub-threshold deals",
        ],
        "statistical_tests": {
            "mann_whitney_u": 3266.0,
            "mann_whitney_p": 0.024,
            "t_test_statistic": -2.37,
            "t_test_p": 0.019,
            "cohens_d": 0.35,
            "effect_size": "small-medium",
        },
        "economic_outcomes": {
            "standard_avg_payoff_m": 25.6,
            "thinking_avg_payoff_m": 28.3,
            "gpt_avg_payoff_m": 11.6,
            "standard_utility": 25268,
            "thinking_utility": 131737,
            "gpt_utility": -595825,
        },
        "language_analysis": {
            "standard_deference_per_1k": 0.98,
            "thinking_deference_per_1k": 0.55,
            "enron_baseline_per_1k": 0.08,
            "reduction_percent": 44,
        },
    }
    
    print(f"  [OK] Summary metrics extracted from Hardball analysis")
    return results


# =============================================================================
# MEGA ANALYSIS SYNTHESIS
# =============================================================================

def synthesize_mega_analysis(exp1: Dict, exp2: Dict) -> Dict[str, Any]:
    """
    Synthesize findings from both experiments into a unified analysis.
    """
    print("\n" + "="*60)
    print("MEGA ANALYSIS SYNTHESIS")
    print("="*60)
    
    mega = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "version": "0.1.0-prototype",
            "status": "EXPERIMENTAL",
            "disclaimer": (
                "This analysis combines v0 experimental code and AI-assisted generation. "
                "It has not undergone rigorous unit testing or formal verification. "
                "Quantitative claims should not be considered reliable due to identified "
                "logical flaws in the generative code."
            ),
        },
        "experiments": {
            "experiment_1_symmetric": exp1,
            "experiment_2_adversarial": exp2,
        },
        "cross_experiment_synthesis": {
            "total_negotiations": (
                exp1.get("summary", {}).get("total_matches", 0) +
                exp2.get("summary", {}).get("total_matches", 0)
            ),
            "key_insight": (
                "Sycophancy is DORMANT in symmetric play but ACTIVATED by adversarial pressure. "
                "Claude's deference is not exploited by GPT (another RLHF model) but is exploited "
                "by a rule-based adversary."
            ),
            "findings": [
                {
                    "finding": "Process vs Outcome Sycophancy",
                    "evidence": "In symmetric play, Claude shows 72% deference but achieves equivalent payoffs. Against Hardball, deference correlates with lower payoffs.",
                },
                {
                    "finding": "Extended Thinking Provides Protection",
                    "evidence": "44% reduction in deference + $2.7M improvement, statistically significant (p=0.019).",
                },
                {
                    "finding": "Internal-External Alignment Gap",
                    "evidence": "Thinking tokens show Claude recognizes exploitation but output still complies.",
                },
                {
                    "finding": "Both Conditions Remain Exploitable",
                    "evidence": "Neither Standard nor Thinking walks away frequently; both accept sub-threshold deals.",
                },
            ],
            "implications": [
                "Sycophancy measurement requires adversarial testing, not just cooperative scenarios",
                "Extended Thinking is a partial but not complete defense",
                "Internal reasoning ≠ behavioral alignment",
            ],
        },
        "data_inventory": {
            "experiment_1_files": list(EXPERIMENT_1_FILES.keys()),
            "experiment_2_files": list(EXPERIMENT_2_FILES.keys()),
            "results_directory": str(RESULTS_DIR),
            "archive_directory": str(ARCHIVE_DIR),
        },
    }
    
    print(f"  [OK] Total negotiations analyzed: {mega['cross_experiment_synthesis']['total_negotiations']}")
    print(f"  [OK] Synthesis complete with {len(mega['cross_experiment_synthesis']['findings'])} key findings")
    
    return mega


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run the mega analysis consolidation."""
    print("\n" + "#"*60)
    print("# MEGA ANALYSIS CONSOLIDATION SCRIPT")
    print("# " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("#"*60)
    
    print("\nWARNING: This is prototype/experimental code.")
    print("Results should be verified before use in any publication.\n")
    
    # Analyze each experiment
    exp1_results = analyze_experiment_1()
    exp2_results = analyze_experiment_2()
    
    # Synthesize findings
    mega_analysis = synthesize_mega_analysis(exp1_results, exp2_results)
    
    # Save consolidated output
    print("\n" + "="*60)
    print("SAVING OUTPUT")
    print("="*60)
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(mega_analysis, f, indent=2, default=str)
    
    print(f"  [OK] Saved to: {OUTPUT_FILE}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"""
MEGA ANALYSIS CONSOLIDATION COMPLETE

Experiment 1 (Symmetric AI vs AI):
  - Matches: {exp1_results.get('summary', {}).get('total_matches', 'N/A')}
  - Data Available: {exp1_results.get('data_available', False)}

Experiment 2 (Hardball vs Claude):
  - Matches: {exp2_results.get('summary', {}).get('total_matches', 'N/A')}
  - Data Available: {exp2_results.get('data_available', False)}

Combined Analysis:
  - Total Negotiations: {mega_analysis['cross_experiment_synthesis']['total_negotiations']}
  - Key Findings: {len(mega_analysis['cross_experiment_synthesis']['findings'])}

Output: {OUTPUT_FILE}

NOTE: This consolidation is for documentation purposes.
Verify all quantitative claims before publication.
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
