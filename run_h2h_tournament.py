#!/usr/bin/env python3
"""
Head-to-Head Tournament Runner

Runs a systematic tournament where each model plays every other model N times.
Uses Batch APIs for 50% cost reduction and to bypass rate limits.

Usage:
    python run_h2h_tournament.py [--matches-per-pair N] [--real-time] [--dry-run]

Default: 100 matches per pair using batch APIs (300 total matches)
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent / ".env")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_tournament import (
    HeadToHeadTournament,
    TournamentConfig,
    ModelConfig,
    TOURNAMENT_MODELS,
)


def check_api_keys():
    """Check that required API keys are available."""
    required_keys = {
        "OPENAI_API_KEY": "OpenAI (GPT-5.1)",
        "ANTHROPIC_API_KEY": "Anthropic (Claude Opus 4.5)",
        "GOOGLE_API_KEY": "Google (Gemini 3 Pro)",
    }
    
    missing_keys = []
    for key_name, provider in required_keys.items():
        if not os.environ.get(key_name):
            missing_keys.append(f"{provider} ({key_name})")
    
    if missing_keys:
        print("ERROR: Missing required API keys:")
        for key in missing_keys:
            print(f"  - {key}")
        print("\nPlease set the required environment variables before running.")
        return False
    
    print("✓ All required API keys are present")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run a head-to-head LLM negotiation tournament"
    )
    parser.add_argument(
        "--matches-per-pair", "-n",
        type=int,
        default=100,
        help="Number of matches per model pairing (default: 100)"
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Use real-time API instead of batch API (not recommended for Gemini 3 Pro)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show cost estimates without running the tournament"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(TOURNAMENT_MODELS.keys()),
        default=list(TOURNAMENT_MODELS.keys()),
        help="Models to include in tournament"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt (for automated runs)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("HEAD-TO-HEAD LLM NEGOTIATION TOURNAMENT")
    print("="*70)
    print("\nThis tournament tests SOTA models in bilateral negotiation:")
    print("  • Claude Opus 4.5 (Anthropic)")
    print("  • Gemini 3 Pro (Google)")
    print("  • GPT 5.1 High (OpenAI)")
    print("\nGame: Negotiation Arena")
    print("  • $100M joint venture to divide")
    print("  • Natural language dialogue")
    print("  • Max 10 rounds per match")
    print("  • $10M salvage if no deal")
    print("  • AI sees sigmoid utility; arena scores with linear formula")
    print(f"\nFormat: Head-to-Head ({args.matches_per_pair} matches per pair)")
    print("="*70)
    
    # Check API keys
    if not check_api_keys():
        sys.exit(1)
    
    # Configure tournament
    models = [TOURNAMENT_MODELS[m] for m in args.models]
    
    config = TournamentConfig(
        matches_per_pair=args.matches_per_pair,
        use_batch_api=not args.real_time,
    )
    
    tournament = HeadToHeadTournament(models, config)
    
    # Print estimates
    tournament.print_estimates()
    
    if args.dry_run:
        print("\n[DRY RUN] Tournament not executed.")
        sys.exit(0)
    
    # Warnings
    if args.real_time:
        print("\n" + "="*70)
        print("⚠️  WARNING: Real-time mode selected")
        print("="*70)
        print("Gemini 3 Pro Preview has only ~2 RPM rate limit.")
        print("This will take ~5 HOURS for 100 matches per pair!")
        print("Consider using --batch (default) instead.")
        print("="*70)
    
    # Confirm execution
    if not args.yes:
    print("\n" + "="*70)
    response = input("Proceed with tournament? [y/N]: ").strip().lower()
    if response != 'y':
        print("Tournament cancelled.")
        sys.exit(0)
    else:
        print("\n[--yes flag: Skipping confirmation]")
    
    # Run tournament
    print("\n" + "="*70)
    print("STARTING TOURNAMENT")
    print("="*70)
    
    if args.real_time:
        results = tournament.run_real_time(verbose=True)
    else:
        # Batch mode - not yet fully implemented
        print("\n⚠️  Batch API mode requires multi-turn conversation support.")
        print("Running in real-time mode with rate limiting...")
        results = tournament.run_real_time(verbose=True)
    
    # Save and summarize
    tournament.save_results(results)
    tournament.print_summary(results)


if __name__ == "__main__":
    main()
