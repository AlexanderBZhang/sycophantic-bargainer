#!/usr/bin/env python3
"""Estimate cost for semantic analysis of Claude negotiations using token counts."""

import json
from pathlib import Path

# Batch API pricing for Claude Opus 4.5 (from LLM Pricing.txt)
BATCH_INPUT_COST = 2.50   # $2.50 per MTok (50% off standard)
BATCH_OUTPUT_COST = 12.50  # $12.50 per MTok (50% off standard)

# Approximate tokens per character (rough estimate for English)
CHARS_PER_TOKEN = 4

def estimate_tokens(text: str) -> int:
    """Estimate token count from text."""
    return len(text) // CHARS_PER_TOKEN

def analyze_thinking_tokens():
    """Analyze Claude Extended Thinking data."""
    with open('results/THINKING_TOKENS_COMBINED.json') as f:
        data = json.load(f)
    
    entries = data.get('entries', [])
    
    total_thinking_tokens = 0
    total_output_tokens = 0
    
    for entry in entries:
        thinking = entry.get('thinking', '')
        output = entry.get('output', '')
        
        total_thinking_tokens += estimate_tokens(thinking)
        total_output_tokens += estimate_tokens(output)
    
    return {
        'name': 'Claude Extended Thinking',
        'entries': len(entries),
        'thinking_tokens': total_thinking_tokens,
        'output_tokens': total_output_tokens,
        'total_content_tokens': total_thinking_tokens + total_output_tokens,
    }

def analyze_standard_negotiations():
    """Analyze Claude Standard negotiations."""
    total_tokens = 0
    total_entries = 0
    negotiations = 0
    
    with open('results/HARDBALL_STANDARD_90.jsonl') as f:
        for line in f:
            data = json.loads(line)
            negotiations += 1
            transcript = data.get('transcript', [])
            
            for turn in transcript:
                speaker = turn.get('speaker', '')
                message = turn.get('message', '')
                
                # Count LLM turns (not Hardball agent)
                if 'hardball' not in speaker.lower():
                    total_tokens += estimate_tokens(message)
                    total_entries += 1
    
    return {
        'name': 'Claude Standard',
        'negotiations': negotiations,
        'llm_turns': total_entries,
        'total_content_tokens': total_tokens,
    }

def calculate_analysis_cost(content_tokens: int) -> dict:
    """Calculate batch API cost for semantic analysis.
    
    For each entry, we send:
    - Rubric (~500 tokens) + prompt template (~150 tokens) + content
    - Receive: JSON response (~250 tokens)
    """
    RUBRIC_TOKENS = 500
    PROMPT_TEMPLATE_TOKENS = 150
    OUTPUT_TOKENS_PER_ENTRY = 250
    
    # For thinking analysis, we analyze 3 components (thinking, output, bracketed)
    # so input is larger
    input_tokens = content_tokens + RUBRIC_TOKENS + PROMPT_TEMPLATE_TOKENS
    output_tokens = OUTPUT_TOKENS_PER_ENTRY
    
    input_cost = (input_tokens / 1_000_000) * BATCH_INPUT_COST
    output_cost = (output_tokens / 1_000_000) * BATCH_OUTPUT_COST
    
    return {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': input_cost + output_cost,
    }

def main():
    print("=" * 70)
    print("COST ESTIMATION FOR CLAUDE SEMANTIC ANALYSIS")
    print("Using Anthropic Batch API (50% off standard pricing)")
    print("=" * 70)
    
    # Analyze thinking tokens
    thinking_stats = analyze_thinking_tokens()
    print(f"\n--- {thinking_stats['name']} ---")
    print(f"  Entries: {thinking_stats['entries']}")
    print(f"  Thinking tokens (est): {thinking_stats['thinking_tokens']:,}")
    print(f"  Output tokens (est): {thinking_stats['output_tokens']:,}")
    print(f"  Total content tokens: {thinking_stats['total_content_tokens']:,}")
    
    # Analyze standard negotiations
    standard_stats = analyze_standard_negotiations()
    print(f"\n--- {standard_stats['name']} ---")
    print(f"  Negotiations: {standard_stats['negotiations']}")
    print(f"  LLM turns: {standard_stats['llm_turns']}")
    print(f"  Total content tokens: {standard_stats['total_content_tokens']:,}")
    
    # Calculate costs
    print("\n" + "=" * 70)
    print("COST ESTIMATES (Batch API)")
    print("=" * 70)
    
    # For thinking: each entry is one LLM turn
    thinking_per_entry_tokens = thinking_stats['total_content_tokens'] // thinking_stats['entries']
    thinking_total_input = thinking_stats['entries'] * (thinking_per_entry_tokens + 650)  # rubric + template
    thinking_total_output = thinking_stats['entries'] * 300  # slightly larger JSON for 3-component analysis
    
    thinking_input_cost = (thinking_total_input / 1_000_000) * BATCH_INPUT_COST
    thinking_output_cost = (thinking_total_output / 1_000_000) * BATCH_OUTPUT_COST
    thinking_total_cost = thinking_input_cost + thinking_output_cost
    
    print(f"\n{thinking_stats['name']}:")
    print(f"  Input: {thinking_total_input:,} tokens = ${thinking_input_cost:.2f}")
    print(f"  Output: {thinking_total_output:,} tokens = ${thinking_output_cost:.2f}")
    print(f"  TOTAL: ${thinking_total_cost:.2f}")
    
    # For standard: each LLM turn is one entry
    standard_per_entry_tokens = standard_stats['total_content_tokens'] // max(standard_stats['llm_turns'], 1)
    standard_total_input = standard_stats['llm_turns'] * (standard_per_entry_tokens + 650)
    standard_total_output = standard_stats['llm_turns'] * 250
    
    standard_input_cost = (standard_total_input / 1_000_000) * BATCH_INPUT_COST
    standard_output_cost = (standard_total_output / 1_000_000) * BATCH_OUTPUT_COST
    standard_total_cost = standard_input_cost + standard_output_cost
    
    print(f"\n{standard_stats['name']}:")
    print(f"  Input: {standard_total_input:,} tokens = ${standard_input_cost:.2f}")
    print(f"  Output: {standard_total_output:,} tokens = ${standard_output_cost:.2f}")
    print(f"  TOTAL: ${standard_total_cost:.2f}")
    
    # Grand total
    print("\n" + "-" * 70)
    grand_total = thinking_total_cost + standard_total_cost
    print(f"GRAND TOTAL: ${grand_total:.2f}")
    print("-" * 70)
    
    # Comparison with Enron
    print("\nFor reference:")
    print("  Enron (600 emails): ~$3.64")
    print(f"  Claude analysis: ~${grand_total:.2f}")

if __name__ == "__main__":
    main()

