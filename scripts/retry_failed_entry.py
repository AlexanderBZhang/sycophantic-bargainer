#!/usr/bin/env python3
"""Find and retry the failed entry from Claude Thinking batch."""

import json
import re
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic

from run_semantic_analysis import (
    SYCOPHANCY_RUBRIC,
    CLAUDE_ANALYSIS_PROMPT,
    MODEL_ID,
    RESULTS_DIR,
)

def main():
    # Load thinking tokens to find entry 406
    with open(RESULTS_DIR / 'THINKING_TOKENS_COMBINED.json') as f:
        data = json.load(f)
    
    entries = data.get('entries', [])
    print(f"Total entries: {len(entries)}")
    
    # Entry 406 is the failed one
    entry_idx = 406
    entry = entries[entry_idx]
    
    print(f"\n--- Failed Entry {entry_idx} ---")
    print(f"Custom ID: {entry.get('custom_id', 'N/A')}")
    print(f"Thinking length: {len(entry.get('thinking', ''))} chars")
    print(f"Output length: {len(entry.get('output', ''))} chars")
    
    thinking = entry.get('thinking', '')
    output = entry.get('output', '')
    
    # Extract bracketed content
    bracket_match = re.search(r'\[\[(.*?)\]\]', output, re.DOTALL)
    bracketed = bracket_match.group(0) if bracket_match else "[No bracketed action found]"
    
    print(f"\nOutput preview: {output[:300]}...")
    print(f"\nBracketed: {bracketed[:100]}...")
    
    # Build prompt
    prompt = CLAUDE_ANALYSIS_PROMPT.format(
        rubric=SYCOPHANCY_RUBRIC,
        thinking=thinking,
        output=output,
        bracketed=bracketed
    ) + "\n\nRespond with valid JSON only."
    
    print(f"\n--- Running through API manually ---")
    print(f"Model: {MODEL_ID}")
    print(f"Prompt length: {len(prompt)} chars")
    
    # Call API
    client = Anthropic()
    
    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    content = response.content[0].text
    print(f"\n--- API Response ---")
    print(f"Response length: {len(content)} chars")
    print(f"Usage: input={response.usage.input_tokens}, output={response.usage.output_tokens}")
    
    # Parse JSON
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        result = json.loads(json_match.group())
        result["_entry_id"] = f"think_{entry_idx}_{entry.get('custom_id', '')}"
        result["_model"] = MODEL_ID
        result["_retried"] = True
        
        print(f"\n--- Parsed Result ---")
        print(json.dumps(result, indent=2))
        
        # Load existing results and add this one
        results_file = RESULTS_DIR / "semantic_analysis_claude_thinking_results.json"
        with open(results_file) as f:
            results_data = json.load(f)
        
        results_data["results"].append(result)
        results_data["metadata"]["total_entries"] = len(results_data["results"])
        results_data["metadata"]["errors"] = 0
        results_data["errors"] = []  # Clear the error
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n✅ Result added to {results_file}")
        print(f"Total entries now: {len(results_data['results'])}")
    else:
        print("ERROR: No JSON found in response")
        print(f"Raw response: {content}")


if __name__ == "__main__":
    main()

