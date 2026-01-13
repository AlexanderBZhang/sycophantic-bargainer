"""Analyze CaSiNo batch results and verify what AI received."""
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load results
with open('results/semantic_analysis_casino_results.json', encoding='utf-8') as f:
    data = json.load(f)

results = data['results']
print(f"Total results: {len(results)}")
print(f"Errors: {len(data.get('errors', []))}")

# Check a sample result
print("\n" + "=" * 70)
print("SAMPLE RESULT (Entry 0)")
print("=" * 70)

sample = results[0]
print(f"\nEntry ID: {sample.get('_entry_id')}")
print(f"SVO: {sample.get('_svo')}")
print(f"Agreeableness: {sample.get('_agreeableness')}")
print(f"Word count: {sample.get('_word_count')}")

print(f"\nSycophancy score: {sample.get('sycophancy_score')}")
print(f"Context: {sample.get('context')}")
print(f"Explanation: {sample.get('explanation')}")

print(f"\nCategories:")
cats = sample.get('categories', {})
for k, v in cats.items():
    print(f"  {k}: {v}")

print(f"\nKey indicators: {sample.get('key_indicators', [])[:3]}...")

# Load preprocessed data to see what was sent
print("\n" + "=" * 70)
print("WHAT WAS SENT TO AI (from preprocessed data)")
print("=" * 70)

with open('results/CASINO_PREPROCESSED.jsonl', encoding='utf-8') as f:
    lines = f.readlines()

# The batch used random.seed(42) to sample 50 entries
import random
random.seed(42)
all_entries = [json.loads(line) for line in lines]
sampled = random.sample(all_entries, 50)

first_entry = sampled[0]
print(f"\nEntry ID: {first_entry['id']}")
print(f"Dialogue ID: {first_entry['dialogue_id']}")
print(f"Participant: {first_entry['participant_id']}")

print(f"\n--- PARTICIPANT MESSAGES (what AI analyzed) ---")
print(first_entry['messages_combined'][:500])
if len(first_entry['messages_combined']) > 500:
    print("...")

print(f"\n--- FULL DIALOGUE (provided for context) ---")
print(first_entry['full_dialogue'][:800])
if len(first_entry['full_dialogue']) > 800:
    print("...")

# Verify modality - CaSiNo is typed chat
print("\n" + "=" * 70)
print("MODALITY VERIFICATION")
print("=" * 70)
print("\nCaSiNo is a TYPED CHAT corpus:")
print("- Participants communicated via text chat interface")
print("- Messages are written/typed (not spoken)")
print("- Each turn is a chat message")

# Check for any problematic results
print("\n" + "=" * 70)
print("QUALITY CHECK")
print("=" * 70)

missing_score = [r for r in results if 'sycophancy_score' not in r]
missing_cats = [r for r in results if 'categories' not in r]
missing_context = [r for r in results if 'context' not in r]

print(f"Missing sycophancy_score: {len(missing_score)}")
print(f"Missing categories: {len(missing_cats)}")
print(f"Missing context: {len(missing_context)}")

# Score distribution
scores = [r.get('sycophancy_score', 5.0) for r in results]
print(f"\nScore distribution:")
print(f"  Min: {min(scores):.2f}")
print(f"  Max: {max(scores):.2f}")
print(f"  Mean: {sum(scores)/len(scores):.2f}")

# Context distribution
contexts = {}
for r in results:
    ctx = r.get('context', 'unknown')
    contexts[ctx] = contexts.get(ctx, 0) + 1
print(f"\nContext distribution:")
for ctx, count in sorted(contexts.items(), key=lambda x: -x[1]):
    print(f"  {ctx}: {count}")

