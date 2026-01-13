"""Full analysis of merged CaSiNo results."""
import json
import sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

with open('results/semantic_analysis_casino_results.json', encoding='utf-8') as f:
    data = json.load(f)

results = data['results']
errors = data.get('errors', [])
metadata = data.get('metadata', {})

print("=" * 70)
print("CASINO FULL ANALYSIS")
print("=" * 70)

print(f"\nTotal results: {len(results)}")
print(f"Total errors: {len(errors)}")

if errors:
    print("\nError details:")
    for e in errors:
        print(f"  - {e}")

# Score distribution
scores = [r.get('sycophancy_score', 5.0) for r in results]
print(f"\n--- Score Distribution ---")
print(f"Mean: {np.mean(scores):.3f}")
print(f"Std:  {np.std(scores):.3f}")
print(f"Min:  {min(scores):.2f}")
print(f"Max:  {max(scores):.2f}")
print(f"Median: {np.median(scores):.2f}")

# Histogram
print(f"\nHistogram (score buckets):")
buckets = {
    "0-2 (Very Assertive)": 0,
    "2-4 (Assertive)": 0,
    "4-6 (Balanced)": 0,
    "6-8 (Deferential)": 0,
    "8-10 (Sycophantic)": 0
}
for s in scores:
    if s < 2:
        buckets["0-2 (Very Assertive)"] += 1
    elif s < 4:
        buckets["2-4 (Assertive)"] += 1
    elif s < 6:
        buckets["4-6 (Balanced)"] += 1
    elif s < 8:
        buckets["6-8 (Deferential)"] += 1
    else:
        buckets["8-10 (Sycophantic)"] += 1

for bucket, count in buckets.items():
    pct = 100 * count / len(scores)
    bar = "#" * int(pct / 2)
    print(f"  {bucket}: {count:4d} ({pct:5.1f}%) {bar}")

# Stratified analysis
print(f"\n--- Stratified by SVO ---")
svo_groups = {}
for r in results:
    svo = r.get("_svo", "unknown")
    if svo not in svo_groups:
        svo_groups[svo] = []
    svo_groups[svo].append(r.get("sycophancy_score", 5.0))

for svo in ['proself', 'prosocial', 'unclassified']:
    if svo in svo_groups:
        scores_g = svo_groups[svo]
        print(f"  {svo:15s}: n={len(scores_g):4d}, mean={np.mean(scores_g):.3f}, std={np.std(scores_g):.3f}")

# Agreeableness
print(f"\n--- Stratified by Agreeableness (Big Five) ---")
agree_groups = {"low (<=3.5)": [], "medium (3.5-5.5)": [], "high (>5.5)": []}
for r in results:
    agree = r.get("_agreeableness")
    score = r.get("sycophancy_score", 5.0)
    if agree is not None:
        if agree <= 3.5:
            agree_groups["low (<=3.5)"].append(score)
        elif agree <= 5.5:
            agree_groups["medium (3.5-5.5)"].append(score)
        else:
            agree_groups["high (>5.5)"].append(score)

for bucket in ["low (<=3.5)", "medium (3.5-5.5)", "high (>5.5)"]:
    scores_g = agree_groups[bucket]
    if scores_g:
        print(f"  {bucket:20s}: n={len(scores_g):4d}, mean={np.mean(scores_g):.3f}, std={np.std(scores_g):.3f}")

# Context distribution
print(f"\n--- Context Classification ---")
contexts = {}
for r in results:
    ctx = r.get('context', 'unknown')
    contexts[ctx] = contexts.get(ctx, 0) + 1
for ctx, count in sorted(contexts.items(), key=lambda x: -x[1]):
    print(f"  {ctx}: {count} ({100*count/len(results):.1f}%)")

# Category counts
print(f"\n--- Average Category Counts ---")
cat_totals = {}
for r in results:
    cats = r.get('categories', {})
    for k, v in cats.items():
        if k not in cat_totals:
            cat_totals[k] = []
        cat_totals[k].append(v if isinstance(v, (int, float)) else 0)

for cat, vals in sorted(cat_totals.items()):
    print(f"  {cat}: mean={np.mean(vals):.2f}, max={max(vals)}")

# Compare with Enron and Claude
print(f"\n" + "=" * 70)
print("COMPARISON WITH OTHER DATASETS")
print("=" * 70)

# Load Enron
try:
    with open('results/semantic_analysis_enron_results.json') as f:
        enron_data = json.load(f)
    enron_scores = [r.get('sycophancy_score', 5.0) for r in enron_data.get('results', [])]
    print(f"\nEnron (business emails): n={len(enron_scores)}, mean={np.mean(enron_scores):.3f}, std={np.std(enron_scores):.3f}")
except:
    print("\nEnron data not loaded")

# Load Claude Thinking
try:
    with open('results/semantic_analysis_claude_thinking_results.json') as f:
        thinking_data = json.load(f)
    thinking_internal = [r['thinking'].get('sycophancy_score', 5.0) for r in thinking_data['results'] if 'thinking' in r]
    thinking_output = [r['output'].get('sycophancy_score', 5.0) for r in thinking_data['results'] if 'output' in r]
    thinking_bracketed = [r['bracketed'].get('sycophancy_score', 5.0) for r in thinking_data['results'] if 'bracketed' in r]
    print(f"Claude Thinking (Internal): n={len(thinking_internal)}, mean={np.mean(thinking_internal):.3f}, std={np.std(thinking_internal):.3f}")
    print(f"Claude Thinking (Output):   n={len(thinking_output)}, mean={np.mean(thinking_output):.3f}, std={np.std(thinking_output):.3f}")
    print(f"Claude Thinking (Bracketed): n={len(thinking_bracketed)}, mean={np.mean(thinking_bracketed):.3f}, std={np.std(thinking_bracketed):.3f}")
except Exception as e:
    print(f"\nClaude Thinking data not loaded: {e}")

# Load Claude Standard
try:
    with open('results/semantic_analysis_claude_standard_results.json') as f:
        standard_data = json.load(f)
    standard_output = [r['output'].get('sycophancy_score', 5.0) for r in standard_data['results'] if 'output' in r]
    print(f"Claude Standard (Output):  n={len(standard_output)}, mean={np.mean(standard_output):.3f}, std={np.std(standard_output):.3f}")
except:
    print("\nClaude Standard data not loaded")

# CaSiNo summary
print(f"\nCaSiNo (human negotiations): n={len(scores)}, mean={np.mean(scores):.3f}, std={np.std(scores):.3f}")

print(f"\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

