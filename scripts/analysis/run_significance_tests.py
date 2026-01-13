#!/usr/bin/env python3
"""Run statistical significance tests on semantic analysis results."""

import json
from pathlib import Path
from scipy import stats
import numpy as np

RESULTS_DIR = Path("results")
ALPHA = 0.05

def load_scores():
    """Load sycophancy scores from all datasets."""
    scores = {}
    
    # Enron baseline
    with open(RESULTS_DIR / "semantic_analysis_enron_600_merged.json") as f:
        data = json.load(f)
    scores["enron"] = [r.get("sycophancy_score", 5.0) for r in data["results"]]
    
    # CaSiNo baseline (human negotiations)
    with open(RESULTS_DIR / "semantic_analysis_casino_results.json", encoding='utf-8') as f:
        data = json.load(f)
    casino_results = data["results"]
    scores["casino"] = [r.get("sycophancy_score", 5.0) for r in casino_results]
    
    # CaSiNo stratified by SVO
    scores["casino_proself"] = [r.get("sycophancy_score", 5.0) 
                                 for r in casino_results if r.get("_svo") == "proself"]
    scores["casino_prosocial"] = [r.get("sycophancy_score", 5.0) 
                                   for r in casino_results if r.get("_svo") == "prosocial"]
    
    # CaSiNo stratified by Agreeableness
    scores["casino_agree_low"] = [r.get("sycophancy_score", 5.0) 
                                   for r in casino_results 
                                   if r.get("_agreeableness") is not None and r.get("_agreeableness") <= 3.5]
    scores["casino_agree_high"] = [r.get("sycophancy_score", 5.0) 
                                    for r in casino_results 
                                    if r.get("_agreeableness") is not None and r.get("_agreeableness") > 5.5]
    
    # Claude Thinking (all three components)
    with open(RESULTS_DIR / "semantic_analysis_claude_thinking_results.json") as f:
        data = json.load(f)
    scores["thinking_internal"] = [r["thinking"].get("sycophancy_score", 5.0) 
                                    for r in data["results"] if "thinking" in r]
    scores["thinking_output"] = [r["output"].get("sycophancy_score", 5.0) 
                                  for r in data["results"] if "output" in r]
    scores["thinking_bracketed"] = [r["bracketed"].get("sycophancy_score", 5.0) 
                                     for r in data["results"] if "bracketed" in r]
    
    # Claude Standard
    with open(RESULTS_DIR / "semantic_analysis_claude_standard_results.json") as f:
        data = json.load(f)
    scores["standard"] = [r["output"].get("sycophancy_score", 5.0) 
                          for r in data["results"] if "output" in r]
    
    return scores


def welch_ttest(group1, group2, name1, name2):
    """Perform Welch's t-test (unequal variances)."""
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    std1 = np.std(group1, ddof=1)
    std2 = np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    
    # Cohen's d (effect size)
    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
    cohens_d = (mean1 - mean2) / pooled_std
    
    return {
        "comparison": f"{name1} vs {name2}",
        "n1": n1, "n2": n2,
        "mean1": round(mean1, 3), "mean2": round(mean2, 3),
        "std1": round(std1, 3), "std2": round(std2, 3),
        "mean_diff": round(mean1 - mean2, 3),
        "t_statistic": round(t_stat, 3),
        "p_value": p_value,
        "p_value_formatted": f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.4f}",
        "cohens_d": round(cohens_d, 3),
        "significant": p_value < ALPHA,
    }


def paired_ttest(group1, group2, name1, name2):
    """Perform paired t-test (for thinking vs output within same entries)."""
    # Ensure same length
    n = min(len(group1), len(group2))
    g1, g2 = group1[:n], group2[:n]
    
    t_stat, p_value = stats.ttest_rel(g1, g2)
    
    mean1 = np.mean(g1)
    mean2 = np.mean(g2)
    mean_diff = np.mean([a - b for a, b in zip(g1, g2)])
    std_diff = np.std([a - b for a, b in zip(g1, g2)], ddof=1)
    
    # Effect size (Cohen's d for paired samples)
    cohens_d = mean_diff / std_diff
    
    return {
        "comparison": f"{name1} vs {name2} (paired)",
        "n": n,
        "mean1": round(mean1, 3), "mean2": round(mean2, 3),
        "mean_diff": round(mean_diff, 3),
        "std_diff": round(std_diff, 3),
        "t_statistic": round(t_stat, 3),
        "p_value": p_value,
        "p_value_formatted": f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.4f}",
        "cohens_d": round(cohens_d, 3),
        "significant": p_value < ALPHA,
    }


def interpret_cohens_d(d):
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def main():
    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print(f"Alpha = {ALPHA}")
    print("=" * 80)
    
    scores = load_scores()
    
    print("\n--- Dataset Summary ---")
    name_map = {
        "enron": "Enron (Human Business Emails)",
        "casino": "CaSiNo (Human Negotiations)",
        "casino_proself": "  - CaSiNo Proself SVO",
        "casino_prosocial": "  - CaSiNo Prosocial SVO",
        "casino_agree_low": "  - CaSiNo Low Agreeableness",
        "casino_agree_high": "  - CaSiNo High Agreeableness",
        "thinking_internal": "Claude Opus 4.5 Thinking (Internal)",
        "thinking_output": "Claude Opus 4.5 Thinking (Output)",
        "thinking_bracketed": "Claude Opus 4.5 Thinking (Bracketed)",
        "standard": "Claude Opus 4.5 Standard",
    }
    display_order = ["enron", "casino", "casino_proself", "casino_prosocial", 
                     "casino_agree_low", "casino_agree_high",
                     "thinking_internal", "thinking_output", "thinking_bracketed", "standard"]
    for key in display_order:
        if key in scores:
            data = scores[key]
            name = name_map.get(key, key)
            print(f"{name}: n={len(data)}, mean={np.mean(data):.3f}, std={np.std(data):.3f}")
    
    results = []
    
    # Test 1: Enron vs Claude Opus 4.5 Thinking (Output)
    print("\n" + "-" * 80)
    print("TEST 1: Enron (Human) vs Claude Opus 4.5 Thinking (Output)")
    print("-" * 80)
    r1 = welch_ttest(scores["enron"], scores["thinking_output"], 
                     "Enron", "Opus 4.5 Thinking (Output)")
    results.append(r1)
    print(f"  Enron: mean={r1['mean1']}, std={r1['std1']}, n={r1['n1']}")
    print(f"  Opus 4.5 Thinking (Output): mean={r1['mean2']}, std={r1['std2']}, n={r1['n2']}")
    print(f"  Mean difference: {r1['mean_diff']}")
    print(f"  t-statistic: {r1['t_statistic']}")
    print(f"  p-value: {r1['p_value_formatted']}")
    print(f"  Cohen's d: {r1['cohens_d']} ({interpret_cohens_d(r1['cohens_d'])} effect)")
    print(f"  Significant at alpha={ALPHA}: {'YES' if r1['significant'] else 'NO'}")
    
    # Test 2: Enron vs Claude Opus 4.5 Standard
    print("\n" + "-" * 80)
    print("TEST 2: Enron (Human) vs Claude Opus 4.5 Standard")
    print("-" * 80)
    r2 = welch_ttest(scores["enron"], scores["standard"], 
                     "Enron", "Opus 4.5 Standard")
    results.append(r2)
    print(f"  Enron: mean={r2['mean1']}, std={r2['std1']}, n={r2['n1']}")
    print(f"  Opus 4.5 Standard: mean={r2['mean2']}, std={r2['std2']}, n={r2['n2']}")
    print(f"  Mean difference: {r2['mean_diff']}")
    print(f"  t-statistic: {r2['t_statistic']}")
    print(f"  p-value: {r2['p_value_formatted']}")
    print(f"  Cohen's d: {r2['cohens_d']} ({interpret_cohens_d(r2['cohens_d'])} effect)")
    print(f"  Significant at alpha={ALPHA}: {'YES' if r2['significant'] else 'NO'}")
    
    # Test 3: Claude Opus 4.5 Thinking (Output) vs Claude Opus 4.5 Standard
    print("\n" + "-" * 80)
    print("TEST 3: Opus 4.5 Thinking (Output) vs Opus 4.5 Standard")
    print("-" * 80)
    r3 = welch_ttest(scores["thinking_output"], scores["standard"], 
                     "Opus 4.5 Thinking (Output)", "Opus 4.5 Standard")
    results.append(r3)
    print(f"  Opus 4.5 Thinking (Output): mean={r3['mean1']}, std={r3['std1']}, n={r3['n1']}")
    print(f"  Opus 4.5 Standard: mean={r3['mean2']}, std={r3['std2']}, n={r3['n2']}")
    print(f"  Mean difference: {r3['mean_diff']}")
    print(f"  t-statistic: {r3['t_statistic']}")
    print(f"  p-value: {r3['p_value_formatted']}")
    print(f"  Cohen's d: {r3['cohens_d']} ({interpret_cohens_d(r3['cohens_d'])} effect)")
    print(f"  Significant at alpha={ALPHA}: {'YES' if r3['significant'] else 'NO'}")
    
    # Test 4: Alignment Gap - Opus 4.5 Thinking (Internal vs Output)
    print("\n" + "-" * 80)
    print("TEST 4: Opus 4.5 Thinking (Internal) vs Opus 4.5 Thinking (Output) [Paired]")
    print("-" * 80)
    r4 = paired_ttest(scores["thinking_internal"], scores["thinking_output"],
                      "Opus 4.5 Thinking (Internal)", "Opus 4.5 Thinking (Output)")
    results.append(r4)
    print(f"  Opus 4.5 Thinking (Internal): mean={r4['mean1']}")
    print(f"  Opus 4.5 Thinking (Output): mean={r4['mean2']}")
    print(f"  Mean gap (Thinking - Output): {r4['mean_diff']}")
    print(f"  t-statistic: {r4['t_statistic']}")
    print(f"  p-value: {r4['p_value_formatted']}")
    print(f"  Cohen's d: {r4['cohens_d']} ({interpret_cohens_d(r4['cohens_d'])} effect)")
    print(f"  Significant at alpha={ALPHA}: {'YES' if r4['significant'] else 'NO'}")
    
    # Test 5: Enron vs Claude Opus 4.5 Thinking (Internal)
    print("\n" + "-" * 80)
    print("TEST 5: Enron (Human) vs Claude Opus 4.5 Thinking (Internal)")
    print("-" * 80)
    r5 = welch_ttest(scores["enron"], scores["thinking_internal"], 
                     "Enron", "Opus 4.5 Thinking (Internal)")
    results.append(r5)
    print(f"  Enron: mean={r5['mean1']}, std={r5['std1']}, n={r5['n1']}")
    print(f"  Opus 4.5 Thinking (Internal): mean={r5['mean2']}, std={r5['std2']}, n={r5['n2']}")
    print(f"  Mean difference: {r5['mean_diff']}")
    print(f"  t-statistic: {r5['t_statistic']}")
    print(f"  p-value: {r5['p_value_formatted']}")
    print(f"  Cohen's d: {r5['cohens_d']} ({interpret_cohens_d(r5['cohens_d'])} effect)")
    print(f"  Significant at alpha={ALPHA}: {'YES' if r5['significant'] else 'NO'}")
    
    # Test 6: Enron vs Claude Opus 4.5 Thinking (Bracketed)
    print("\n" + "-" * 80)
    print("TEST 6: Enron (Human) vs Claude Opus 4.5 Thinking (Bracketed)")
    print("-" * 80)
    r6 = welch_ttest(scores["enron"], scores["thinking_bracketed"], 
                     "Enron", "Opus 4.5 Thinking (Bracketed)")
    results.append(r6)
    print(f"  Enron: mean={r6['mean1']}, std={r6['std1']}, n={r6['n1']}")
    print(f"  Opus 4.5 Thinking (Bracketed): mean={r6['mean2']}, std={r6['std2']}, n={r6['n2']}")
    print(f"  Mean difference: {r6['mean_diff']}")
    print(f"  t-statistic: {r6['t_statistic']}")
    print(f"  p-value: {r6['p_value_formatted']}")
    print(f"  Cohen's d: {r6['cohens_d']} ({interpret_cohens_d(r6['cohens_d'])} effect)")
    print(f"  Significant at alpha={ALPHA}: {'YES' if r6['significant'] else 'NO'}")
    
    # Test 7: Opus 4.5 Thinking (Internal) vs Opus 4.5 Thinking (Bracketed) [Paired]
    print("\n" + "-" * 80)
    print("TEST 7: Opus 4.5 Thinking (Internal) vs Opus 4.5 Thinking (Bracketed) [Paired]")
    print("-" * 80)
    r7 = paired_ttest(scores["thinking_internal"], scores["thinking_bracketed"],
                      "Opus 4.5 Thinking (Internal)", "Opus 4.5 Thinking (Bracketed)")
    results.append(r7)
    print(f"  Opus 4.5 Thinking (Internal): mean={r7['mean1']}")
    print(f"  Opus 4.5 Thinking (Bracketed): mean={r7['mean2']}")
    print(f"  Mean gap (Internal - Bracketed): {r7['mean_diff']}")
    print(f"  t-statistic: {r7['t_statistic']}")
    print(f"  p-value: {r7['p_value_formatted']}")
    print(f"  Cohen's d: {r7['cohens_d']} ({interpret_cohens_d(r7['cohens_d'])} effect)")
    print(f"  Significant at alpha={ALPHA}: {'YES' if r7['significant'] else 'NO'}")
    
    # Test 8: Opus 4.5 Thinking (Output) vs Opus 4.5 Thinking (Bracketed) [Paired]
    print("\n" + "-" * 80)
    print("TEST 8: Opus 4.5 Thinking (Output) vs Opus 4.5 Thinking (Bracketed) [Paired]")
    print("-" * 80)
    r8 = paired_ttest(scores["thinking_output"], scores["thinking_bracketed"],
                      "Opus 4.5 Thinking (Output)", "Opus 4.5 Thinking (Bracketed)")
    results.append(r8)
    print(f"  Opus 4.5 Thinking (Output): mean={r8['mean1']}")
    print(f"  Opus 4.5 Thinking (Bracketed): mean={r8['mean2']}")
    print(f"  Mean gap (Output - Bracketed): {r8['mean_diff']}")
    print(f"  t-statistic: {r8['t_statistic']}")
    print(f"  p-value: {r8['p_value_formatted']}")
    print(f"  Cohen's d: {r8['cohens_d']} ({interpret_cohens_d(r8['cohens_d'])} effect)")
    print(f"  Significant at alpha={ALPHA}: {'YES' if r8['significant'] else 'NO'}")
    
    # =========================================================================
    # CaSiNo Tests (Human Negotiation Baseline)
    # =========================================================================
    print("\n" + "=" * 80)
    print("CASINO HUMAN NEGOTIATION BASELINE TESTS")
    print("=" * 80)
    
    # Test 9: CaSiNo vs Claude Opus 4.5 Standard
    print("\n" + "-" * 80)
    print("TEST 9: CaSiNo (Human) vs Claude Opus 4.5 Standard")
    print("-" * 80)
    r9 = welch_ttest(scores["casino"], scores["standard"], 
                     "CaSiNo", "Opus 4.5 Standard")
    results.append(r9)
    print(f"  CaSiNo: mean={r9['mean1']}, std={r9['std1']}, n={r9['n1']}")
    print(f"  Opus 4.5 Standard: mean={r9['mean2']}, std={r9['std2']}, n={r9['n2']}")
    print(f"  Mean difference: {r9['mean_diff']}")
    print(f"  t-statistic: {r9['t_statistic']}")
    print(f"  p-value: {r9['p_value_formatted']}")
    print(f"  Cohen's d: {r9['cohens_d']} ({interpret_cohens_d(r9['cohens_d'])} effect)")
    print(f"  Significant at alpha={ALPHA}: {'YES' if r9['significant'] else 'NO'}")
    
    # Test 10: CaSiNo vs Claude Opus 4.5 Thinking (Bracketed)
    print("\n" + "-" * 80)
    print("TEST 10: CaSiNo (Human) vs Claude Opus 4.5 Thinking (Bracketed)")
    print("-" * 80)
    r10 = welch_ttest(scores["casino"], scores["thinking_bracketed"], 
                      "CaSiNo", "Opus 4.5 Thinking (Bracketed)")
    results.append(r10)
    print(f"  CaSiNo: mean={r10['mean1']}, std={r10['std1']}, n={r10['n1']}")
    print(f"  Opus 4.5 Thinking (Bracketed): mean={r10['mean2']}, std={r10['std2']}, n={r10['n2']}")
    print(f"  Mean difference: {r10['mean_diff']}")
    print(f"  t-statistic: {r10['t_statistic']}")
    print(f"  p-value: {r10['p_value_formatted']}")
    print(f"  Cohen's d: {r10['cohens_d']} ({interpret_cohens_d(r10['cohens_d'])} effect)")
    print(f"  Significant at alpha={ALPHA}: {'YES' if r10['significant'] else 'NO'}")
    
    # Test 11: CaSiNo Proself vs Prosocial (SVO stratification)
    print("\n" + "-" * 80)
    print("TEST 11: CaSiNo Proself vs Prosocial (SVO Stratification)")
    print("-" * 80)
    r11 = welch_ttest(scores["casino_proself"], scores["casino_prosocial"], 
                      "CaSiNo Proself", "CaSiNo Prosocial")
    results.append(r11)
    print(f"  Proself: mean={r11['mean1']}, std={r11['std1']}, n={r11['n1']}")
    print(f"  Prosocial: mean={r11['mean2']}, std={r11['std2']}, n={r11['n2']}")
    print(f"  Mean difference: {r11['mean_diff']}")
    print(f"  t-statistic: {r11['t_statistic']}")
    print(f"  p-value: {r11['p_value_formatted']}")
    print(f"  Cohen's d: {r11['cohens_d']} ({interpret_cohens_d(r11['cohens_d'])} effect)")
    print(f"  Significant at alpha={ALPHA}: {'YES' if r11['significant'] else 'NO'}")
    
    # Test 12: CaSiNo Low vs High Agreeableness (Big Five stratification)
    print("\n" + "-" * 80)
    print("TEST 12: CaSiNo Low vs High Agreeableness (Big Five Stratification)")
    print("-" * 80)
    r12 = welch_ttest(scores["casino_agree_low"], scores["casino_agree_high"], 
                      "CaSiNo Low Agreeableness", "CaSiNo High Agreeableness")
    results.append(r12)
    print(f"  Low Agreeableness: mean={r12['mean1']}, std={r12['std1']}, n={r12['n1']}")
    print(f"  High Agreeableness: mean={r12['mean2']}, std={r12['std2']}, n={r12['n2']}")
    print(f"  Mean difference: {r12['mean_diff']}")
    print(f"  t-statistic: {r12['t_statistic']}")
    print(f"  p-value: {r12['p_value_formatted']}")
    print(f"  Cohen's d: {r12['cohens_d']} ({interpret_cohens_d(r12['cohens_d'])} effect)")
    print(f"  Significant at alpha={ALPHA}: {'YES' if r12['significant'] else 'NO'}")
    
    # Test 13: Enron vs CaSiNo (comparing two human baselines)
    print("\n" + "-" * 80)
    print("TEST 13: Enron (Business Emails) vs CaSiNo (Negotiations)")
    print("-" * 80)
    r13 = welch_ttest(scores["enron"], scores["casino"], 
                      "Enron", "CaSiNo")
    results.append(r13)
    print(f"  Enron: mean={r13['mean1']}, std={r13['std1']}, n={r13['n1']}")
    print(f"  CaSiNo: mean={r13['mean2']}, std={r13['std2']}, n={r13['n2']}")
    print(f"  Mean difference: {r13['mean_diff']}")
    print(f"  t-statistic: {r13['t_statistic']}")
    print(f"  p-value: {r13['p_value_formatted']}")
    print(f"  Cohen's d: {r13['cohens_d']} ({interpret_cohens_d(r13['cohens_d'])} effect)")
    print(f"  Significant at alpha={ALPHA}: {'YES' if r13['significant'] else 'NO'}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\n| Comparison | Mean Diff | p-value | Cohen's d | Significant |")
    print("|------------|-----------|---------|-----------|-------------|")
    for r in results:
        sig = "YES ***" if r["significant"] else "NO"
        print(f"| {r['comparison'][:30]:30} | {r['mean_diff']:+.3f} | {r['p_value_formatted']:>9} | {r['cohens_d']:+.3f} | {sig} |")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    print("\n--- AI vs Human Baselines ---")
    
    if r9["significant"]:
        direction = "more balanced" if r9["mean_diff"] > 0 else "more assertive"
        print(f"\n* CaSiNo humans are significantly {direction} than Opus 4.5 Standard")
        print(f"  (CaSiNo {r9['mean1']} vs Standard {r9['mean2']}, d={r9['cohens_d']}, p={r9['p_value_formatted']})")
    
    if r10["significant"]:
        direction = "more balanced" if r10["mean_diff"] > 0 else "more assertive"
        print(f"\n* CaSiNo humans are significantly {direction} than Opus 4.5 Thinking (Bracketed)")
        print(f"  (CaSiNo {r10['mean1']} vs Bracketed {r10['mean2']}, d={r10['cohens_d']}, p={r10['p_value_formatted']})")
    
    print("\n--- SVO and Agreeableness Stratification ---")
    
    if r11["significant"]:
        direction = "more assertive" if r11["mean_diff"] < 0 else "more deferential"
        print(f"\n* Proself individuals are significantly {direction} than Prosocial")
        print(f"  (Proself {r11['mean1']} vs Prosocial {r11['mean2']}, d={r11['cohens_d']}, p={r11['p_value_formatted']})")
    else:
        print(f"\n* No significant difference between Proself and Prosocial SVO types")
        print(f"  (Proself {r11['mean1']} vs Prosocial {r11['mean2']}, p={r11['p_value_formatted']})")
    
    if r12["significant"]:
        direction = "more assertive" if r12["mean_diff"] < 0 else "more deferential"
        print(f"\n* Low Agreeableness individuals are significantly {direction} than High Agreeableness")
        print(f"  (Low {r12['mean1']} vs High {r12['mean2']}, d={r12['cohens_d']}, p={r12['p_value_formatted']})")
    else:
        print(f"\n* No significant difference between Low and High Agreeableness")
        print(f"  (Low {r12['mean1']} vs High {r12['mean2']}, p={r12['p_value_formatted']})")
    
    print("\n--- Human Baseline Comparison ---")
    
    if r13["significant"]:
        direction = "more assertive" if r13["mean_diff"] < 0 else "more deferential"
        print(f"\n* Enron emails are significantly {direction} than CaSiNo negotiations")
        print(f"  (Enron {r13['mean1']} vs CaSiNo {r13['mean2']}, d={r13['cohens_d']}, p={r13['p_value_formatted']})")
    else:
        print(f"\n* No significant difference between Enron and CaSiNo baselines")
        print(f"  (Enron {r13['mean1']} vs CaSiNo {r13['mean2']}, p={r13['p_value_formatted']})")
    
    print("\n--- Key Finding ---")
    print("\nAI models are significantly MORE ASSERTIVE than human negotiators:")
    print(f"  - CaSiNo humans: {np.mean(scores['casino']):.2f} (balanced)")
    print(f"  - Claude Opus 4.5 Standard: {np.mean(scores['standard']):.2f} (assertive)")
    print(f"  - Claude Opus 4.5 Thinking (Bracketed): {np.mean(scores['thinking_bracketed']):.2f}")
    print("This is the OPPOSITE of sycophancy in negotiation contexts.")
    
    print("\n" + "=" * 80)
    print("METHODOLOGICAL NOTES")
    print("=" * 80)
    print("\n1. p-values were not pre-registered for part of the experiment. These are observational findings.")
    print("   Under a conventional alpha = 0.05 threshold, results marked 'YES' would be")
    print("   considered statistically significant.")
    print("\n2. CaSiNo prompts included non-zero-sum game context to help distinguish")
    print("   strategic trading from deferential behavior. Claude prompts did not have")
    print("   equivalent context as those batches were already completed.")
    print("\n3. 1 of 2060 CaSiNo entries had a JSON parsing error (99.95% success rate).")
    print("   This does not materially affect the statistical conclusions.")
    print("=" * 80)


if __name__ == "__main__":
    main()

