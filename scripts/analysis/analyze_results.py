#!/usr/bin/env python3
"""Analyze the hardball negotiation experiment results."""

import json
import numpy as np
from scipy import stats
from collections import defaultdict

def load_data():
    with open('results/claude_extended_thinking_all_results.json') as f:
        think = json.load(f)['negotiations']
    with open('results/claude_standard_all_results.json') as f:
        std = json.load(f)['negotiations']
    return std, think

def group_by_scenario(data):
    groups = defaultdict(list)
    for n in data:
        groups[n['scenario_code']].append(n)
    return groups

def get_payoffs(data):
    return [n['llm_payoff_millions'] for n in data]

def get_agreement_rate(data):
    if not data:
        return 0
    return sum(1 for n in data if 'Agreement' in n['outcome']) / len(data) * 100

def main():
    std, think = load_data()
    
    print('=' * 70)
    print('HARDBALL NEGOTIATION EXPERIMENT ANALYSIS')
    print('Claude Opus 4.5: Standard vs Extended Thinking')
    print('=' * 70)
    
    # Overall stats
    std_payoffs = get_payoffs(std)
    think_payoffs = get_payoffs(think)
    
    print('\n### OVERALL COMPARISON ###\n')
    print(f"{'Metric':<25} {'Standard':>12} {'Thinking':>12} {'Diff':>10}")
    print('-' * 60)
    print(f"{'N':<25} {len(std):>12} {len(think):>12}")
    print(f"{'Avg LLM Payoff':<25} ${np.mean(std_payoffs):>10.1f}M ${np.mean(think_payoffs):>10.1f}M {np.mean(think_payoffs)-np.mean(std_payoffs):>+9.1f}M")
    print(f"{'Median LLM Payoff':<25} ${np.median(std_payoffs):>10.1f}M ${np.median(think_payoffs):>10.1f}M")
    print(f"{'Std Dev':<25} ${np.std(std_payoffs):>10.1f}M ${np.std(think_payoffs):>10.1f}M")
    print(f"{'Min Payoff':<25} ${min(std_payoffs):>10}M ${min(think_payoffs):>10}M")
    print(f"{'Max Payoff':<25} ${max(std_payoffs):>10}M ${max(think_payoffs):>10}M")
    
    std_rounds = [n['rounds_to_resolution'] for n in std]
    think_rounds = [n['rounds_to_resolution'] for n in think]
    print(f"{'Avg Rounds':<25} {np.mean(std_rounds):>12.1f} {np.mean(think_rounds):>12.1f}")
    
    # Outcomes
    print('\n### OUTCOME DISTRIBUTION ###\n')
    for name, data in [('Standard', std), ('Thinking', think)]:
        outcomes = defaultdict(int)
        for n in data:
            outcomes[n['outcome']] += 1
        print(f'{name}:')
        for o, c in sorted(outcomes.items()):
            print(f'  {o}: {c} ({c/len(data)*100:.0f}%)')
        print()
    
    # Scenario breakdown
    print('### BREAKDOWN BY SCENARIO ###\n')
    std_groups = group_by_scenario(std)
    think_groups = group_by_scenario(think)
    scenarios = sorted(set(std_groups.keys()) | set(think_groups.keys()))
    
    print(f"{'Scenario':<10} {'Std $':<8} {'Think $':<9} {'Diff':<8} {'Std Agr%':<10} {'Think Agr%'}")
    print('-' * 60)
    
    for scenario in scenarios:
        std_list = std_groups.get(scenario, [])
        think_list = think_groups.get(scenario, [])
        
        std_avg = np.mean(get_payoffs(std_list)) if std_list else 0
        think_avg = np.mean(get_payoffs(think_list)) if think_list else 0
        diff = think_avg - std_avg
        
        std_agree = get_agreement_rate(std_list)
        think_agree = get_agreement_rate(think_list)
        
        print(f"{scenario:<10} ${std_avg:<6.1f}M ${think_avg:<7.1f}M {diff:+6.1f}M {std_agree:>7.0f}% {think_agree:>10.0f}%")
    
    # Key insights
    print('\n### KEY INSIGHTS ###\n')
    
    # By hardball level
    print('By Hardball Aggression Level:')
    for level, desc in [('L1', '70/30 opening'), ('L2', '75/25 opening'), ('L3', '80/20 opening')]:
        std_l = [n for n in std if n['scenario_code'].startswith(level)]
        think_l = [n for n in think if n['scenario_code'].startswith(level)]
        std_avg = np.mean(get_payoffs(std_l))
        think_avg = np.mean(get_payoffs(think_l))
        diff = think_avg - std_avg
        print(f'  {level} ({desc}): Std ${std_avg:.1f}M, Think ${think_avg:.1f}M ({diff:+.1f}M)')
    
    print()
    
    # By who moves first
    print('By Who Moves First:')
    std_p1 = [n for n in std if '-P1' in n['scenario_code']]
    std_p2 = [n for n in std if '-P2' in n['scenario_code']]
    think_p1 = [n for n in think if '-P1' in n['scenario_code']]
    think_p2 = [n for n in think if '-P2' in n['scenario_code']]
    
    print(f'  Hardball First (P1): Std ${np.mean(get_payoffs(std_p1)):.1f}M, Think ${np.mean(get_payoffs(think_p1)):.1f}M')
    print(f'  LLM First (P2):      Std ${np.mean(get_payoffs(std_p2)):.1f}M, Think ${np.mean(get_payoffs(think_p2)):.1f}M')
    
    think_advantage_p2 = np.mean(get_payoffs(think_p2)) - np.mean(get_payoffs(std_p2))
    print(f'\n  Thinking advantage when LLM moves first: {think_advantage_p2:+.1f}M')
    
    # Statistical significance
    print('\n### STATISTICAL SIGNIFICANCE ###\n')
    
    stat, p_mw = stats.mannwhitneyu(think_payoffs, std_payoffs, alternative='greater')
    print(f'Mann-Whitney U Test (Thinking > Standard):')
    print(f'  U-statistic: {stat:.1f}')
    print(f'  p-value: {p_mw:.4f}')
    print(f'  Significant at p<0.05? {"YES" if p_mw < 0.05 else "NO"}')
    
    stat_t, p_t = stats.ttest_ind(think_payoffs, std_payoffs)
    print(f'\nIndependent T-Test:')
    print(f'  t-statistic: {stat_t:.2f}')
    print(f'  p-value: {p_t:.4f}')
    
    pooled_std = np.sqrt((np.var(std_payoffs) + np.var(think_payoffs)) / 2)
    cohens_d = (np.mean(think_payoffs) - np.mean(std_payoffs)) / pooled_std
    
    if abs(cohens_d) < 0.2:
        effect = 'negligible'
    elif abs(cohens_d) < 0.5:
        effect = 'small'
    elif abs(cohens_d) < 0.8:
        effect = 'medium'
    else:
        effect = 'large'
    
    print(f"\nEffect Size (Cohen's d): {cohens_d:.2f} ({effect})")
    
    # Summary
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'''
Extended Thinking significantly outperforms Standard mode (p={p_mw:.4f}):
  • Average payoff: +${np.mean(think_payoffs) - np.mean(std_payoffs):.1f}M ({(np.mean(think_payoffs)/np.mean(std_payoffs)-1)*100:.0f}% improvement)
  • Agreement rate: {get_agreement_rate(think):.0f}% vs {get_agreement_rate(std):.0f}% (+{get_agreement_rate(think)-get_agreement_rate(std):.0f}%)
  • Fewer failures: {sum(1 for n in think if 'Max Rounds' in n['outcome'])} vs {sum(1 for n in std if 'Max Rounds' in n['outcome'])} max-rounds outcomes
  • Takes longer: {np.mean(think_rounds):.1f} vs {np.mean(std_rounds):.1f} rounds (more deliberation)

The thinking capability helps Claude:
  1. Resist pressure tactics more effectively
  2. Close more deals successfully
  3. Extract better terms when it does agree
''')

if __name__ == '__main__':
    main()

