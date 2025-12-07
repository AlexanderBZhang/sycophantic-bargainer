"""
Demo script for enhanced visualizations.

Tests new visualization features added in TASK-008:
- Exploitability heatmap
- Statistical significance annotations
- Time-series plots
- Environment comparison
"""

import pandas as pd
import sys
import os

# Add project to path
sys.path.insert(0, "/workspace")

from utils.visualization import (
    plot_exploitability_heatmap,
    plot_utility_gap_with_significance,
    plot_agreement_rates_with_significance,
    plot_environment_comparison,
    plot_all_results,
)


def main():
    """Test enhanced visualizations with mock experiment data."""

    # Load mock experiment results
    results_path = "/workspace/results/mock_experiments_20251125_172155.csv"

    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}")
        print("Please run mock experiments first.")
        return 1

    print(f"Loading results from {results_path}")
    results_df = pd.read_csv(results_path)
    print(f"Loaded {len(results_df)} experiments\n")

    # Create output directory
    output_dir = "/workspace/plots/enhanced"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("TESTING ENHANCED VISUALIZATIONS")
    print("=" * 60)

    # Test 1: Exploitability heatmap
    print("\n[1/5] Testing exploitability heatmap...")
    try:
        plot_exploitability_heatmap(
            results_df,
            save_path=f"{output_dir}/exploitability_heatmap.png"
        )
        print("✓ Exploitability heatmap created successfully")
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1

    # Test 2: Utility gap with significance
    print("\n[2/5] Testing utility gap with statistical significance...")
    try:
        plot_utility_gap_with_significance(
            results_df,
            save_path=f"{output_dir}/utility_gap_significance.png"
        )
        print("✓ Utility gap with significance created successfully")
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1

    # Test 3: Agreement rates with significance
    print("\n[3/5] Testing agreement rates with statistical significance...")
    try:
        plot_agreement_rates_with_significance(
            results_df,
            save_path=f"{output_dir}/agreement_rates_significance.png"
        )
        print("✓ Agreement rates with significance created successfully")
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1

    # Test 4: Environment comparison
    print("\n[4/5] Testing environment comparison...")
    try:
        plot_environment_comparison(
            results_df,
            metric="total_payoff",
            save_path=f"{output_dir}/environment_comparison.png"
        )
        print("✓ Environment comparison created successfully")
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1

    # Test 5: Generate all plots (including enhanced)
    print("\n[5/5] Testing plot_all_results with enhanced visualizations...")
    try:
        plot_all_results(
            results_df,
            output_dir=output_dir,
            include_enhanced=True
        )
        print("✓ All plots generated successfully")
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    print(f"\nPlots saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - exploitability_heatmap.png")
    print("  - utility_gap_significance.png")
    print("  - agreement_rates_significance.png")
    print("  - environment_comparison.png")
    print("  - payoff_comparison.png")
    print("  - utility_gap.png")
    print("  - agreement_rates.png")
    print("  - payoff_scatter.png")
    print("  - summary_statistics.csv")

    return 0


if __name__ == "__main__":
    exit(main())
