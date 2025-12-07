"""
Visualization tools for bargaining experiments.

Creates plots and charts for analyzing results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Optional scipy import for statistical significance testing
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    stats = None

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_payoff_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """
    Plot payoff comparison between agent personas.

    Args:
        results_df: DataFrame with experiment results
        save_path: Path to save figure (if None, displays)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Agent A payoffs
    sns.boxplot(
        data=results_df,
        x="agent_A_persona",
        y="agent_A_payoff",
        ax=axes[0],
    )
    axes[0].set_title("Agent A Payoffs by Persona")
    axes[0].set_xlabel("Persona Type")
    axes[0].set_ylabel("Payoff")

    # Agent B payoffs
    sns.boxplot(
        data=results_df,
        x="agent_B_persona",
        y="agent_B_payoff",
        ax=axes[1],
    )
    axes[1].set_title("Agent B Payoffs by Persona")
    axes[1].set_xlabel("Persona Type")
    axes[1].set_ylabel("Payoff")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def plot_utility_gap(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """
    Plot utility gap between agents across matchups.

    Args:
        results_df: DataFrame with experiment results
        save_path: Path to save figure
    """
    # Calculate utility gap (A - B)
    results_df = results_df.copy()
    results_df["utility_gap"] = results_df["agent_A_payoff"] - results_df["agent_B_payoff"]

    # Create matchup label
    results_df["matchup"] = (
        results_df["agent_A_persona"] + " vs " + results_df["agent_B_persona"]
    )

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=results_df,
        x="matchup",
        y="utility_gap",
        hue="matchup",
        legend=False,
    )

    plt.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Equal payoffs")
    plt.title("Utility Gap: Agent A Payoff - Agent B Payoff")
    plt.xlabel("Matchup")
    plt.ylabel("Utility Gap (A - B)")
    plt.xticks(rotation=45, ha="right")
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def plot_agreement_rates(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """
    Plot agreement rates by matchup.

    Args:
        results_df: DataFrame with experiment results
        save_path: Path to save figure
    """
    results_df = results_df.copy()
    results_df["agreement"] = results_df["total_payoff"] > 0
    results_df["matchup"] = (
        results_df["agent_A_persona"] + " vs " + results_df["agent_B_persona"]
    )

    # Calculate agreement rates
    agreement_rates = results_df.groupby("matchup")["agreement"].mean() * 100

    plt.figure(figsize=(10, 6))

    agreement_rates.plot(kind="bar", color="steelblue")

    plt.title("Agreement Rates by Matchup")
    plt.xlabel("Matchup")
    plt.ylabel("Agreement Rate (%)")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 105)

    # Add value labels on bars
    for i, v in enumerate(agreement_rates):
        plt.text(i, v + 2, f"{v:.1f}%", ha="center", va="bottom")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def plot_payoff_scatter(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """
    Scatter plot of Agent A vs Agent B payoffs.

    Args:
        results_df: DataFrame with experiment results
        save_path: Path to save figure
    """
    results_df = results_df.copy()
    results_df["matchup"] = (
        results_df["agent_A_persona"] + " vs " + results_df["agent_B_persona"]
    )

    plt.figure(figsize=(10, 8))

    for matchup in results_df["matchup"].unique():
        subset = results_df[results_df["matchup"] == matchup]
        plt.scatter(
            subset["agent_A_payoff"],
            subset["agent_B_payoff"],
            label=matchup,
            alpha=0.6,
            s=50,
        )

    # Add diagonal line (equal payoffs)
    max_val = max(results_df["agent_A_payoff"].max(), results_df["agent_B_payoff"].max())
    plt.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="Equal split")

    plt.xlabel("Agent A Payoff")
    plt.ylabel("Agent B Payoff")
    plt.title("Payoff Distribution: Agent A vs Agent B")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def plot_fairness_comparison(
    results_df: pd.DataFrame,
    fairness_metric: str = "gini",
    save_path: Optional[str] = None,
):
    """
    Plot fairness metric comparison.

    Args:
        results_df: DataFrame with experiment results
        fairness_metric: Metric column name
        save_path: Path to save figure
    """
    if fairness_metric not in results_df.columns:
        raise ValueError(f"Metric {fairness_metric} not found in results")

    results_df = results_df.copy()
    results_df["matchup"] = (
        results_df["agent_A_persona"] + " vs " + results_df["agent_B_persona"]
    )

    plt.figure(figsize=(10, 6))

    sns.violinplot(
        data=results_df,
        x="matchup",
        y=fairness_metric,
        hue="matchup",
        legend=False,
    )

    plt.title(f"Fairness Comparison: {fairness_metric}")
    plt.xlabel("Matchup")
    plt.ylabel(fairness_metric)
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def create_summary_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics table.

    Args:
        results_df: DataFrame with experiment results

    Returns:
        Summary DataFrame
    """
    results_df = results_df.copy()
    results_df["matchup"] = (
        results_df["agent_A_persona"] + " vs " + results_df["agent_B_persona"]
    )
    results_df["agreement"] = results_df["total_payoff"] > 0

    summary = results_df.groupby("matchup").agg({
        "agent_A_payoff": ["mean", "std"],
        "agent_B_payoff": ["mean", "std"],
        "total_payoff": ["mean", "std"],
        "agreement": "mean",
    })

    # Flatten column names
    summary.columns = [
        "A_mean", "A_std",
        "B_mean", "B_std",
        "Total_mean", "Total_std",
        "Agreement_rate",
    ]

    # Convert agreement rate to percentage
    summary["Agreement_rate"] = summary["Agreement_rate"] * 100

    return summary


def plot_all_results(
    results_df: pd.DataFrame,
    output_dir: str = "./plots",
    include_enhanced: bool = True,
):
    """
    Generate all standard plots.

    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save plots
        include_enhanced: Whether to include new enhanced visualizations
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("Generating plots...")

    plot_payoff_comparison(results_df, save_path=f"{output_dir}/payoff_comparison.png")
    print("✓ Payoff comparison")

    plot_utility_gap(results_df, save_path=f"{output_dir}/utility_gap.png")
    print("✓ Utility gap")

    plot_agreement_rates(results_df, save_path=f"{output_dir}/agreement_rates.png")
    print("✓ Agreement rates")

    plot_payoff_scatter(results_df, save_path=f"{output_dir}/payoff_scatter.png")
    print("✓ Payoff scatter")

    if include_enhanced:
        print("\nGenerating enhanced visualizations...")

        plot_exploitability_heatmap(results_df, save_path=f"{output_dir}/exploitability_heatmap.png")
        print("✓ Exploitability heatmap")

        plot_utility_gap_with_significance(results_df, save_path=f"{output_dir}/utility_gap_significance.png")
        print("✓ Utility gap with significance")

        plot_agreement_rates_with_significance(results_df, save_path=f"{output_dir}/agreement_rates_significance.png")
        print("✓ Agreement rates with significance")

        plot_environment_comparison(results_df, metric="total_payoff", save_path=f"{output_dir}/environment_comparison.png")
        print("✓ Environment comparison")

    # Save summary table
    summary = create_summary_table(results_df)
    summary.to_csv(f"{output_dir}/summary_statistics.csv")
    print("✓ Summary table")

    print(f"\nAll plots saved to {output_dir}/")


def compute_statistical_significance(
    group1: np.ndarray,
    group2: np.ndarray,
    test: str = "mann-whitney",
) -> Tuple[float, str]:
    """
    Compute statistical significance between two groups.

    Args:
        group1: First group samples
        group2: Second group samples
        test: Statistical test to use ("mann-whitney" or "t-test")

    Returns:
        Tuple of (p_value, significance_marker)
        Markers: "***" (p<0.001), "**" (p<0.01), "*" (p<0.05), "ns" (not significant)

    Note:
        Requires scipy. If scipy is not installed, returns (1.0, "ns").
    """
    if not HAS_SCIPY:
        print("Warning: scipy not installed. Statistical significance testing disabled.")
        return 1.0, "ns"

    if len(group1) < 2 or len(group2) < 2:
        return 1.0, "ns"

    if test == "mann-whitney":
        # Non-parametric test (doesn't assume normal distribution)
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative="two-sided")
    elif test == "t-test":
        # Parametric test (assumes normal distribution)
        statistic, p_value = stats.ttest_ind(group1, group2)
    else:
        raise ValueError(f"Unknown test: {test}")

    # Determine significance marker
    if p_value < 0.001:
        marker = "***"
    elif p_value < 0.01:
        marker = "**"
    elif p_value < 0.05:
        marker = "*"
    else:
        marker = "ns"

    return p_value, marker


def plot_exploitability_heatmap(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """
    Plot exploitability heatmap showing payoff differential matrix.

    Shows mean payoff difference (Agent A - Agent B) for each persona matchup.
    Positive values indicate Agent A exploits Agent B.
    Negative values indicate Agent B exploits Agent A.

    Args:
        results_df: DataFrame with experiment results
        save_path: Path to save figure
    """
    results_df = results_df.copy()

    # Calculate utility gap
    results_df["utility_gap"] = results_df["agent_A_payoff"] - results_df["agent_B_payoff"]

    # Create pivot table: rows = Agent A persona, cols = Agent B persona
    pivot = results_df.pivot_table(
        values="utility_gap",
        index="agent_A_persona",
        columns="agent_B_persona",
        aggfunc="mean",
    )

    plt.figure(figsize=(10, 8))

    # Create heatmap with diverging colormap (red = A exploits B, blue = B exploits A)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        cbar_kws={"label": "Payoff Difference (A - B)"},
        linewidths=0.5,
        linecolor="gray",
    )

    plt.title("Exploitability Matrix: Agent A Payoff - Agent B Payoff\n(Positive = A exploits B, Negative = B exploits A)")
    plt.xlabel("Agent B Persona")
    plt.ylabel("Agent A Persona")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def plot_utility_gap_with_significance(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
    reference_matchup: Optional[str] = None,
):
    """
    Plot utility gap with statistical significance annotations.

    Args:
        results_df: DataFrame with experiment results
        save_path: Path to save figure
        reference_matchup: Matchup to compare against (e.g., "rational vs rational")
                          If None, compares adjacent matchups
    """
    results_df = results_df.copy()
    results_df["utility_gap"] = results_df["agent_A_payoff"] - results_df["agent_B_payoff"]
    results_df["matchup"] = (
        results_df["agent_A_persona"] + " vs " + results_df["agent_B_persona"]
    )

    matchups = results_df["matchup"].unique()

    fig, ax = plt.subplots(figsize=(12, 7))

    # Create box plot
    sns.boxplot(
        data=results_df,
        x="matchup",
        y="utility_gap",
        hue="matchup",
        legend=False,
        ax=ax,
    )

    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Equal payoffs")
    ax.set_title("Utility Gap with Statistical Significance")
    ax.set_xlabel("Matchup")
    ax.set_ylabel("Utility Gap (A - B)")
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Add significance annotations
    if reference_matchup and reference_matchup in matchups:
        # Compare all matchups against reference
        ref_data = results_df[results_df["matchup"] == reference_matchup]["utility_gap"].values

        y_max = results_df["utility_gap"].max()
        y_range = results_df["utility_gap"].max() - results_df["utility_gap"].min()
        annotation_height = y_max + 0.1 * y_range

        for i, matchup in enumerate(matchups):
            if matchup != reference_matchup:
                group_data = results_df[results_df["matchup"] == matchup]["utility_gap"].values
                p_value, marker = compute_statistical_significance(ref_data, group_data)

                if marker != "ns":
                    ax.text(
                        i, annotation_height, marker,
                        ha="center", va="bottom", fontsize=12, fontweight="bold"
                    )
    else:
        # Compare adjacent matchups
        for i in range(len(matchups) - 1):
            group1 = results_df[results_df["matchup"] == matchups[i]]["utility_gap"].values
            group2 = results_df[results_df["matchup"] == matchups[i + 1]]["utility_gap"].values

            p_value, marker = compute_statistical_significance(group1, group2)

            if marker != "ns":
                y_max = max(group1.max(), group2.max())
                y_range = results_df["utility_gap"].max() - results_df["utility_gap"].min()
                annotation_height = y_max + 0.15 * y_range

                # Draw bracket
                x1, x2 = i, i + 1
                ax.plot([x1, x1, x2, x2],
                       [annotation_height - 0.05 * y_range, annotation_height,
                        annotation_height, annotation_height - 0.05 * y_range],
                       'k-', lw=1)

                # Add significance marker
                ax.text(
                    (x1 + x2) / 2, annotation_height + 0.02 * y_range,
                    marker, ha="center", va="bottom", fontsize=10, fontweight="bold"
                )

    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def plot_agreement_rates_with_significance(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """
    Plot agreement rates with statistical significance annotations.

    Args:
        results_df: DataFrame with experiment results
        save_path: Path to save figure
    """
    results_df = results_df.copy()
    results_df["agreement"] = results_df["total_payoff"] > 0
    results_df["matchup"] = (
        results_df["agent_A_persona"] + " vs " + results_df["agent_B_persona"]
    )

    # Calculate agreement rates and counts
    matchups = results_df["matchup"].unique()
    agreement_data = {}

    for matchup in matchups:
        subset = results_df[results_df["matchup"] == matchup]
        agreement_data[matchup] = subset["agreement"].values

    # Calculate means for plotting
    agreement_rates = {k: v.mean() * 100 for k, v in agreement_data.items()}

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot bars
    x_pos = np.arange(len(matchups))
    bars = ax.bar(x_pos, [agreement_rates[m] for m in matchups], color="steelblue")

    ax.set_title("Agreement Rates with Statistical Significance")
    ax.set_xlabel("Matchup")
    ax.set_ylabel("Agreement Rate (%)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(matchups, rotation=45, ha="right")
    ax.set_ylim(0, 110)

    # Add value labels on bars
    for i, matchup in enumerate(matchups):
        ax.text(i, agreement_rates[matchup] + 2,
               f"{agreement_rates[matchup]:.1f}%",
               ha="center", va="bottom")

    # Add significance annotations between adjacent matchups
    for i in range(len(matchups) - 1):
        group1 = agreement_data[matchups[i]]
        group2 = agreement_data[matchups[i + 1]]

        p_value, marker = compute_statistical_significance(group1, group2)

        if marker != "ns":
            y_max = max(agreement_rates[matchups[i]], agreement_rates[matchups[i + 1]])
            annotation_height = y_max + 8

            # Draw bracket
            x1, x2 = i, i + 1
            ax.plot([x1, x1, x2, x2],
                   [annotation_height - 2, annotation_height,
                    annotation_height, annotation_height - 2],
                   'k-', lw=1)

            # Add significance marker
            ax.text(
                (x1 + x2) / 2, annotation_height + 1,
                marker, ha="center", va="bottom", fontsize=10, fontweight="bold"
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def plot_negotiation_timeline(
    history: List[Dict[str, Any]],
    save_path: Optional[str] = None,
):
    """
    Plot round-by-round negotiation timeline for sequential games (Rubinstein).

    Args:
        history: List of round histories from a single game
                 Each entry should have: {round, proposer, offer, response, payoffs}
        save_path: Path to save figure
    """
    if not history:
        raise ValueError("History is empty")

    rounds = []
    offers = []
    proposers = []
    responses = []

    for entry in history:
        if "round" in entry and "offer" in entry:
            rounds.append(entry["round"])
            offers.append(entry.get("offer", 0))
            proposers.append(entry.get("proposer", "unknown"))
            responses.append(entry.get("response", "pending"))

    if not rounds:
        raise ValueError("No round data found in history")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Offer evolution
    agent_a_offers = [offers[i] if proposers[i] == "agent_A" else None for i in range(len(rounds))]
    agent_b_offers = [offers[i] if proposers[i] == "agent_B" else None for i in range(len(rounds))]

    ax1.plot(rounds, agent_a_offers, 'o-', label="Agent A offers", markersize=8, linewidth=2)
    ax1.plot(rounds, agent_b_offers, 's-', label="Agent B offers", markersize=8, linewidth=2)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Proposer's Share")
    ax1.set_title("Negotiation Timeline: Offers by Round")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Response pattern
    ax2.bar(rounds, [1 if r == "accept" else 0.5 if r == "reject" else 0 for r in responses],
            color=["green" if r == "accept" else "red" if r == "reject" else "gray" for r in responses])
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Response")
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_yticklabels(["Pending", "Reject", "Accept"])
    ax2.set_title("Response Pattern by Round")
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def plot_environment_comparison(
    results_df: pd.DataFrame,
    metric: str = "total_payoff",
    save_path: Optional[str] = None,
):
    """
    Compare metric across all three environments side-by-side.

    Args:
        results_df: DataFrame with experiment results
        metric: Metric to compare ("total_payoff", "agent_A_payoff", etc.)
        save_path: Path to save figure
    """
    if metric not in results_df.columns:
        raise ValueError(f"Metric {metric} not found in results")

    results_df = results_df.copy()
    results_df["matchup"] = (
        results_df["agent_A_persona"] + " vs " + results_df["agent_B_persona"]
    )

    environments = results_df["environment"].unique()

    fig, axes = plt.subplots(1, len(environments), figsize=(6 * len(environments), 6), sharey=True)

    if len(environments) == 1:
        axes = [axes]

    for i, env in enumerate(environments):
        env_data = results_df[results_df["environment"] == env]

        sns.boxplot(
            data=env_data,
            x="matchup",
            y=metric,
            hue="matchup",
            legend=False,
            ax=axes[i],
        )

        axes[i].set_title(f"{env}")
        axes[i].set_xlabel("Matchup")
        axes[i].set_ylabel(metric if i == 0 else "")
        axes[i].tick_params(axis="x", rotation=45)
        plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.suptitle(f"Environment Comparison: {metric}", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
