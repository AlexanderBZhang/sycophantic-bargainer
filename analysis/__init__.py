"""
Tournament Analysis Module

Comprehensive analysis tools for the sycophantic bargainer experiment.

Modules:
- language_markers: Detect sycophantic language patterns (deference, softening)
- concession_dynamics: Analyze negotiation trajectories (who concedes first/more)
- reasoning_analysis: Check if models reason about incentives vs norms
- sycophancy_metrics: Aggregate sycophancy metrics (SI, CI, EMS)
"""

from .language_markers import (
    analyze_transcript_language,
    analyze_player_language,
    compare_player_language,
    extract_language_metrics,
    LanguageAnalysis,
)

from .concession_dynamics import (
    analyze_concession_dynamics,
    calculate_concession_index,
    calculate_first_mover_advantage,
    extract_concession_metrics,
    ConcessionAnalysis,
)

from .reasoning_analysis import (
    analyze_reasoning_content,
    analyze_api_call_reasoning,
    compare_reasoning_to_behavior,
    extract_reasoning_metrics,
    ReasoningAnalysis,
)

from .sycophancy_metrics import (
    analyze_single_match,
    aggregate_model_metrics,
    compute_exploitability,
    generate_sycophancy_report,
    PlayerMatchMetrics,
    ModelTournamentMetrics,
)

from .process_outcome_analysis import (
    analyze_negotiation,
    compute_process_outcome_correlation,
    identify_mismatch_candidates,
    counterparty_sensitivity_analysis,
    outcome_conditional_language_analysis,
    generate_process_outcome_report,
    NegotiationAnalysis,
    RoundAnalysis,
)

from .canonical_markers import (
    CATEGORY_1_APOLOGIES,
    CATEGORY_2_DEFERENCE,
    CATEGORY_3_HEDGES,
    CATEGORY_4_FLATTERY,
    ALL_MARKERS,
    ENRON_BASELINE,
    count_markers,
    count_all_markers,
    calculate_rates_per_1k,
    analyze_text,
    MarkerAnalysis,
)

__all__ = [
    # Language Analysis
    "analyze_transcript_language",
    "analyze_player_language",
    "compare_player_language",
    "extract_language_metrics",
    "LanguageAnalysis",
    # Concession Dynamics
    "analyze_concession_dynamics",
    "calculate_concession_index",
    "calculate_first_mover_advantage",
    "extract_concession_metrics",
    "ConcessionAnalysis",
    # Reasoning Analysis
    "analyze_reasoning_content",
    "analyze_api_call_reasoning",
    "compare_reasoning_to_behavior",
    "extract_reasoning_metrics",
    "ReasoningAnalysis",
    # Sycophancy Metrics
    "analyze_single_match",
    "aggregate_model_metrics",
    "compute_exploitability",
    "generate_sycophancy_report",
    "PlayerMatchMetrics",
    "ModelTournamentMetrics",
    # Process-Outcome Analysis
    "analyze_negotiation",
    "compute_process_outcome_correlation",
    "identify_mismatch_candidates",
    "counterparty_sensitivity_analysis",
    "outcome_conditional_language_analysis",
    "generate_process_outcome_report",
    "NegotiationAnalysis",
    "RoundAnalysis",
    # Canonical Markers
    "CATEGORY_1_APOLOGIES",
    "CATEGORY_2_DEFERENCE",
    "CATEGORY_3_HEDGES",
    "CATEGORY_4_FLATTERY",
    "ALL_MARKERS",
    "ENRON_BASELINE",
    "count_markers",
    "count_all_markers",
    "calculate_rates_per_1k",
    "analyze_text",
    "MarkerAnalysis",
]
