# Usage:
#   chmod +x cleanup_for_submission.sh
#   ./cleanup_for_submission.sh

set -e

echo "========================================"
echo "Repository Organization"
echo "========================================"
echo ""

# Create organized folder structure
mkdir -p scripts/experiment
mkdir -p scripts/batch
mkdir -p scripts/preprocess
mkdir -p scripts/analysis

echo "Moving experiment runner scripts to scripts/experiment/..."
for f in run_hardball_batch.py run_hardball_experiment.py \
         run_semantic_analysis.py run_h2h_tournament.py \
         run_realtime_tournament.py run_battle_of_giants.py \
         batch_tournament.py hybrid_batch_tournament.py; do
    if [ -f "$f" ]; then
        mv "$f" scripts/experiment/
        echo "  → scripts/experiment/$f"
    fi
done

echo ""
echo "Moving batch API scripts to scripts/batch/..."
for f in submit_casino_batch.py submit_casino_remaining_batch.py \
         submit_claude_analysis_batch.py submit_enron_remaining_batch.py \
         download_and_merge_batch.py download_and_merge_casino.py \
         download_casino_batch.py download_claude_batches.py; do
    if [ -f "$f" ]; then
        mv "$f" scripts/batch/
        echo "  → scripts/batch/$f"
    fi
done

echo ""
echo "Moving preprocessing scripts to scripts/preprocess/..."
for f in preprocess_enron.py preprocess_casino.py \
         parse_enron_baseline.py parse_enron_fast.py; do
    if [ -f "$f" ]; then
        mv "$f" scripts/preprocess/
        echo "  → scripts/preprocess/$f"
    fi
done

echo ""
echo "Moving analysis scripts to scripts/analysis/..."
for f in analyze_casino_sample.py analyze_full_casino.py analyze_results.py \
         run_significance_tests.py run_full_language_analysis.py \
         generate_semantic_report.py generate_figures.py; do
    if [ -f "$f" ]; then
        mv "$f" scripts/analysis/
        echo "  → scripts/analysis/$f"
    fi
done

echo ""
echo "Moving utility scripts to scripts/..."
for f in extract_titles.py retry_failed_entry.py \
         estimate_claude_analysis_cost.py explore_casino.py; do
    if [ -f "$f" ]; then
        mv "$f" scripts/
        echo "  → scripts/$f"
    fi
done

echo ""
echo "Moving stale state files to archive/..."
mkdir -p archive
for f in batch_state.json enron_checkpoint.json; do
    if [ -f "$f" ]; then
        mv "$f" archive/
        echo "  → archive/$f"
    fi
done

echo ""
echo "Moving reference data files to archive/..."
for f in "Gemini Deep Research Enron Employee Titles.csv" \
         "lmarena_expert_elo (12-4-2025).csv" \
         "ENRON_EMPLOYEES.txt" \
         "LLM Pricing.txt" "LLM Rate Limits.txt"; do
    if [ -f "$f" ]; then
        mv "$f" archive/
        echo "  → archive/$f"
    fi
done

echo ""
echo "========================================"
echo "Organization Complete!"
echo "========================================"
echo ""
echo "Root directory now contains:"
ls -1 *.py 2>/dev/null | head -10 || echo "  (no .py files)"
echo ""
echo "scripts/ structure:"
find scripts -type f -name "*.py" 2>/dev/null | head -20 || echo "  (empty)"
echo ""
echo "To undo: git checkout ."
