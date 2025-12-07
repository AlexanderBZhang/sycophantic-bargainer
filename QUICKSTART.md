# Quick Start Guide

## The Sycophantic Bargainer - Getting Started

This guide will help you run your first experiments in under 10 minutes.

## Prerequisites

- Python 3.8+
- Virtual environment (recommended)

## Installation

### 1. Set up environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up API keys (for LLM experiments)

```bash
# For OpenAI
export OPENAI_API_KEY="your-key-here"

# OR for Anthropic
export ANTHROPIC_API_KEY="your-key-here"
```

## Running Your First Experiment

### Option 1: Simple Demo (No API Required)

The fastest way to see the framework in action:

```bash
python simple_demo.py
```

This runs:
- Nash Demand Game
- Ultimatum Game
- Rubinstein Bargaining

All using MockAgents (no API calls, no external dependencies).

### Option 2: Batch Experiments with Mock Agents

Run comprehensive experiments across all environments and matchups:

```bash
python -m sycophantic_bargainer.experiments.run_mock
```

This runs 600+ experiments and generates CSV results. No API keys required.

### Option 3: Validation Script

Quick smoke test to verify everything works:

```bash
python validate.py
```

### Option 4: Real LLM Experiments

Create a file `run_experiment.py`:

```python
from sycophantic_bargainer.environments import NashDemandGame
from sycophantic_bargainer.agents import LLMAgent
from sycophantic_bargainer.experiments import ExperimentRunner

# Create environment
env = NashDemandGame(pie_size=10.0)

# Create LLM agents
sycophantic_agent = LLMAgent(
    agent_id="agent_A",
    persona_type="sycophantic",
    model="gpt-3.5-turbo",
    provider="openai",
)

rational_agent = LLMAgent(
    agent_id="agent_B",
    persona_type="rational",
    model="gpt-3.5-turbo",
    provider="openai",
)

# Run experiment
runner = ExperimentRunner(
    environment=env,
    agent_A=sycophantic_agent,
    agent_B=rational_agent,
    num_simulations=50,
    verbose=True,
)

results = runner.run()
runner.print_summary()

# Save results
runner.save_results("results.csv")
```

Then run:

```bash
python run_experiment.py
```

## Project Structure

```
sycophantic_bargainer/
├── environments/       # Game implementations
│   ├── nash_demand.py
│   ├── ultimatum.py
│   └── rubinstein.py
├── agents/            # Agent implementations
│   ├── llm_agent.py   # LLM-backed agents
│   └── personas.py    # Persona prompts
├── metrics/           # Evaluation metrics
│   ├── utility.py
│   ├── fairness.py
│   └── convergence.py
├── experiments/       # Experiment runners
│   └── runner.py
└── utils/            # Visualization tools
    └── visualization.py
```

## Running Tests

```bash
pytest
```

Or with coverage:

```bash
pytest --cov=sycophantic_bargainer
```

## Key Concepts

### Environments

Three bargaining games:
1. **Nash Demand Game**: Simultaneous demands
2. **Ultimatum Game**: Sequential offer/response
3. **Rubinstein Bargaining**: Alternating offers with discounting

### Agent Personas

- **Sycophantic**: Agreeable, wants to be liked
- **Rational**: Maximizes personal payoff
- **Control**: Neutral baseline
- **Fairness**: Prioritizes equitable outcomes

### Metrics

- **Utility**: Average payoffs, utility gaps
- **Fairness**: Gini coefficient, Nash product
- **Convergence**: Agreement speed, equilibrium deviation

## Example Experiment Grid

Run all combinations:

```python
from sycophantic_bargainer.experiments import BatchExperimentRunner
from sycophantic_bargainer.environments import *
from sycophantic_bargainer.agents import MockAgent

# Create environments
environments = [
    NashDemandGame(pie_size=10.0),
    UltimatumGame(pie_size=10.0),
    RubinsteinBargaining(pie_size=10.0),
]

# Create agent pairs
agent_pairs = [
    (MockAgent("A", "sycophantic", "generous"),
     MockAgent("B", "rational", "greedy")),
    (MockAgent("A", "rational", "equal_split"),
     MockAgent("B", "rational", "equal_split")),
    (MockAgent("A", "sycophantic", "generous"),
     MockAgent("B", "sycophantic", "generous")),
]

# Run batch
runner = BatchExperimentRunner()
results_df = runner.run_experiment_grid(
    environments=environments,
    agent_pairs=agent_pairs,
    num_simulations=100,
)

# Save and visualize
results_df.to_csv("all_results.csv")
```

## Visualization

```python
from sycophantic_bargainer.utils import plot_all_results
import pandas as pd

# Load results
df = pd.read_csv("results.csv")

# Generate all plots
plot_all_results(df, output_dir="./plots")
```

## Next Steps

1. **Modify personas**: Edit `agents/personas.py` to test different prompts
2. **Add new metrics**: Extend `metrics/` modules
3. **Custom environments**: Inherit from `BargainEnvironment`
4. **Try different LLMs**: Use Claude, Llama, or other models

## Troubleshooting

### Import errors
```bash
# Make sure you're in the project root
cd /workspace
export PYTHONPATH=/workspace:$PYTHONPATH
```

### API errors
- Check your API key is set
- Verify you have credits/quota
- Try reducing `num_simulations` for testing

### Dependency issues
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## Support

- Check `PROJECT_PLAN.md` for detailed architecture
- See `TODO.md` for development roadmap
- Review tests in `sycophantic_bargainer/tests/` for examples

## Research Context

This project tests the hypothesis:

> **Do RLHF-trained LLMs exhibit suboptimal bargaining behavior due to sycophancy?**

By comparing sycophantic agents (trained to be agreeable) vs rational agents (trained to maximize payoff), we can measure:
- Utility gaps
- Exploitability
- Deviations from game-theoretic equilibria

This connects to broader AI safety research on cooperative AI, multi-agent systems, and incentive alignment.
