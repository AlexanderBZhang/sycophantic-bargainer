# API Setup Guide: Sycophantic Bargainer

This guide explains how to configure API keys for running LLM experiments.

---

## Supported LLM Providers

The Sycophantic Bargainer framework supports:
1. **OpenAI** 
2. **Anthropic**

You only need API keys for the provider(s) you intend to use.

---

## Option 1: OpenAI Setup

### Step 1: Get an API Key

1. Create an account at https://platform.openai.com/
2. Navigate to: **Settings** → **API Keys**
3. Click **"Create new secret key"**
4. Copy the key (starts with `sk-...`)

### Step 2: Set Environment Variable

**Linux/macOS:**
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

To make it permanent, add to `~/.bashrc` or `~/.zshrc`:
```bash
echo 'export OPENAI_API_KEY="sk-your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

**Windows (Command Prompt):**
```cmd
setx OPENAI_API_KEY "sk-your-key-here"
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="sk-your-key-here"
```

### Step 3: Verify Setup

```python
from sycophantic_bargainer.agents import LLMAgent

agent = LLMAgent(
    agent_id="test",
    persona_type="rational",
    provider="openai",
    model="gpt-3.5-turbo"
)
print("✓ OpenAI configured successfully!")
```

---

## Option 2: Anthropic Setup

### Step 1: Get an API Key

1. Create an account at https://console.anthropic.com/
2. Navigate to: **Settings** → **API Keys**
3. Click **"Create Key"**
4. Copy the key (starts with `sk-ant-...`)

### Step 2: Set Environment Variable

**Linux/macOS:**
```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

To make it permanent:
```bash
echo 'export ANTHROPIC_API_KEY="sk-ant-your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

**Windows (Command Prompt):**
```cmd
setx ANTHROPIC_API_KEY "sk-ant-your-key-here"
```

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

### Step 3: Verify Setup

```python
from sycophantic_bargainer.agents import LLMAgent

agent = LLMAgent(
    agent_id="test",
    persona_type="sycophantic",
    provider="anthropic",
    model="claude-3-haiku-20240307"
)
print("✓ Anthropic configured successfully!")
```

---

## Cost Estimates

### OpenAI Pricing (as of 2024)

| Model | Input Cost | Output Cost | Est. Cost per Game* |
|-------|------------|-------------|---------------------|
| GPT-3.5-turbo | $0.50 / 1M tokens | $1.50 / 1M tokens | $0.001 - $0.003 |
| GPT-4-turbo | $10.00 / 1M tokens | $30.00 / 1M tokens | $0.02 - $0.05 |
| GPT-4 | $30.00 / 1M tokens | $60.00 / 1M tokens | $0.06 - $0.15 |

### Anthropic Pricing (as of 2024)

| Model | Input Cost | Output Cost | Est. Cost per Game* |
|-------|------------|-------------|---------------------|
| Claude 3 Haiku | $0.25 / 1M tokens | $1.25 / 1M tokens | $0.0005 - $0.002 |
| Claude 3 Sonnet | $3.00 / 1M tokens | $15.00 / 1M tokens | $0.01 - $0.03 |
| Claude 3 Opus | $15.00 / 1M tokens | $75.00 / 1M tokens | $0.05 - $0.12 |

*\*Cost per game depends on game type and number of rounds. Estimates assume typical prompts (~300 tokens input, ~100 tokens output per turn).*

### Experiment Cost Estimates

For **100 simulations** per configuration:

| Model | Nash Demand | Ultimatum | Rubinstein (10 rounds) | Full Suite** |
|-------|-------------|-----------|------------------------|--------------|
| GPT-3.5-turbo | $0.15 | $0.20 | $1.50 | $5.50 |
| GPT-4-turbo | $3.00 | $4.00 | $30.00 | $110.00 |
| Claude Haiku | $0.10 | $0.15 | $1.00 | $3.75 |
| Claude Sonnet | $1.50 | $2.00 | $15.00 | $55.00 |

**\*\*Full suite = 3 environments × 3 matchups × 100 runs = 900 games**

---

## Recommended Starting Configuration

For **initial testing** (validating LLM parsing works):
- **Provider**: OpenAI
- **Model**: `gpt-3.5-turbo`
- **Runs**: 10 per configuration
- **Estimated cost**: < $1.00

For **full experiments**:
- **Provider**: Anthropic (better instruction-following for structured tasks)
- **Model**: `claude-3-haiku-20240307` (fast and cheap)
- **Runs**: 100 per configuration
- **Estimated cost**: ~$3.75

For **publication-quality** results:
- **Provider**: Anthropic
- **Model**: `claude-3-sonnet-20240229`
- **Runs**: 100+ per configuration
- **Estimated cost**: ~$55+

---

## Usage in Code

### Example 1: Single LLM Agent

```python
from sycophantic_bargainer.environments import NashDemandGame
from sycophantic_bargainer.agents import LLMAgent
from sycophantic_bargainer.experiments import ExperimentRunner

# Create environment
env = NashDemandGame(pie_size=10.0)

# Create LLM agents
agent_A = LLMAgent(
    agent_id="agent_A",
    persona_type="rational",
    provider="openai",
    model="gpt-3.5-turbo",
    temperature=0.7
)

agent_B = LLMAgent(
    agent_id="agent_B",
    persona_type="sycophantic",
    provider="openai",
    model="gpt-3.5-turbo",
    temperature=0.7
)

# Run experiment
runner = ExperimentRunner(
    environment=env,
    agent_A=agent_A,
    agent_B=agent_B,
    num_simulations=10,
    verbose=True
)

results = runner.run()
runner.print_summary()
```

### Example 2: Mixed Provider Experiment

```python
# OpenAI rational agent vs Anthropic sycophantic agent
agent_A = LLMAgent(
    agent_id="agent_A",
    persona_type="rational",
    provider="openai",
    model="gpt-4-turbo"
)

agent_B = LLMAgent(
    agent_id="agent_B",
    persona_type="sycophantic",
    provider="anthropic",
    model="claude-3-sonnet-20240229"
)
```

---

## Troubleshooting

### Error: "OpenAI API key not found"

**Cause**: Environment variable not set.

**Solution**:
```bash
# Check if set
echo $OPENAI_API_KEY

# If empty, set it
export OPENAI_API_KEY="sk-your-key-here"
```

### Error: "Anthropic API key not found"

**Cause**: Environment variable not set.

**Solution**:
```bash
# Check if set
echo $ANTHROPIC_API_KEY

# If empty, set it
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

### Error: "Rate limit exceeded"

**Cause**: Too many API requests in short time.

**Solution**:
- Reduce `num_simulations`
- Add delays between requests
- Upgrade API tier (if available)

### Error: "Invalid model name"

**Cause**: Typo in model name or deprecated model.

**Valid models**:
- OpenAI: `gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo`, `gpt-4-turbo-preview`
- Anthropic: `claude-3-haiku-20240307`, `claude-3-sonnet-20240229`, `claude-3-opus-20240229`

---

## Security Best Practices

1. **Never commit API keys** to version control
   - Already added to `.gitignore`: `.env`, `*.key`, `secrets/`

2. **Use environment variables** instead of hardcoding

3. **Rotate keys regularly** (monthly recommended)

4. **Set spending limits** in provider dashboard

5. **Monitor usage** via provider console

---

## Alternative: Using .env File

Instead of setting environment variables manually, you can use a `.env` file:

### Step 1: Create `.env` file

```bash
# .env (in project root)
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
```

### Step 2: Load with python-dotenv

```bash
pip install python-dotenv
```

```python
from dotenv import load_dotenv
load_dotenv()  # Loads variables from .env

# Now API keys are available
from sycophantic_bargainer.agents import LLMAgent
agent = LLMAgent(...) # Will automatically find keys
```

---

## Next Steps

After configuring API keys:

1. **Test with small experiment**: `python run_small_llm_test.py` (if available)
2. **Check costs in dashboard** after test run
3. **Proceed to full experiments** if costs acceptable
4. **Monitor rate limits** during batch runs

---

## Support

- OpenAI docs: https://platform.openai.com/docs
- Anthropic docs: https://docs.anthropic.com/
- Project issues: See `TASKS.md` or contact maintainer
