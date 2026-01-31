# RLM-DSPY Provider Support

RLM-DSPY supports **32+ models across 9 providers**, with native support for custom API providers and OAuth authentication.

## Quick Start

```bash
# Set your API key (auto-detected from provider-specific env vars)
export KIMI_API_KEY="sk-..."

# Model format: provider/model-name
rlm-dspy ask "What does this code do?" ./src --model kimi/k2p5
```

## How It Works

1. **Model format**: `provider/model-name` (e.g., `kimi/k2p5`)
2. **API key**: Auto-detected from `{PROVIDER}_API_KEY` env var
3. **Custom providers**: Native LMs bypass LiteLLM for OAuth and custom APIs

---

## Provider Benchmark Results

We benchmarked all providers on real code analysis tasks:

| Provider | Model | Accuracy | Hallucination | Avg Time | Notes |
|----------|-------|----------|---------------|----------|-------|
| **Kimi** | k2p5 | **100%** | **0%** | **14.4s** | ⭐ Recommended |
| Kimi | k2-thinking | 100% | 0% | 15.5s | Reasoning model |
| Z.AI | glm-4.7 | 100% | 0% | 44.2s | Slower but accurate |
| MiniMax | M2.1 | ~50% | 0% | ~70s | Timeout issues |

**Recommendation**: Use `kimi/k2p5` as the default model for best speed and accuracy.

---

## OAuth Providers (Free, No API Key Required)

These providers support OAuth authentication - no API key needed:

### Anthropic (Claude Pro/Max)

Requires a Claude Pro or Max subscription at claude.ai.

```bash
# Login with OAuth
rlm-dspy auth login anthropic

# Use Claude models
rlm-dspy ask "..." ./src --model anthropic/claude-sonnet-4-20250514
rlm-dspy ask "..." ./src --model anthropic/claude-3-5-sonnet-20241022
```

### Google Gemini CLI

Uses the same OAuth as the Gemini CLI tool.

```bash
# Login with OAuth
rlm-dspy auth login google-gemini

# Use Gemini models
rlm-dspy ask "..." ./src --model google/gemini-2.0-flash
rlm-dspy ask "..." ./src --model google/gemini-2.5-flash
```

### Antigravity (Experimental)

Google's experimental API with access to latest models.

```bash
# Login with OAuth
rlm-dspy auth login antigravity

# Use Antigravity models
rlm-dspy ask "..." ./src --model antigravity/gemini-2.5-flash
rlm-dspy ask "..." ./src --model antigravity/gemini-3-flash
```

### Managing OAuth

```bash
# Check authentication status
rlm-dspy auth status

# Refresh tokens
rlm-dspy auth refresh anthropic

# Logout
rlm-dspy auth logout google-gemini
```

---

## Native Providers (Custom LMs)

These providers use custom Language Model implementations that bypass LiteLLM for direct API access:

### Kimi (Moonshot) ⭐ Recommended

```bash
export KIMI_API_KEY="sk-..."

# Available models
rlm-dspy ask "..." ./src --model kimi/k2p5           # Fast, accurate
rlm-dspy ask "..." ./src --model kimi/k2-0130        # K2 base
rlm-dspy ask "..." ./src --model kimi/k2-thinking    # Reasoning model
rlm-dspy ask "..." ./src --model kimi/kimi-latest    # Latest stable
```

**Features:**
- Fastest provider (14.4s average)
- 100% accuracy on benchmarks
- 0% hallucination rate
- Uses custom `KimiLM` for direct API access

### MiniMax

```bash
export MINIMAX_API_KEY="..."

# Available models
rlm-dspy ask "..." ./src --model minimax/MiniMax-M2.1
rlm-dspy ask "..." ./src --model minimax/MiniMax-M2.1-lightning
rlm-dspy ask "..." ./src --model minimax/MiniMax-Text-01
```

**Features:**
- Large context window
- Good for long documents
- Uses custom `MiniMaxLM` for direct API access

### Z.AI (GLM Coding)

```bash
export ZAI_API_KEY="..."

# Available models
rlm-dspy ask "..." ./src --model zai/glm-4.7         # Reasoning model
rlm-dspy ask "..." ./src --model zai/glm-4-flash     # Fast model
rlm-dspy ask "..." ./src --model zai/glm-4-plus      # Enhanced
```

**Features:**
- Specialized for coding tasks
- 100% accuracy on benchmarks
- Uses custom `ZAILM` for direct API access

### OpenCode

```bash
export OPENCODE_API_KEY="..."

# Available models
rlm-dspy ask "..." ./src --model opencode/gpt-4o
rlm-dspy ask "..." ./src --model opencode/gpt-4o-mini
rlm-dspy ask "..." ./src --model opencode/claude-3-5-sonnet
```

**Features:**
- OpenAI-compatible endpoint
- Uses custom `OpenCodeLM` for direct API access

---

## LiteLLM Providers

These providers use LiteLLM for API routing:

### OpenAI

```bash
export OPENAI_API_KEY="sk-..."
rlm-dspy ask "..." ./src --model openai/gpt-4o
rlm-dspy ask "..." ./src --model openai/gpt-4o-mini
rlm-dspy ask "..." ./src --model openai/o1-preview
```

### Google (Gemini)

```bash
export GEMINI_API_KEY="..."
rlm-dspy ask "..." ./src --model gemini/gemini-2.0-flash
rlm-dspy ask "..." ./src --model gemini/gemini-1.5-pro
```

### DeepSeek

```bash
export DEEPSEEK_API_KEY="sk-..."
rlm-dspy ask "..." ./src --model deepseek/deepseek-chat
rlm-dspy ask "..." ./src --model deepseek/deepseek-r1        # Reasoning model
rlm-dspy ask "..." ./src --model deepseek/deepseek-coder
```

### Qwen (Alibaba Dashscope)

```bash
export DASHSCOPE_API_KEY="sk-..."
rlm-dspy ask "..." ./src --model dashscope/qwen-max
rlm-dspy ask "..." ./src --model dashscope/qwen-plus
rlm-dspy ask "..." ./src --model dashscope/qwen-turbo
```

---

## Cloud Providers

### AWS Bedrock

```bash
# Uses AWS credentials from environment or ~/.aws/credentials
export AWS_REGION="us-east-1"
rlm-dspy ask "..." ./src --model bedrock/amazon.nova-pro-v1:0
rlm-dspy ask "..." ./src --model bedrock/amazon.titan-text-express-v1
```

### Azure OpenAI

```bash
export AZURE_API_KEY="..."
export AZURE_API_BASE="https://your-resource.openai.azure.com"
export AZURE_API_VERSION="2024-02-15-preview"
rlm-dspy ask "..." ./src --model azure/your-deployment-name
```

### Google Vertex AI

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
export VERTEXAI_PROJECT="your-project"
export VERTEXAI_LOCATION="us-central1"
rlm-dspy ask "..." ./src --model vertex_ai/gemini-1.5-pro
```

---

## Open Source / Self-Hosted

### Ollama

```bash
# Start Ollama locally
ollama serve

# Use local models
rlm-dspy ask "..." ./src --model ollama/llama3.2
rlm-dspy ask "..." ./src --model ollama/codellama
rlm-dspy ask "..." ./src --model ollama/deepseek-r1:14b
```

### vLLM

```bash
export VLLM_API_BASE="http://localhost:8000"
rlm-dspy ask "..." ./src --model hosted_vllm/meta-llama/Llama-3-70b
```

### LM Studio

```bash
rlm-dspy ask "..." ./src \
    --model openai/local-model \
    --api-base http://localhost:1234/v1
```

---

## API Aggregators

### OpenRouter

```bash
export OPENROUTER_API_KEY="sk-or-..."
rlm-dspy ask "..." ./src --model openrouter/google/gemini-pro
rlm-dspy ask "..." ./src --model openrouter/deepseek/deepseek-chat
```

### Together AI

```bash
export TOGETHER_API_KEY="..."
rlm-dspy ask "..." ./src --model together_ai/meta-llama/Llama-3-70b-chat-hf
```

### Groq

```bash
export GROQ_API_KEY="..."
rlm-dspy ask "..." ./src --model groq/llama-3.1-70b-versatile
rlm-dspy ask "..." ./src --model groq/mixtral-8x7b-32768
```

### Fireworks AI

```bash
export FIREWORKS_API_KEY="..."
rlm-dspy ask "..." ./src --model fireworks_ai/accounts/fireworks/models/llama-v3-70b-instruct
```

---

## Environment Variables

| Provider | Environment Variable | Example |
|----------|---------------------|---------|
| **Kimi** | `KIMI_API_KEY` | `sk-...` |
| **MiniMax** | `MINIMAX_API_KEY` | `...` |
| **Z.AI** | `ZAI_API_KEY` | `...` |
| **OpenCode** | `OPENCODE_API_KEY` | `...` |
| OpenAI | `OPENAI_API_KEY` | `sk-...` |
| Google | `GEMINI_API_KEY` | `AI...` |
| DeepSeek | `DEEPSEEK_API_KEY` | `sk-...` |
| Qwen | `DASHSCOPE_API_KEY` | `sk-...` |
| Groq | `GROQ_API_KEY` | `gsk_...` |
| Together | `TOGETHER_API_KEY` | `...` |
| OpenRouter | `OPENROUTER_API_KEY` | `sk-or-...` |

---

## Using Custom Base URLs

For any OpenAI-compatible API, use the `openai/` prefix with `--api-base`:

```bash
# Generic pattern
rlm-dspy ask "..." ./src \
    --model openai/model-name \
    --api-base https://your-api-endpoint/v1

# Example: Self-hosted OpenAI-compatible server
rlm-dspy ask "..." ./src \
    --model openai/gpt-4 \
    --api-base http://localhost:8080/v1
```

---

## Programmatic Usage

```python
from rlm_dspy import RLM, RLMConfig

# Kimi (Recommended)
rlm = RLM(RLMConfig(
    model="kimi/k2p5",
    api_key="sk-..."
))

# OpenAI
rlm = RLM(RLMConfig(
    model="openai/gpt-4o",
    api_key="sk-..."
))

# DeepSeek
rlm = RLM(RLMConfig(
    model="deepseek/deepseek-chat",
    api_key="sk-..."
))

# Custom endpoint
rlm = RLM(RLMConfig(
    model="openai/glm-4",
    api_base="https://open.bigmodel.cn/api/paas/v4",
    api_key="..."
))

# Query
result = rlm.query("What does this code do?", code_content)
```

---

## List Available Models

```bash
# List all registered models
rlm-dspy models list

# Filter by provider
rlm-dspy models list --provider kimi

# Show model details
rlm-dspy models info kimi/k2p5
```

---

## Full Provider List

RLM-DSPY supports:

- **Native Providers**: Kimi, MiniMax, Z.AI, OpenCode (custom LMs)
- **Major**: OpenAI, Google (Gemini), Cohere, AI21
- **Chinese**: DeepSeek, Qwen/Dashscope
- **Cloud**: AWS Bedrock, Azure, Google Vertex AI, Databricks
- **Open Source**: Ollama, vLLM, LM Studio, Hugging Face
- **Aggregators**: OpenRouter, Together AI, Groq, Fireworks, Replicate

For LiteLLM providers, see: https://docs.litellm.ai/docs/providers

---

## Troubleshooting

### API Key Not Found

```bash
# Check if key is set
echo $KIMI_API_KEY

# Set temporarily
KIMI_API_KEY="sk-..." rlm-dspy ask "..." ./src
```

### Wrong Base URL

```bash
# For OpenAI-compatible APIs, always use openai/ prefix
rlm-dspy ask "..." ./src --model openai/model --api-base https://...
```

### Model Not Found

```bash
# List available models
rlm-dspy models list

# Check if provider is registered
python -c "from rlm_dspy.core.models import get_model_registry; print(get_model_registry().list_providers())"
```

### Timeout Issues

```bash
# Increase timeout for slow providers
rlm-dspy ask "..." ./src --model minimax/M2.1 --timeout 300
```
