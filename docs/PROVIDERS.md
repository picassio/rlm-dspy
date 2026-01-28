# RLM-DSPY Provider Support

RLM-DSPY uses [DSPy](https://dspy-docs.vercel.app/) with [LiteLLM](https://docs.litellm.ai/) as the backend, giving you access to **100+ LLM providers** out of the box.

## Quick Start

```bash
# Set your API key (auto-detected from provider-specific env vars)
export OPENAI_API_KEY="sk-..."

# Model format: provider/model-name
rlm-dspy ask "What does this code do?" ./src --model openai/gpt-4o
```

## How It Works

1. **Model format**: `provider/model-name` (e.g., `deepseek/deepseek-chat`)
2. **API key**: Auto-detected from `{PROVIDER}_API_KEY` env var
3. **No api_base needed**: LiteLLM handles routing automatically

---

## Major Providers

### OpenAI
```bash
export OPENAI_API_KEY="sk-..."
rlm-dspy ask "..." ./src --model openai/gpt-4o
rlm-dspy ask "..." ./src --model openai/gpt-4o-mini
rlm-dspy ask "..." ./src --model openai/o1-preview
```

### Anthropic
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
rlm-dspy ask "..." ./src --model anthropic/claude-sonnet-4-20250514
rlm-dspy ask "..." ./src --model anthropic/claude-3-5-sonnet-20241022
rlm-dspy ask "..." ./src --model anthropic/claude-3-haiku-20240307
```

### Google (Gemini)
```bash
export GEMINI_API_KEY="..."
rlm-dspy ask "..." ./src --model gemini/gemini-2.0-flash
rlm-dspy ask "..." ./src --model gemini/gemini-1.5-pro
```

---

## Chinese LLM Providers

### DeepSeek
```bash
export DEEPSEEK_API_KEY="sk-..."
rlm-dspy ask "..." ./src --model deepseek/deepseek-chat
rlm-dspy ask "..." ./src --model deepseek/deepseek-r1        # Reasoning model
rlm-dspy ask "..." ./src --model deepseek/deepseek-coder
```

### Moonshot (Kimi)
```bash
export MOONSHOT_API_KEY="sk-..."
rlm-dspy ask "..." ./src --model moonshot/kimi-latest
rlm-dspy ask "..." ./src --model moonshot/kimi-k2-thinking   # Reasoning model
rlm-dspy ask "..." ./src --model moonshot/moonshot-v1-128k   # Long context
```

### MiniMax
```bash
export MINIMAX_API_KEY="..."
rlm-dspy ask "..." ./src --model minimax/MiniMax-M2.1
rlm-dspy ask "..." ./src --model minimax/MiniMax-M2.1-lightning
```

### Qwen (Alibaba Dashscope)
```bash
export DASHSCOPE_API_KEY="sk-..."
rlm-dspy ask "..." ./src --model dashscope/qwen-max
rlm-dspy ask "..." ./src --model dashscope/qwen-plus
rlm-dspy ask "..." ./src --model dashscope/qwen-turbo
```

### ZhiPu (GLM)

GLM uses OpenAI-compatible API with a custom base URL:

```bash
export OPENAI_API_KEY="your-glm-api-key"

# Standard GLM API
rlm-dspy ask "..." ./src \
    --model openai/glm-4-flash \
    --api-base https://open.bigmodel.cn/api/paas/v4

# GLM Coding Plan (different endpoint!)
rlm-dspy ask "..." ./src \
    --model openai/glm-4.7 \
    --api-base https://api.z.ai/api/coding/paas/v4
```

**Note**: GLM-4.7 is a reasoning model - it may be slower but provides detailed analysis.

---

## Cloud Providers

### AWS Bedrock
```bash
# Uses AWS credentials from environment or ~/.aws/credentials
export AWS_REGION="us-east-1"
rlm-dspy ask "..." ./src --model bedrock/anthropic.claude-3-sonnet-20240229-v1:0
rlm-dspy ask "..." ./src --model bedrock/amazon.nova-pro-v1:0
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
rlm-dspy ask "..." ./src --model openrouter/anthropic/claude-3.5-sonnet
rlm-dspy ask "..." ./src --model openrouter/google/gemini-pro
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

## Environment Variables

| Provider | Environment Variable | Example |
|----------|---------------------|---------|
| OpenAI | `OPENAI_API_KEY` | `sk-...` |
| Anthropic | `ANTHROPIC_API_KEY` | `sk-ant-...` |
| Google | `GEMINI_API_KEY` | `AI...` |
| DeepSeek | `DEEPSEEK_API_KEY` | `sk-...` |
| Moonshot | `MOONSHOT_API_KEY` | `sk-...` |
| MiniMax | `MINIMAX_API_KEY` | `...` |
| Qwen | `DASHSCOPE_API_KEY` | `sk-...` |
| ZhiPu | `ZHIPU_API_KEY` | `...` |
| Groq | `GROQ_API_KEY` | `gsk_...` |
| Together | `TOGETHER_API_KEY` | `...` |
| OpenRouter | `OPENROUTER_API_KEY` | `sk-or-...` |

---

## Programmatic Usage

```python
from rlm_dspy import RLM, RLMConfig

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

# Moonshot/Kimi
rlm = RLM(RLMConfig(
    model="moonshot/kimi-latest",
    api_key="sk-..."
))

# Custom endpoint (GLM, self-hosted, etc.)
rlm = RLM(RLMConfig(
    model="openai/glm-4",
    api_base="https://open.bigmodel.cn/api/paas/v4",
    api_key="..."
))

# Query
result = rlm.query("What does this code do?", code_content)
```

---

## Full Provider List

RLM-DSPY supports all providers available in LiteLLM:

- **Major**: OpenAI, Anthropic, Google (Gemini), Cohere, AI21
- **Chinese**: DeepSeek, Moonshot/Kimi, MiniMax, Qwen/Dashscope, ZhiPu/GLM
- **Cloud**: AWS Bedrock, Azure, Google Vertex AI, Databricks
- **Open Source**: Ollama, vLLM, LM Studio, Hugging Face
- **Aggregators**: OpenRouter, Together AI, Groq, Fireworks, Replicate

For the complete list, see: https://docs.litellm.ai/docs/providers

---

## Troubleshooting

### API Key Not Found
```bash
# Check if key is set
echo $OPENAI_API_KEY

# Set temporarily
OPENAI_API_KEY="sk-..." rlm-dspy ask "..." ./src
```

### Wrong Base URL
```bash
# For OpenAI-compatible APIs, always use openai/ prefix
rlm-dspy ask "..." ./src --model openai/model --api-base https://...
```

### Model Not Found
```bash
# List available models for a provider
python -c "import litellm; print(litellm.deepseek_models)"
```
