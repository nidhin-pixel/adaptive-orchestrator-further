# Adaptive LLM Orchestrator: Provider & Analytics Modules

> **HackCBS 9.0 Project**  
> Production-ready, asynchronous LLM provider adapters, offline token counter with fallback heuristics, dynamic per-million cost tracking, and in-memory telemetry metrics.

---

## 1. Project Purpose

This repository implements the **Provider & Analytics Modules** for the **Adaptive LLM Orchestrator** architecture at HackCBS 9.0. It provides:
1. **ModelProvider Abstraction**: A unified, asynchronous provider interface compatible with the central orchestrator's `ModelDescriptor` and `ExecutionResult` contracts.
2. **Provider Adapters**:
   - `Provider A` (`OpenAICompatibleProvider`): Supports OpenAI and all OpenAI-compatible endpoints (Groq, DeepSeek, vLLM, Ollama, Together AI).
   - `Provider B` (`AnthropicProvider`): Native adapter for Anthropic Claude Messages API.
3. **Token Counter (`token_counter.py`)**: Uses `tiktoken` when available with a robust, documented heuristic fallback (~4 chars/token, word balanced) when offline or encountering unsupported model architectures.
4. **Cost Tracker (`cost_tracker.py`)**: Configurable pricing per million input/output tokens, with per-model pricing registries and safe missing-pricing handling without hardcoded assumptions.
5. **Metrics Collector (`metrics.py`)**: In-memory telemetry engine tracking request counts, success rates, average latencies, token consumption, and per-model breakdowns.

---

## 2. Folder Structure

```
adaptive-orchestrator-further/
├── app/
│   ├── __init__.py
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py              # ModelDescriptor, ExecutionResult, ModelProvider ABC, exceptions
│   │   ├── provider_a.py        # OpenAI-compatible async adapter (Provider A)
│   │   └── provider_b.py        # Anthropic Claude Messages async adapter (Provider B)
│   ├── utils/
│   │   ├── __init__.py
│   │   └── token_counter.py     # tiktoken + heuristic fallback tokenizer
│   └── analytics/
│       ├── __init__.py
│       ├── cost_tracker.py      # Configurable per-million token pricing & ledger
│       └── metrics.py           # Latency, success rate, and token telemetry
├── tests/
│   ├── __init__.py
│   ├── test_providers.py        # 100% offline tests with httpx.MockTransport
│   ├── test_token_counter.py    # Tiktoken, fallback heuristic, and edge-case tests
│   └── test_analytics.py        # Cost calculation and metrics telemetry tests
├── .env.example                 # Template for API keys and endpoint configuration
├── .gitignore                   # Python, virtualenv, and test cache exclusions
├── requirements.txt             # Minimal, production dependencies
├── README.md                    # Module documentation & quickstart
└── walkthrough.md               # Integration guide for Akash's main orchestrator
```

---

## 3. Installation Instructions (Windows PowerShell)

Ensure Python 3.10+ is installed on your Windows machine:

```powershell
# Navigate into the project folder
cd adaptive-orchestrator-further

# (Recommended) Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

---

## 4. Environment Setup

Copy `.env.example` to create your local `.env`:

```powershell
Copy-Item .env.example .env
```

Set your API keys and configuration in `.env`:

```ini
# Primary Provider (OpenAI Compatible)
LLM_API_KEY=your_openai_or_groq_or_deepseek_api_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_TIMEOUT=30.0

# Secondary Provider (Anthropic)
ANTHROPIC_API_KEY=your_anthropic_api_key
ANTHROPIC_BASE_URL=https://api.anthropic.com
PROVIDER_B_TIMEOUT=30.0
```

> **Security Note:** Secrets are never hardcoded. The application reads from environment variables, constructor parameters, or `.env`.

---

## 5. How to Run Tests

All unit tests are **100% offline**, require **no paid API keys**, and use `httpx.MockTransport` to simulate responses, errors, and timeouts.

Run the complete test suite:

```powershell
pytest -v
```

Run specific test modules:

```powershell
pytest tests/test_providers.py -v
pytest tests/test_token_counter.py -v
pytest tests/test_analytics.py -v
```

---

## 6. Usage Examples

### A. Provider Execution

```python
import asyncio
from app.providers import OpenAICompatibleProvider, ModelDescriptor

async def main():
    # 1. Define model descriptor matching orchestrator contract
    model = ModelDescriptor(
        model_id="gpt-4o",
        provider="openai",
        tier="flagship",
        supported_capabilities=["chat", "tools"],
        context_window=128000,
        input_price_per_million=2.50,
        output_price_per_million=10.00,
        enabled=True,
    )

    # 2. Instantiate provider
    provider = OpenAICompatibleProvider()

    # 3. Call model with prompt and optional context
    context = {
        "system_prompt": "You are an expert AI orchestrator assistant.",
        "temperature": 0.3,
        "max_tokens": 500,
    }

    result = await provider.call(
        prompt="Explain adaptive LLM orchestration in two sentences.",
        model=model,
        context=context,
    )

    if result.success:
        print("Response:", result.content)
        print(f"Latency: {result.latency_ms} ms")
        print(f"Tokens: prompt={result.prompt_tokens}, completion={result.completion_tokens}")
    else:
        print("Call failed:", result.error)

if __name__ == "__main__":
    asyncio.run(main())
```

### B. Token Counting

```python
from app.utils.token_counter import count_tokens, count_prompt_and_completion_tokens

prompt = "Analyze the latency profile of the secondary provider."
completion = "The secondary provider shows 120ms average response time."

# Count prompt & completion
p_tokens, c_tokens = count_prompt_and_completion_tokens(prompt, completion, model_name="gpt-4o")
print(f"Prompt tokens: {p_tokens}, Completion tokens: {c_tokens}")

# Safe handling of empty or non-string inputs
print(count_tokens(""))    # 0
print(count_tokens(None))  # 0
```

> **Authoritative Billing Note:** Pre-execution token counts produced by `token_counter` are estimates for routing decisions and context limits. Provider-reported usage (`result.prompt_tokens`, `result.completion_tokens`) is authoritative for final accounting.

### C. Cost Tracking

```python
from app.analytics import CostTracker, calculate_cost
from app.providers import ModelDescriptor, ExecutionResult

# Direct calculation
breakdown = calculate_cost(
    prompt_tokens=1500,
    completion_tokens=600,
    input_price_per_million=2.50,
    output_price_per_million=10.00,
    model_id="gpt-4o",
)
print(f"Total cost: ${breakdown.total_cost:.6f}")

# Continuous tracking across orchestrator requests
tracker = CostTracker()
model = ModelDescriptor(
    model_id="gpt-4o",
    provider="openai",
    input_price_per_million=2.50,
    output_price_per_million=10.00,
)
tracker.register_model_descriptor(model)

result = ExecutionResult.create_success(
    content="Hello!",
    model_id="gpt-4o",
    latency_ms=85.0,
    prompt_tokens=1200,
    completion_tokens=400,
)
tracker.record_execution(result, model=model)

summary = tracker.get_summary()
print(f"Accumulated Cost: ${summary.total_cost:.6f}")
print(f"Total Tokens: {summary.total_tokens}")
```

### D. Telemetry & Metrics Collection

```python
from app.analytics import MetricsCollector
from app.providers import ExecutionResult

collector = MetricsCollector()

# Record successful execution
res1 = ExecutionResult.create_success(
    content="Result 1",
    model_id="gpt-4o",
    latency_ms=120.0,
    prompt_tokens=50,
    completion_tokens=100,
)
collector.record_execution(res1)

# Record failed execution
res2 = ExecutionResult.create_failure(
    model_id="claude-3-5-sonnet",
    error="Rate limit exceeded",
    latency_ms=25.0,
)
collector.record_execution(res2)

# Query snapshot
snapshot = collector.get_metrics()
print(f"Total Requests: {snapshot.total_requests}")
print(f"Success Rate: {snapshot.success_rate * 100:.1f}%")
print(f"Average Latency: {snapshot.avg_latency_ms} ms")
print(f"Per-Model Breakdown: {snapshot.per_model}")
```

---

## 7. Known Limitations & Extensibility

1. **In-Memory Telemetry**: `MetricsCollector` and `CostTracker` store counters in memory. For distributed production clusters, attach a persistent store (e.g., Redis via Nidhi's module) or Prometheus exporter.
2. **Streaming Completions**: Current implementation focuses on standard single-turn completion calls. Streaming SSE response chunks can be added by implementing an async generator `stream(...)` on `ModelProvider`.
3. **Provider-Specific Tool/Function Calling**: Adapters accept general JSON context and prompts. Native tool schemas (OpenAI `tools` parameter / Anthropic `tools` block) can be extended via the context dictionary.
