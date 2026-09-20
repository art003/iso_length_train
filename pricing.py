# тарифы 18.09.2026, $ за 1M токенов
# https://ai.google.dev/gemini-api/docs/pricing
# https://openai.com/api/pricing

from llm_client import LlmUsage

PRICES_USD_PER_M = {
    "gpt-5-codex": (1.25, 10.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
}


def estimate_cost_usd(usage: LlmUsage) -> float:
    key = usage.model.lower()
    if ":free" in key or usage.provider in ("openrouter", "groq", "ollama"):
        return 0.0
    pair = PRICES_USD_PER_M.get(key, (0.30, 2.50))
    for name, p in PRICES_USD_PER_M.items():
        if name in key:
            pair = p
            break
    inn, out = pair
    return usage.input_tokens * inn / 1_000_000 + usage.output_tokens * out / 1_000_000
