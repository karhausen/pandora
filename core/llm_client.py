from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str


class LLMClient:
    """Minimal LLM facade. MVP uses safe stub; Ollama/OpenAI adapters come next."""

    def __init__(self, provider: str = "stub"):
        self.provider = provider

    def generate(self, prompt: str, complexity: str = "simple") -> LLMResponse:
        if self.provider == "stub":
            return LLMResponse(
                text=f"Stub-Plan: Aufgabe analysiert. Komplexität={complexity}. Prompt-Auszug={prompt[:120]}",
                provider="stub",
                model="none",
            )
        raise NotImplementedError(f"Provider not implemented in MVP1: {self.provider}")

    def healthcheck(self) -> bool:
        return bool(self.generate("ping").text)
