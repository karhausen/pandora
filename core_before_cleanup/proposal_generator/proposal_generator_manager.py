from __future__ import annotations

from typing import Any

from .proposal_generator import ProposalGenerator


class ProposalGeneratorManager:
    """Facade for CLI/API integration of MVP 29.0 Proposal Generator."""

    VERSION = "29.0"

    def __init__(self, generator: ProposalGenerator | None = None) -> None:
        self.generator = generator or ProposalGenerator()

    def status(self) -> dict[str, Any]:
        return self.generator.status()

    def prompt(self, request: str, proposal_type: str | None = None, context: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.generator.prompt(request, proposal_type=proposal_type, context=context)

    def generate(self, request: str, proposal_type: str | None = None, context: dict[str, Any] | None = None, provider_name: str | None = None, model: str | None = None, timeout: float = 8.0, use_llm: bool = False) -> dict[str, Any]:
        return self.generator.generate(request, proposal_type=proposal_type, context=context, provider_name=provider_name, model=model, timeout=timeout, use_llm=use_llm)

    def enqueue(self, request: str, proposal_type: str | None = None, context: dict[str, Any] | None = None, provider_name: str | None = None, model: str | None = None, timeout: float = 8.0, use_llm: bool = False) -> dict[str, Any]:
        return self.generator.generate_and_enqueue(request, proposal_type=proposal_type, context=context, provider_name=provider_name, model=model, timeout=timeout, use_llm=use_llm)

    def batch(self, items: list[dict[str, Any]], enqueue: bool = False, provider_name: str | None = None, model: str | None = None, timeout: float = 8.0, use_llm: bool = False) -> dict[str, Any]:
        return self.generator.batch_generate(items, enqueue=enqueue, provider_name=provider_name, model=model, timeout=timeout, use_llm=use_llm)
