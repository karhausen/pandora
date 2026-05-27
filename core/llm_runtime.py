from __future__ import annotations

import json
from pydantic import ValidationError

from .llm_config import LLMConfig
from .llm_providers import LLMProviderFactory
from .llm_router import LLMRouter
from .models import LLMProvider, LLMRequest, LLMResponse, LLMTaskAnalysis, LLMTaskType


class LLMRuntime:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self.router = LLMRouter(self.config)
        self.factory = LLMProviderFactory(self.config)

    def complete(self, request: LLMRequest) -> LLMResponse:
        route = self.router.route(request.task_type, request.provider, request.model)
        provider = self.factory.get(route.provider)
        response = provider.complete(request, route.model)

        if response.success and request.expect_json and response.parsed_json is None:
            try:
                response.parsed_json = json.loads(response.content)
            except Exception as exc:
                response.success = False
                response.error = f"Invalid JSON response: {type(exc).__name__}: {exc}"

        return response

    def analyze_task(self, task: str, provider: LLMProvider | None = None, model: str | None = None) -> LLMTaskAnalysis:
        request = LLMRequest(
            task_type=LLMTaskType.PLANNING,
            prompt=task,
            context={"task": task},
            provider=provider,
            model=model,
            expect_json=True,
        )
        response = self.complete(request)
        if not response.success:
            raise RuntimeError(response.error or "LLM task analysis failed")
        try:
            return LLMTaskAnalysis.model_validate(response.parsed_json)
        except ValidationError as exc:
            raise RuntimeError(f"Invalid LLMTaskAnalysis schema: {exc}") from exc
