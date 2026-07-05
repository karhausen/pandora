from __future__ import annotations
import json
import urllib.request
from core.models import LLMProvider, LLMRequest, LLMResponse

class OllamaClient:
    provider = LLMProvider.OLLAMA

    def complete(self, request: LLMRequest, model: str, provider_name: str, provider_config: dict) -> LLMResponse:
        base_url = provider_config.get("base_url", "http://localhost:11434").rstrip("/")
        prompt = request.prompt
        if request.context:
            prompt = "Context JSON:\n" + json.dumps(request.context, ensure_ascii=False) + "\n\nUser:\n" + prompt
        payload = {"model": model, "prompt": prompt, "stream": False}
        if request.expect_json:
            payload["format"] = "json"
        try:
            req = urllib.request.Request(f"{base_url}/api/generate", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=float(request.timeout or provider_config.get("timeout", 30.0))) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            return LLMResponse(success=True, provider=self.provider, provider_name=provider_name, model=model, content=raw.get("response", ""), raw=raw)
        except Exception as exc:
            return LLMResponse(success=False, provider=self.provider, provider_name=provider_name, model=model, content="", error=f"{type(exc).__name__}: {exc}")
