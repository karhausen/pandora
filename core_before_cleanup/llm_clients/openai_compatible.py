from __future__ import annotations
import json
import os
import socket
import urllib.error
import urllib.request
from core.models import LLMProvider, LLMRequest, LLMResponse

class OpenAICompatibleClient:
    provider = LLMProvider.OPENAI_COMPATIBLE

    def complete(self, request: LLMRequest, model: str, provider_name: str, provider_config: dict) -> LLMResponse:
        base_url = provider_config.get("base_url", "http://localhost:1234/v1").rstrip("/")
        api_key = provider_config.get("api_key") or os.environ.get(provider_config.get("api_key_env", ""), "lm-studio")
        timeout = float(request.timeout or provider_config.get("timeout", 20.0))
        messages = [{"role": "system", "content": request.system_prompt or "You are Pandora's structured LLM runtime. Return valid JSON when requested."}]
        if request.context:
            messages.append({"role": "system", "content": "Context JSON: " + json.dumps(request.context, ensure_ascii=False)})
        messages.append({"role": "user", "content": request.prompt})
        payload = {"model": model, "messages": messages, "temperature": provider_config.get("temperature", 0.2), "stream": False}
        if request.expect_json and provider_config.get("supports_response_format", False):
            payload["response_format"] = {"type": "json_object"}

        def _post(data: dict) -> dict:
            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))

        try:
            raw = _post(payload)
            message = raw["choices"][0]["message"]
            return LLMResponse(success=True, provider=self.provider, provider_name=provider_name, model=model, content=message.get("content") or "", raw=raw)
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            if "response_format" in payload:
                retry_payload = dict(payload)
                retry_payload.pop("response_format", None)
                try:
                    raw = _post(retry_payload)
                    message = raw["choices"][0]["message"]
                    raw["pandora_retry"] = {"reason": "response_format_rejected", "status": exc.code, "body": body}
                    return LLMResponse(success=True, provider=self.provider, provider_name=provider_name, model=model, content=message.get("content") or "", raw=raw)
                except Exception as retry_exc:
                    return LLMResponse(success=False, provider=self.provider, provider_name=provider_name, model=model, content="", error=f"HTTPError {exc.code}: {body} | retry_without_response_format_failed={type(retry_exc).__name__}: {retry_exc}")
            return LLMResponse(success=False, provider=self.provider, provider_name=provider_name, model=model, content="", error=f"HTTPError {exc.code}: {body or exc.reason}")
        except socket.timeout:
            return LLMResponse(success=False, provider=self.provider, provider_name=provider_name, model=model, content="", error=f"Timeout after {timeout}s talking to {base_url}")
        except Exception as exc:
            return LLMResponse(success=False, provider=self.provider, provider_name=provider_name, model=model, content="", error=f"{type(exc).__name__}: {exc}")

class OpenAIClient(OpenAICompatibleClient):
    provider = LLMProvider.OPENAI

    def complete(self, request: LLMRequest, model: str, provider_name: str, provider_config: dict) -> LLMResponse:
        provider_config = dict(provider_config)
        provider_config["base_url"] = provider_config.get("base_url", "https://api.openai.com/v1")
        api_key_env = provider_config.get("api_key_env", "OPENAI_API_KEY")
        provider_config["api_key"] = os.environ.get(api_key_env)
        if not provider_config["api_key"]:
            return LLMResponse(success=False, provider=self.provider, provider_name=provider_name, model=model, content="", error=f"{api_key_env} not set")
        response = super().complete(request, model, provider_name, provider_config)
        response.provider = self.provider
        return response
