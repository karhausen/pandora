from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .config import MEMORY_DIR
from .models import LLMResponse, LLMTaskAnalysis, LLMTaskType


class LLMReliabilityReport(BaseModel):
    valid_json: bool = False
    schema_valid: bool = False
    recovered: bool = False
    confidence: float = 0.0
    error: str | None = None
    source: str = "raw"
    parsed_json: Any = None
    reasoning: str | None = None


class LLMReliabilityLayer:
    """Recovery and diagnostics for small/local LLM responses.

    Local models often return Markdown fences, <think> blocks, explanatory text,
    reasoning-only responses or a valid JSON object with the wrong schema. This
    layer centralizes cleanup and records useful diagnostics without making the
    caller depend on model-specific quirks.
    """

    def __init__(self, reasoning_root: Path | None = None):
        self.reasoning_root = reasoning_root or (MEMORY_DIR / "reasoning")

    def process_response(self, response: LLMResponse, task_type: LLMTaskType, task: str | None = None) -> LLMResponse:
        reasoning = self.extract_reasoning(response)
        if reasoning:
            response.reasoning = reasoning
            self.store_reasoning(task_type.value, reasoning, response, task=task)

        report = self.recover_json(response.content)
        response.reliability = report.model_dump(mode="json")
        response.recovered = report.recovered
        response.confidence = report.confidence
        if report.parsed_json is not None and response.parsed_json is None:
            response.parsed_json = report.parsed_json
        return response

    def recover_json(self, content: str) -> LLMReliabilityReport:
        text = content or ""
        if not text.strip():
            return LLMReliabilityReport(valid_json=False, confidence=0.0, error="empty content", source="empty")

        cleaned = self._remove_think_blocks(text).strip()
        attempts = [cleaned]
        fenced = self._extract_fenced_json(cleaned)
        if fenced:
            attempts.insert(0, fenced)
        embedded = self._extract_json_object(cleaned)
        if embedded:
            attempts.append(embedded)

        seen = set()
        for candidate in attempts:
            candidate = candidate.strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            try:
                parsed = json.loads(candidate)
                recovered = candidate != text.strip()
                return LLMReliabilityReport(
                    valid_json=True,
                    recovered=recovered,
                    confidence=0.85 if recovered else 0.95,
                    parsed_json=parsed,
                    source="recovered_json" if recovered else "raw_json",
                )
            except Exception:
                pass

        return LLMReliabilityReport(valid_json=False, confidence=0.15, error="no valid JSON object found", source="unparsed")

    def validate_task_analysis(self, data: Any, task: str) -> tuple[dict[str, Any], LLMReliabilityReport]:
        report = LLMReliabilityReport(valid_json=isinstance(data, dict), parsed_json=data, confidence=0.0)
        try:
            model = LLMTaskAnalysis.model_validate(data)
            report.schema_valid = True
            report.confidence = 0.95
            report.source = "schema_valid"
            return model.model_dump(mode="json"), report
        except ValidationError as exc:
            recovered = self.recover_task_analysis(data, task)
            if recovered:
                report.schema_valid = False
                report.recovered = True
                report.confidence = 0.62
                report.error = f"schema recovery used after ValidationError: {exc}"
                report.source = "schema_recovered"
                return recovered, report
            report.error = f"schema invalid and unrecoverable: {exc}"
            report.source = "schema_invalid"
            return {}, report

    def recover_task_analysis(self, data: Any, task: str) -> dict[str, Any] | None:
        if not isinstance(data, dict):
            return None
        text = task.strip().lower()
        result = data.get("result")
        if result is not None and re.search(r"\d+\s*[+\-*/]\s*\d+", text):
            return {
                "task": task,
                "summary": f"Recovered planner analysis for calculation task: {task}",
                "intent": "calculation",
                "complexity": "low",
                "required_capabilities": ["calculation"],
                "suggested_tools": ["calculator"],
                "suggested_skills": [],
                "missing_capabilities": [],
                "risk_level": "LOW",
                "next_action": "use_tool",
                "recovered_from": data,
            }
        return None

    def extract_reasoning(self, response: LLMResponse) -> str | None:
        raw = response.raw
        try:
            if isinstance(raw, dict):
                choices = raw.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
                    if reasoning:
                        return str(reasoning).strip()
        except Exception:
            return None
        return None

    def store_reasoning(self, task_type: str, reasoning: str, response: LLMResponse, task: str | None = None) -> None:
        folder = self.reasoning_root / task_type
        folder.mkdir(parents=True, exist_ok=True)
        filename = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f") + ".json"
        payload = {
            "created_at": datetime.now(UTC).isoformat(),
            "task_type": task_type,
            "task": task,
            "provider_name": response.provider_name,
            "model": response.model,
            "reasoning": reasoning,
        }
        (folder / filename).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _remove_think_blocks(self, text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()

    def _extract_fenced_json(self, text: str) -> str | None:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else None

    def _extract_json_object(self, text: str) -> str | None:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return text[start:end + 1]
        return None
