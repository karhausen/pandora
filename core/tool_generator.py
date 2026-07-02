from __future__ import annotations

import re
from .models import SecurityLevel, ToolSpec


class ToolGenerator:
    def build_spec(self, capability: str) -> ToolSpec:
        tool_id = self._safe_id(capability)
        catalog = {
            "json_pretty": ToolSpec(
                id="json_pretty",
                name="JSON Pretty Printer",
                description="Formats JSON text with indentation.",
                capability="json_pretty",
                input_schema={"text": "str"},
                output_schema={"text": "str"},
                security_level=SecurityLevel.SAFE,
            ),
            "text_reverse": ToolSpec(
                id="text_reverse",
                name="Text Reverse",
                description="Reverses input text.",
                capability="text_reverse",
                input_schema={"text": "str"},
                output_schema={"text": "str"},
                security_level=SecurityLevel.SAFE,
            ),
            "word_count": ToolSpec(
                id="word_count",
                name="Word Count",
                description="Counts words in input text.",
                capability="word_count",
                input_schema={"text": "str"},
                output_schema={"count": "int"},
                security_level=SecurityLevel.SAFE,
            ),
            "prime_number_calculation": ToolSpec(
                id="prime_number_calculation",
                name="Prime Number Calculation",
                description="Checks whether an integer is prime and can list primes up to a limit.",
                capability="prime_number_calculation",
                input_schema={"number": "int", "limit": "int"},
                output_schema={"is_prime": "bool", "primes": "list"},
                security_level=SecurityLevel.SAFE,
            ),
            "timestamp": ToolSpec(
                id="timestamp",
                name="Timestamp",
                description="Returns current UTC timestamp.",
                capability="timestamp",
                input_schema={},
                output_schema={"timestamp": "str"},
                security_level=SecurityLevel.SAFE,
            ),
        }
        return catalog.get(tool_id, ToolSpec(
            id=tool_id,
            name=tool_id.replace("_", " ").title(),
            description=f"Generated candidate tool for capability: {capability}",
            capability=capability,
            input_schema={"text": "str"},
            output_schema={"text": "str"},
            security_level=SecurityLevel.SAFE,
        ))

    def generate_code(self, spec: ToolSpec) -> str:
        if spec.id == "json_pretty":
            return '''import json

TOOL_META = {
    "id": "json_pretty",
    "name": "JSON Pretty Printer",
    "description": "Formats JSON text with indentation.",
    "version": "0.1.0",
    "input_schema": {"text": "str"},
    "output_schema": {"text": "str"},
    "security_level": "SAFE",
    "status": "ACTIVE",
    "module": "generated_tools.json_pretty",
    "function": "run",
}

def run(payload: dict) -> dict:
    text = payload.get("text") or payload.get("input") or ""
    data = json.loads(text)
    return {"text": json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)}
'''
        if spec.id == "text_reverse":
            return '''TOOL_META = {
    "id": "text_reverse",
    "name": "Text Reverse",
    "description": "Reverses input text.",
    "version": "0.1.0",
    "input_schema": {"text": "str"},
    "output_schema": {"text": "str"},
    "security_level": "SAFE",
    "status": "ACTIVE",
    "module": "generated_tools.text_reverse",
    "function": "run",
}

def run(payload: dict) -> dict:
    text = payload.get("text") or payload.get("input") or ""
    return {"text": str(text)[::-1]}
'''
        if spec.id == "word_count":
            return '''TOOL_META = {
    "id": "word_count",
    "name": "Word Count",
    "description": "Counts words in input text.",
    "version": "0.1.0",
    "input_schema": {"text": "str"},
    "output_schema": {"count": "int"},
    "security_level": "SAFE",
    "status": "ACTIVE",
    "module": "generated_tools.word_count",
    "function": "run",
}

def run(payload: dict) -> dict:
    text = payload.get("text") or payload.get("input") or ""
    words = [w for w in str(text).split() if w.strip()]
    return {"count": len(words)}
'''
        if spec.id == "prime_number_calculation" or self._looks_like_prime_tool(spec):
            return f'''TOOL_META = {{
    "id": "{spec.id}",
    "name": "{spec.name}",
    "description": "{spec.description}",
    "version": "0.1.0",
    "input_schema": {spec.input_schema!r},
    "output_schema": {spec.output_schema!r},
    "security_level": "SAFE",
    "status": "ACTIVE",
    "module": "generated_tools.{spec.id}",
    "function": "run",
}}

MAX_LIMIT = 1_000_000

def _to_int(value, field_name: str) -> int:
    try:
        if isinstance(value, bool):
            raise ValueError
        return int(value)
    except Exception as exc:
        raise ValueError(f"{{field_name}} must be an integer") from exc

def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    divisor = 3
    while divisor * divisor <= n:
        if n % divisor == 0:
            return False
        divisor += 2
    return True

def _primes_up_to(limit: int) -> list[int]:
    if limit < 2:
        return []
    if limit > MAX_LIMIT:
        raise ValueError(f"limit must be <= {{MAX_LIMIT}}")
    return [n for n in range(2, limit + 1) if _is_prime(n)]

def run(payload: dict) -> dict:
    payload = payload or {{}}
    has_number = "number" in payload and payload.get("number") is not None
    has_limit = "limit" in payload and payload.get("limit") is not None
    if not has_number and not has_limit:
        raise ValueError("number or limit is required")
    is_prime = False
    primes = []
    if has_number:
        number = _to_int(payload.get("number"), "number")
        is_prime = _is_prime(number)
    if has_limit:
        limit = _to_int(payload.get("limit"), "limit")
        primes = _primes_up_to(limit)
    return {{"is_prime": is_prime, "primes": primes}}
'''

        if spec.id == "timestamp":
            return '''from datetime import datetime, UTC

TOOL_META = {
    "id": "timestamp",
    "name": "Timestamp",
    "description": "Returns current UTC timestamp.",
    "version": "0.1.0",
    "input_schema": {},
    "output_schema": {"timestamp": "str"},
    "security_level": "SAFE",
    "status": "ACTIVE",
    "module": "generated_tools.timestamp",
    "function": "run",
}

def run(payload: dict) -> dict:
    return {"timestamp": datetime.now(UTC).isoformat()}
'''
        if self._looks_like_word_counter(spec):
            output_key = self._first_output_key(spec, default="count")
            return f'''TOOL_META = {{
    "id": "{spec.id}",
    "name": "{spec.name}",
    "description": "{spec.description}",
    "version": "0.1.0",
    "input_schema": {spec.input_schema!r},
    "output_schema": {spec.output_schema!r},
    "security_level": "SAFE",
    "status": "ACTIVE",
    "module": "generated_tools.{spec.id}",
    "function": "run",
}}

def run(payload: dict) -> dict:
    text = payload.get("text") or payload.get("input") or ""
    words = [word for word in str(text).split() if word.strip()]
    return {{"{output_key}": len(words)}}
'''

        return f'''TOOL_META = {{
    "id": "{spec.id}",
    "name": "{spec.name}",
    "description": "{spec.description}",
    "version": "0.1.0",
    "input_schema": {spec.input_schema!r},
    "output_schema": {spec.output_schema!r},
    "security_level": "SAFE",
    "status": "ACTIVE",
    "module": "generated_tools.{spec.id}",
    "function": "run",
}}

def run(payload: dict) -> dict:
    text = payload.get("text") or payload.get("input") or ""
    return {{"text": str(text)}}
'''

    def _looks_like_prime_tool(self, spec: ToolSpec) -> bool:
        """Return True when a ToolSpec semantically describes prime-number work.

        This is intentionally a generator-contract fallback only. It is not used
        for capability-gap routing or user intent detection. Routing is handled
        by the LLM Capability Gap Analyzer.
        """
        text = " ".join([spec.id, spec.capability, spec.name, spec.description]).lower()
        output_keys = {str(key).lower() for key in spec.output_schema.keys()}
        return (
            "prime" in text
            or "primzahl" in text
            or "primzahlen" in text
            or bool(output_keys.intersection({"is_prime", "primes", "prime_numbers"}))
        )

    def _looks_like_word_counter(self, spec: ToolSpec) -> bool:
        text = " ".join([spec.id, spec.capability, spec.name, spec.description]).lower()
        output_keys = {str(key).lower() for key in spec.output_schema.keys()}
        output_types = {str(value).lower() for value in spec.output_schema.values()}
        return (
            ("word" in text and ("count" in text or "counter" in text))
            or bool(output_keys.intersection({"count", "word_count", "words"}) and output_types.intersection({"int", "integer", "number"}))
        )

    def _first_output_key(self, spec: ToolSpec, default: str = "result") -> str:
        for key in spec.output_schema.keys():
            return str(key)
        return default

    def _safe_id(self, capability: str) -> str:
        value = re.sub(r"[^a-zA-Z0-9_]+", "_", capability.strip().lower()).strip("_")
        return value or "generated_tool"
