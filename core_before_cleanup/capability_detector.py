from __future__ import annotations

import re
import unicodedata

from .tool_registry import ToolRegistry


class CapabilityDetector:

    DOMAIN_TERMS = {
        "aktienkurs": "stock_price",
        "aktienkurse": "stock_price",
        "aktie": "stock_price",
        "aktien": "stock_price",
        "boersenkurs": "stock_price",
        "borsenkurs": "stock_price",
        "börsenkurs": "stock_price",
        "börse": "stock_price",
        "boerse": "stock_price",
        "stock": "stock_price",
        "stocks": "stock_price",
        "share price": "stock_price",
        "dollar": "exchange_rate",
        "dollarkurs": "exchange_rate",
        "dollar kurs": "exchange_rate",
        "wechselkurs": "exchange_rate",
        "exchange rate": "exchange_rate",
        "eur usd": "exchange_rate",
        "euro dollar": "exchange_rate",
        "währungskurs": "exchange_rate",
        "waehrungskurs": "exchange_rate",
        "wetter": "weather",
        "weather": "weather",
        "euro": "exchange_rate",
        "usd": "exchange_rate",
        "eurusd": "exchange_rate",
        "bitcoin": "crypto_price",
        "btc": "crypto_price",
    }

    LIVE_DATA_HINTS = [
        "aktuell", "aktueller", "aktuelle", "aktuellen", "jetzt", "heute", "morgen", "live",
        "abrufen", "holen", "anzeigen", "abfragen", "lookup", "fetch", "get",
        "kurs", "kurse", "preis", "preise", "price", "rate", "quote",
        "wird", "werden", "vorhersage", "forecast",
    ]

    TOOL_REQUEST_HINTS = [
        "ich brauche ein tool", "brauche ein tool", "tool um", "werkzeug um",
        "ich möchte ein tool", "ich brauche eine fähigkeit", "capability",
    ]

    KEYWORDS = {
        "json_pretty": ["json format", "pretty json", "json hübsch", "json formatieren"],
        "text_reverse": ["reverse text", "text umdrehen", "rückwärts"],
        "word_count": [
            "word count",
            "count words",
            "wörter zählen",
            "woerter zaehlen",
            "wörter zaehlen",
            "wörter",
            "woerter",
            "wortanzahl",
            "anzahl der wörter",
            "anzahl wörter",
        ],
        "timestamp": ["timestamp", "zeitstempel"],
        "weather_lookup": [
            "aktuelles wetter",
            "aktuelle wetterdaten",
            "aktuelle wetterinformationen",
            "wetter abrufen",
            "wetterdaten abrufen",
            "wetterinformationen abrufen",
            "wetterbericht",
            "weather lookup",
            "current weather",
            "weather forecast",
        ],
    }

    def __init__(self, registry: ToolRegistry | None = None):
        self.registry = registry or ToolRegistry()
        self.registry.discover()

    def _available_tool_id(self, capability: str) -> str | None:
        return self.registry.resolve_id(capability)

    def _existing_tool_ids(self) -> set[str]:
        return {tool.id for tool in self.registry.list()}

    def detect(self, task: str, analysis: dict | None = None) -> dict:
        task_l = task.lower()
        existing_tool_ids = self._existing_tool_ids()

        # If LLM already suggested a missing capability, prefer that.
        missing = []
        if analysis:
            missing = analysis.get("missing_capabilities") or []

        for capability in missing:
            available_tool = self._available_tool_id(capability)
            if not available_tool:
                return {
                    "gap_detected": True,
                    "capability": capability,
                    "reason": "LLM analysis reported missing capability.",
                    "existing_tools": sorted(existing_tool_ids),
                }

        for capability, keywords in self.KEYWORDS.items():
            if any(keyword in task_l for keyword in keywords):
                available_tool = self._available_tool_id(capability)
                if not available_tool:
                    return {
                        "gap_detected": True,
                        "capability": capability,
                        "reason": f"Task matched capability keywords for {capability}.",
                        "existing_tools": sorted(existing_tool_ids),
                    }
                return {
                    "gap_detected": False,
                    "capability": capability,
                    "reason": f"Capability {capability} is already covered by installed tool {available_tool}.",
                    "existing_tools": sorted(existing_tool_ids),
                    "tool_available": True,
                    "suggested_existing_tool": available_tool,
                }

        generic = self._detect_generic_capability(task_l)
        if generic:
            capability, reason = generic
            available_tool = self._available_tool_id(capability)
            if available_tool:
                return {
                    "gap_detected": False,
                    "capability": capability,
                    "reason": f"Generic capability detection matched {capability}, covered by installed tool {available_tool}.",
                    "existing_tools": sorted(existing_tool_ids),
                    "tool_available": True,
                    "suggested_existing_tool": available_tool,
                }
            return {
                "gap_detected": True,
                "capability": capability,
                "reason": reason,
                "existing_tools": sorted(existing_tool_ids),
            }

        return {
            "gap_detected": False,
            "capability": None,
            "reason": "No missing capability detected.",
            "existing_tools": sorted(existing_tool_ids),
        }

    def _detect_generic_capability(self, task_l: str) -> tuple[str, str] | None:
        normalized = self._normalize(task_l)
        explicit_tool_request = any(hint in normalized for hint in self.TOOL_REQUEST_HINTS)
        live_data_request = any(hint in normalized for hint in self.LIVE_DATA_HINTS)

        domain = self._domain_from_text(normalized)
        implicit_live_question = bool(domain and self._looks_like_live_data_question(normalized, domain))

        if not (explicit_tool_request or live_data_request or implicit_live_question):
            return None

        if domain:
            return f"{domain}_lookup", f"Generic capability detection inferred live lookup capability: {domain}_lookup."

        if explicit_tool_request:
            phrase = self._extract_requested_object(normalized)
            slug = self._slugify(phrase) if phrase else None
            if slug and slug not in {"tool", "werkzeug", "faehigkeit", "fahigkeit"}:
                return f"{slug}_lookup", f"Generic tool request inferred capability: {slug}_lookup."

        return None

    def _looks_like_live_data_question(self, text: str, domain: str) -> bool:
        question_hints = [
            "wie", "was", "welche", "welcher", "wieviel", "wie viel",
            "brauche ich", "soll ich", "ist", "wird", "werden", "steht",
            "current", "today", "tomorrow", "forecast", "price", "rate",
        ]
        if domain in {"weather", "stock_price", "exchange_rate", "crypto_price"}:
            return any(hint in text for hint in question_hints)
        return False

    def _domain_from_text(self, text: str) -> str | None:
        for term, domain in self.DOMAIN_TERMS.items():
            if self._normalize(term) in text:
                return domain
        return None

    def _extract_requested_object(self, text: str) -> str | None:
        patterns = [
            r"tool\s+um\s+(.+?)(?:\s+abzurufen|\s+zu\s+holen|\s+zu\s+lesen|$)",
            r"werkzeug\s+um\s+(.+?)(?:\s+abzurufen|\s+zu\s+holen|\s+zu\s+lesen|$)",
            r"ich\s+brauche\s+ein\s+tool\s+um\s+(.+?)(?:\s+abzurufen|$)",
            r"ich\s+moechte\s+(.+?)(?:\s+abrufen|\s+abfragen|$)",
            r"wie\s+ist\s+der\s+aktuelle\s+(.+?)(?:\?|$)",
            r"wie\s+ist\s+die\s+aktuelle\s+(.+?)(?:\?|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                phrase = match.group(1).strip()
                return self._cleanup_phrase(phrase)
        return None

    def _cleanup_phrase(self, phrase: str) -> str:
        stopwords = {
            "ich", "brauche", "ein", "eine", "einen", "tool", "werkzeug", "um",
            "zu", "der", "die", "das", "den", "dem", "aktuelle", "aktueller",
            "aktuellen", "kurs", "kurse", "abrufen", "abfragen", "holen",
        }
        words = [w for w in re.findall(r"[a-z0-9äöüß]+", phrase.lower()) if w not in stopwords]
        return " ".join(words[:4])

    def _slugify(self, text: str | None) -> str | None:
        if not text:
            return None
        text = self._normalize(text)
        text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
        parts = re.findall(r"[a-z0-9]+", text)
        return "_".join(parts[:4]) if parts else None

    def _normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text.lower())
        text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
        return " ".join(text.split())
