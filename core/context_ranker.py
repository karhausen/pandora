from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

_WORD_RE = re.compile(r"[\wäöüÄÖÜß]+", re.UNICODE)


def _words(value: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(value or "") if len(w) > 1]


def _normalize(value: str) -> str:
    return " ".join(_words(value))


@dataclass
class ContextCandidate:
    source_type: str
    source_id: str
    header: str
    text: str
    source: dict[str, Any]
    base_score: float = 0.0
    policy_rank: int = 0

    @property
    def block(self) -> str:
        return f"[{self.header}]\n{self.text.strip()}"


@dataclass
class ContextRanker:
    """Rank, deduplicate and budget context before it reaches an LLM.

    This is deliberately deterministic. LLMs receive only this prepared output;
    they never query files directly and they do not decide which local source is
    allowed to leave Pandora's policy boundary.
    """

    max_total_chars: int = 6000
    max_chars_per_file: int = 1400
    max_sources: int = 5
    duplicate_prefix_words: int = 80

    def build(self, *, query: str, candidates: list[ContextCandidate]) -> dict[str, Any]:
        ranked = self._rank(query, candidates)
        selected: list[ContextCandidate] = []
        duplicate_sources: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        seen_fingerprints: set[str] = set()
        total = 0

        for item in ranked:
            if len(selected) >= self.max_sources:
                break
            source_id = item.source_id
            fingerprint = self._fingerprint(item.text)
            if source_id in seen_ids or fingerprint in seen_fingerprints:
                duplicate_sources.append({**item.source, "duplicate_reason": "same_source_or_text"})
                continue
            block = item.block
            if total + len(block) > self.max_total_chars:
                remaining = max(0, self.max_total_chars - total)
                if remaining < 240:
                    break
                clipped_text = self._clip(item.text, max(0, remaining - len(item.header) - 8))
                item = ContextCandidate(item.source_type, item.source_id, item.header, clipped_text, item.source, item.base_score, item.policy_rank)
                block = item.block
            selected.append(item)
            seen_ids.add(source_id)
            seen_fingerprints.add(fingerprint)
            total += len(block)

        context_text = "\n\n---\n\n".join(item.block for item in selected)
        return {
            "context_text": context_text,
            "context_chars": len(context_text),
            "sources": [self._source_payload(item, idx + 1) for idx, item in enumerate(selected)],
            "source_count": len(selected),
            "ranking": {
                "query_terms": sorted(set(_words(query))),
                "candidate_count": len(candidates),
                "ranked_count": len(ranked),
                "selected_count": len(selected),
                "duplicate_removed_count": len(duplicate_sources),
                "token_budget": {
                    "max_total_chars": self.max_total_chars,
                    "max_chars_per_file": self.max_chars_per_file,
                    "max_sources": self.max_sources,
                    "used_chars": len(context_text),
                    "estimated_tokens": max(1, len(context_text) // 4) if context_text else 0,
                },
                "duplicates": duplicate_sources[:20],
            },
        }

    def _rank(self, query: str, candidates: list[ContextCandidate]) -> list[ContextCandidate]:
        query_terms = set(_words(query))
        ranked: list[ContextCandidate] = []
        for item in candidates:
            haystack = _normalize(" ".join([item.source_id, str(item.source.get("title", "")), " ".join(item.source.get("tags", []) or []), item.text[:4000]]))
            term_score = sum(4 for term in query_terms if term and term in haystack)
            title_score = sum(3 for term in query_terms if term and term in str(item.source.get("title") or item.source.get("name") or "").lower())
            tag_score = sum(3 for term in query_terms if term and any(term in str(tag).lower() for tag in (item.source.get("tags") or [])))
            existing = float(item.source.get("score") or item.base_score or 0.0)
            recency = min(float(item.source.get("modified_at") or 0.0) / 10_000_000_000, 1.0)
            final_score = existing + term_score + title_score + tag_score + item.policy_rank + recency
            new_source = {**item.source, "context_score": round(final_score, 3)}
            ranked.append(ContextCandidate(item.source_type, item.source_id, item.header, self._clip(item.text, self.max_chars_per_file), new_source, final_score, item.policy_rank))
        return sorted(ranked, key=lambda item: (-float(item.base_score), item.source_type, item.source_id))

    def _source_payload(self, item: ContextCandidate, rank: int) -> dict[str, Any]:
        payload = {**item.source}
        payload["context_rank"] = rank
        payload["source_type"] = payload.get("source_type") or item.source_type
        payload["source_id"] = item.source_id
        return payload

    def _fingerprint(self, text: str) -> str:
        prefix = " ".join(_normalize(text).split()[: self.duplicate_prefix_words])
        return sha256(prefix.encode("utf-8")).hexdigest() if prefix else ""

    def _clip(self, text: str, max_chars: int) -> str:
        value = (text or "").strip()
        if max_chars <= 0:
            return ""
        if len(value) <= max_chars:
            return value
        return value[:max_chars].rstrip() + "\n...[gekürzt]"
