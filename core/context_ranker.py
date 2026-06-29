from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class ContextCandidate:
    """One policy-approved context item before final prompt packing."""

    source_type: str
    source_id: str
    title: str
    text: str
    metadata: dict[str, Any]
    base_score: float = 0.0


class ContextRanker:
    """Deterministic ranking, deduplication and budget packing for context.

    The ranker intentionally does not read files and does not decide policy.
    It only orders already-approved candidates and packs them into a bounded
    prompt context. This keeps Governance/Source Access in Python while making
    the Context Builder predictable and testable.
    """

    def __init__(self, *, max_total_chars: int = 6000, max_chars_per_item: int = 1400, max_items: int = 5):
        self.max_total_chars = max(0, int(max_total_chars))
        self.max_chars_per_item = max(0, int(max_chars_per_item))
        self.max_items = max(0, int(max_items))

    def select(self, *, query: str, candidates: list[ContextCandidate]) -> dict[str, Any]:
        scored = [self._score(query, candidate) for candidate in candidates if (candidate.text or "").strip()]
        scored.sort(key=lambda item: (-item["context_score"], item["candidate"].source_type, item["candidate"].source_id))

        duplicates_removed = 0
        seen_hashes: set[str] = set()
        unique: list[dict[str, Any]] = []
        for item in scored:
            digest = self._content_digest(item["candidate"].text)
            if digest in seen_hashes:
                duplicates_removed += 1
                continue
            seen_hashes.add(digest)
            unique.append(item)

        selected: list[dict[str, Any]] = []
        blocks: list[str] = []
        used_chars = 0
        budget_stopped = False
        for rank, item in enumerate(unique, start=1):
            if self.max_items and len(selected) >= self.max_items:
                budget_stopped = True
                break
            candidate: ContextCandidate = item["candidate"]
            excerpt = self._clip(candidate.text, self.max_chars_per_item)
            if not excerpt:
                continue
            header = self._header(candidate, rank=rank, score=item["context_score"])
            block = f"[{header}]\n{excerpt}"
            if self.max_total_chars and used_chars + len(block) > self.max_total_chars:
                if not selected:
                    available = max(0, self.max_total_chars - len(f"[{header}]\n") - 16)
                    excerpt = self._clip(candidate.text, available)
                    block = f"[{header}]\n{excerpt}" if excerpt else ""
                else:
                    budget_stopped = True
                    break
            if not block:
                continue
            blocks.append(block)
            used_chars += len(block)
            meta = dict(candidate.metadata)
            meta.update({
                "source_type": candidate.source_type,
                "source_id": candidate.source_id,
                "title": candidate.title,
                "context_rank": len(selected) + 1,
                "context_score": round(float(item["context_score"]), 3),
                "score_breakdown": item["score_breakdown"],
                "context_chars": len(block),
            })
            selected.append(meta)

        context_text = "\n\n---\n\n".join(blocks)
        return {
            "context_text": context_text,
            "context_chars": len(context_text),
            "sources": selected,
            "source_count": len(selected),
            "diagnostics": {
                "candidate_count": len(candidates),
                "ranked_count": len(scored),
                "unique_count": len(unique),
                "selected_count": len(selected),
                "duplicates_removed": duplicates_removed,
                "budget": {
                    "max_total_chars": self.max_total_chars,
                    "max_chars_per_item": self.max_chars_per_item,
                    "max_items": self.max_items,
                    "used_chars": len(context_text),
                    "stopped_by_budget": budget_stopped,
                },
            },
        }

    def _score(self, query: str, candidate: ContextCandidate) -> dict[str, Any]:
        q_terms = self._terms(query)
        haystack_terms = self._terms(" ".join([candidate.title, candidate.source_id, " ".join(map(str, candidate.metadata.get("tags", []))), " ".join(map(str, candidate.metadata.get("wikilinks", []))), candidate.text[:1000]]))
        if q_terms:
            overlap = len(q_terms & haystack_terms) / max(1, len(q_terms))
        else:
            overlap = 0.0
        title_hit = 1.0 if q_terms and q_terms & self._terms(candidate.title) else 0.0
        source_weight = {
            "obsidian": 18.0,
            "user_knowledge": 15.0,
            "conversation_memory": 12.0,
            "long_term_memory": 12.0,
            "capability": 8.0,
            "tool": 8.0,
            "skill": 8.0,
        }.get(candidate.source_type, 6.0)
        freshness = self._freshness_score(candidate.metadata.get("modified_at") or candidate.metadata.get("updated_at"))
        base = float(candidate.base_score or candidate.metadata.get("score") or 0.0)
        score = source_weight + min(base, 20.0) + overlap * 45.0 + title_hit * 10.0 + freshness * 7.0
        return {
            "candidate": candidate,
            "context_score": score,
            "score_breakdown": {
                "source_weight": source_weight,
                "base_score": round(min(base, 20.0), 3),
                "query_overlap": round(overlap, 3),
                "title_hit": title_hit,
                "freshness": round(freshness, 3),
            },
        }

    def _header(self, candidate: ContextCandidate, *, rank: int, score: float) -> str:
        parts = [f"Quelle: {candidate.source_type}/{candidate.source_id}", f"Rank: {rank}", f"Score: {round(score, 2)}"]
        policy = candidate.metadata.get("policy")
        if policy:
            parts.append(f"Policy: {policy}")
        if "company_allowed" in candidate.metadata:
            parts.append(f"Company erlaubt: {bool(candidate.metadata.get('company_allowed'))}")
        if "cloud_allowed" in candidate.metadata:
            parts.append(f"Cloud erlaubt: {bool(candidate.metadata.get('cloud_allowed'))}")
        return " | ".join(parts)

    def _terms(self, text: str) -> set[str]:
        stop = {"was", "wer", "wie", "wo", "wann", "war", "ist", "sind", "der", "die", "das", "den", "dem", "des", "ein", "eine", "meine", "mein", "letzte", "letzter", "letztes", "last", "the", "and", "oder", "und", "ich", "du", "zu", "in", "im", "am", "an", "auf", "mit", "von"}
        return {term for term in re.findall(r"[A-Za-zÄÖÜäöüß0-9_/-]{2,}", (text or "").lower()) if term not in stop}

    def _freshness_score(self, value: Any) -> float:
        if not value:
            return 0.0
        try:
            if isinstance(value, datetime):
                dt = value
            else:
                dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_days = max(0.0, (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 86400.0)
            if age_days <= 1:
                return 1.0
            if age_days <= 7:
                return 0.8
            if age_days <= 30:
                return 0.55
            if age_days <= 180:
                return 0.25
        except Exception:
            return 0.0
        return 0.0

    def _clip(self, text: str, max_chars: int) -> str:
        text = (text or "").strip()
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return text[: max(0, max_chars - 16)].rstrip() + "\n...[gekürzt]"

    def _content_digest(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", (text or "").strip().lower())[:4000]
        return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()
