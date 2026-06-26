from pathlib import Path

from core.cognitive_context_builder import CognitiveContextBuilder
from core.knowledge_context import KnowledgeContextService
from core.user_knowledge_base import UserKnowledgeBaseService


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_context_builder_exposes_ranking_budget_and_deduplication(tmp_path: Path):
    root = tmp_path / "user_knowledge"
    body = "Pandora Context Ranking Test. Funkgeraet Messtechnik Flughafen. " * 40
    _write(root / "public" / "a.md", "---\ntitle: Ranking A\ntags: [pandora, funk]\ncloud_allowed: true\n---\n" + body)
    _write(root / "public" / "b.md", "---\ntitle: Ranking B\ntags: [pandora, funk]\ncloud_allowed: true\n---\n" + body)
    _write(root / "restricted_cloud_allowed" / "c.md", "---\ntitle: Andere Notiz\ncloud_allowed: true\n---\nPandora Context Ranking anderer Inhalt")

    knowledge = UserKnowledgeBaseService(root_dir=root)
    context = KnowledgeContextService(knowledge=knowledge, include_obsidian=False, max_total_chars=900, max_chars_per_file=420, max_files=5)
    payload = context.build(query="Pandora", target="cloud", limit=5)

    assert payload["source_count"] >= 1
    assert payload["context_chars"] <= 900
    assert payload["ranking"]["candidate_count"] >= 2
    assert payload["ranking"]["duplicate_removed_count"] >= 1
    assert payload["ranking"]["token_budget"]["estimated_tokens"] > 0
    assert all("context_rank" in source for source in payload["sources"])


def test_cognitive_context_builder_status_lists_completion_features():
    status = CognitiveContextBuilder().status()
    assert "context_ranking" in status["completion_features"]
    assert "token_budget" in status["completion_features"]
    assert "duplicate_removal" in status["completion_features"]
