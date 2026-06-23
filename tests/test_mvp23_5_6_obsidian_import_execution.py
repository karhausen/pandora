from pathlib import Path

from core.obsidian_import_execution import ObsidianImportExecutionService
from core.obsidian_import_candidates import ObsidianImportCandidateService
from core.obsidian_vault import ObsidianConfig, ObsidianVaultService
from core.knowledge_editor import KnowledgeEditorService
from core.user_knowledge_base import UserKnowledgeBaseService


def test_obsidian_import_plan_requires_accepted_candidate(tmp_path):
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    (vault_dir / "Radio.md").write_text("# Radio\n\nFunktechnik Spektrumanalyse Testinhalt.", encoding="utf-8")
    knowledge_root = tmp_path / "user_knowledge"
    config = ObsidianConfig(enabled=True, vault_path=vault_dir, inbox_dir="Pandora_Inbox", mode="read_write_inbox_only", cloud_allowed=False)
    vault = ObsidianVaultService(root_dir=tmp_path, config=config)
    candidates = ObsidianImportCandidateService(candidates_dir=tmp_path / "candidates", vault=vault)
    report = candidates.build(limit=10, write=True)
    cid = report["candidates"][0]["id"]
    editor = KnowledgeEditorService(knowledge=UserKnowledgeBaseService(root_dir=knowledge_root))
    service = ObsidianImportExecutionService(candidates=candidates, editor=editor, vault=vault, executions_dir=tmp_path / "executions")
    plan = service.build_plan(cid)
    assert plan["ok"] is True
    assert plan["allowed_to_execute"] is False
    assert "accepted_for_next_step" in " ".join(plan["warnings"])


def test_obsidian_import_execute_writes_user_knowledge_only_after_acceptance(tmp_path):
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    (vault_dir / "Radio.md").write_text("---\ntitle: Old\n---\n# Radio\n\nFunktechnik Spektrumanalyse Testinhalt.", encoding="utf-8")
    knowledge_root = tmp_path / "user_knowledge"
    config = ObsidianConfig(enabled=True, vault_path=vault_dir, inbox_dir="Pandora_Inbox", mode="read_write_inbox_only", cloud_allowed=False)
    vault = ObsidianVaultService(root_dir=tmp_path, config=config)
    candidates = ObsidianImportCandidateService(candidates_dir=tmp_path / "candidates", vault=vault)
    cid = candidates.build(limit=10, write=True)["candidates"][0]["id"]
    candidates.decide(cid, decision="accepted_for_next_step", note="ok")
    editor = KnowledgeEditorService(knowledge=UserKnowledgeBaseService(root_dir=knowledge_root))
    service = ObsidianImportExecutionService(candidates=candidates, editor=editor, vault=vault, executions_dir=tmp_path / "executions")
    result = service.execute(cid, confirm=True)
    assert result["ok"] is True
    target = Path(result["saved"]["area"])
    written = list(knowledge_root.rglob("*.md"))
    assert written
    assert "Importiert aus Obsidian" in written[0].read_text(encoding="utf-8")
    assert (tmp_path / "executions").exists()
