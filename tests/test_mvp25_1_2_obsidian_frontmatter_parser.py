from core.obsidian_vault import ObsidianVaultService


def test_frontmatter_mixed_bool_and_list_fields_do_not_crash():
    text = """---
title: Test Note
cloud_allowed: false
company_allowed: true
tags:
  - funktechnik
  - obsidian
priority: high
---
# Inhalt
"""
    metadata = ObsidianVaultService()._extract_frontmatter(text)

    assert metadata["title"] == "Test Note"
    assert metadata["cloud_allowed"] is False
    assert metadata["company_allowed"] is True
    assert metadata["tags"] == ["funktechnik", "obsidian"]
    assert metadata["priority"] == "high"


def test_frontmatter_stray_list_after_bool_is_tolerated():
    text = """---
cloud_allowed: false
  - should_not_crash
tags: [pandora, obsidian]
---
content
"""
    metadata = ObsidianVaultService()._extract_frontmatter(text)

    assert metadata["cloud_allowed"] is False
    assert metadata["tags"] == ["pandora", "obsidian"]
