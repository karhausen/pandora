from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path

from .changelog_manager import ChangelogManager
from .config import ARCHITECTURE_REPORT_FILE, DOCS_DIR, ROOT_DIR
from .governance import Governance


class DocumentationGenerator:
    DOCS = {
        "architecture.md": "# Architektur\n\nPandora besteht aus Core, Tools, Skills, Memory, Agent Loop, Capability Expansion, Learning und Web-GUI.\n",
        "security.md": "# Sicherheit\n\nDer aktive Core wird nicht unkontrolliert überschrieben. Kritische Änderungen benötigen explizite Freigabe.\n",
        "api.md": "# API\n\nDie REST API basiert auf FastAPI. Wichtige Gruppen: Status, Agent, Tools, Skills, Capability, Learning.\n",
        "cli.md": "# CLI\n\nZentrale Befehle: status, heartbeat, agent-run, tools, skills, learn-from-journal, docs-generate.\n",
        "tools.md": "# Tools\n\nTools sind kleine, validierbare Python-Funktionen mit Metadaten, Registry-Eintrag und Teststatus.\n",
        "skills.md": "# Skills\n\nSkills kombinieren mehrere Tools zu wiederverwendbaren Workflows.\n",
        "learning.md": "# Learning\n\nMVP 13 speichert Rankings, Empfehlungen, Fehleranalysen und Strategien aus dem Task Journal.\n",
        "heartbeat.md": "# Heartbeat\n\nDer Heartbeat prüft zentrale Core-Funktionen und dient als Lebenszeichen des Systems.\n",
        "rollback.md": "# Rollback\n\nRollback bleibt ein geschützter Core-Bereich. Neue Versionen dürfen nicht ungeprüft aktiv werden.\n",
        "evolution.md": "# Evolution\n\nPandora entwickelt Tools und Skills über kontrollierte Proposals, Validierung und explizite Aktivierung.\n",
        "roadmap.md": "# Roadmap\n\nNächste Schritte: bessere GUI, persistente Strategy-Nutzung, Core-Versionierung, echte Sandbox-Isolation.\n",
    }

    def generate(self) -> dict:
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        written = []

        for filename, content in self.DOCS.items():
            path = DOCS_DIR / filename
            if not path.exists():
                path.write_text(content, encoding="utf-8")
            written.append(str(path.relative_to(ROOT_DIR)))

        self._write_readme()
        ChangelogManager().add_entry(
            "MVP 14.0",
            "Documentation & Governance Layer",
            [
                "Added structured docs directory.",
                "Added changelog manager.",
                "Added governance checks.",
                "Added documentation generator.",
                "Added architecture report generation.",
            ],
        )
        governance = Governance().check()
        architecture = self.architecture_report()

        return {
            "generated": True,
            "docs": written,
            "governance": governance,
            "architecture": architecture,
        }

    def architecture_report(self) -> dict:
        core_files = sorted([p.name for p in (ROOT_DIR / "core").glob("*.py")])
        tool_files = sorted([p.name for p in (ROOT_DIR / "tools").glob("*.py") if not p.name.startswith("__")])
        skill_files = sorted([p.name for p in (ROOT_DIR / "skills").glob("*.json")])
        docs_files = sorted([p.name for p in DOCS_DIR.glob("*.md")])

        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "core_files": core_files,
            "tool_files": tool_files,
            "skill_files": skill_files,
            "docs_files": docs_files,
            "counts": {
                "core_files": len(core_files),
                "tools": len(tool_files),
                "skills": len(skill_files),
                "docs": len(docs_files),
            },
        }

        ARCHITECTURE_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        ARCHITECTURE_REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    def _write_readme(self) -> None:
        readme = ROOT_DIR / "README.md"
        readme.write_text("""# Pandora Agent

Lokaler modularer KI-Agent mit stabilem Core, Tool-/Skill-Evolution, Learning Layer und Web-GUI.

## Projektziel

Pandora soll Aufgaben analysieren, Tools und Skills kontrolliert nutzen, aus Erfahrungen lernen und neue Fähigkeiten sicher vorschlagen.

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py status
python main.py api
```

Web-GUI:

```text
http://127.0.0.1:8000
```

## CLI

Wichtige Befehle:

```powershell
python main.py status
python main.py heartbeat
python main.py agent-run "Bitte rechne 2+3*4" --provider mock
python main.py tools
python main.py skills
python main.py learn-from-journal
python main.py recommendations
python main.py docs-generate
python main.py governance-check
```

## API

FastAPI stellt Status-, Agent-, Tool-, Skill-, Capability-, Learning- und Dokumentations-Endpunkte bereit.

## Sicherheit

Der aktive Core darf nicht unkontrolliert überschrieben werden. Kritische Core-Dateien sind geschützt. Neue Tools und Skills entstehen zuerst als Proposal und werden erst nach Validierung und expliziter Aktivierung übernommen.

## Architektur

Siehe `docs/architecture.md`.

## Dokumentation

Weitere Dokumentation befindet sich unter `docs/`.

## Roadmap

Siehe `docs/roadmap.md`.
""", encoding="utf-8")
