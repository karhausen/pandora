# Pandora Agent MVP 5.0

MVP 5 ergänzt die erste echte Evolutionsschicht:

- Episodic Memory
- Reflection Engine
- Skill Quality Scoring
- Pattern Detection
- automatische Skill-Proposals aus wiederkehrenden erfolgreichen Tool-Ketten

Der Core wird weiterhin nicht autonom verändert.

## Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py status
```

## Basisbefehle

```powershell
python main.py heartbeat
python main.py tools
python main.py skills
python main.py run-tool echo --input "Hallo Agent"
python main.py run-skill echo_then_upper --file payload_skill.json
```

## Neue MVP-5-Befehle

Episoden anzeigen:

```powershell
python main.py episodes
```

Skill-Qualität anzeigen:

```powershell
python main.py skill-quality
```

Wiederkehrende Tool-Sequenzen erkennen:

```powershell
python main.py learn-patterns --min-count 2
```

Skill-Vorschläge aus Mustern erzeugen:

```powershell
python main.py propose-skills --min-count 2
```

Reflection anzeigen:

```powershell
python main.py reflections
```

## Beispielablauf für Pattern Learning

Payload-Datei `payload_skill.json`:

```json
{
  "text": "Hallo Agent"
}
```

Skill mehrfach ausführen:

```powershell
python main.py run-skill echo_then_upper --file payload_skill.json
python main.py run-skill echo_then_upper --file payload_skill.json
```

Dann Muster erkennen:

```powershell
python main.py learn-patterns --min-count 2
```

Dann Vorschlag erzeugen:

```powershell
python main.py propose-skills --min-count 2
```

Der Vorschlag landet unter:

```text
proposals/skills/
```

## Architekturregel

MVP 5 erzeugt nur Vorschläge.

Nicht automatisch verändert werden:

- aktiver Core
- Heartbeat
- Rollback
- Recovery
- Security
- Config

## Was jetzt möglich ist

Der Agent kann jetzt aus erfolgreichem Verhalten lernen:

```text
Ausführung
→ Episode
→ Reflection
→ Pattern Detection
→ Skill Proposal
```

Das ist die Grundlage für kontrollierte Evolution.

## Tests

```powershell
pytest
```

## Nächster Schritt: MVP 6

MVP 6 sollte die REST API ausbauen:

- Task Endpoint
- Tool Endpoint
- Skill Endpoint
- Memory Endpoint
- Status Endpoint
- Heartbeat Endpoint
- Proposal Endpoint
- einfache Web/CLI-kompatible JSON-Schnittstelle
