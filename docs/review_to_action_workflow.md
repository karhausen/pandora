# MVP 27.7 – Review-to-Action Workflow

Der Review-to-Action Workflow verbindet Review-Ergebnisse mit Pandoras bestehender Approval- und Proposal-Pipeline.

## Ziel

Aus einem Weekly/Monthly Review entstehen einfache, prüfbare Aktionskarten:

- Vorschlag ausarbeiten
- später
- ablehnen

Die Aktion erzeugt **keine direkte Ausführung**. Bei Zustimmung wird nur ein kontrollierter Vorschlagsplatz vorbereitet.

## Ablauf

```text
Review Cycle
↓
Approval Points / Focus Items
↓
Review-to-Action Cards
↓
User: Vorschlag ausarbeiten / später / ablehnen
↓
Approval Interaction
↓
Proposal Review Loop
↓
Execution Gate
```

## Sicherheitsgarantie

MVP 27.7 führt nicht aus:

- kein Tool-Start
- keine Tool-Aktivierung
- keine Code-Generierung
- keine Vault-/Knowledge-Schreiboperation
- keine Core-Änderung
- keine Release-Erzeugung

## CLI

```bash
python main.py review-to-action-status
python main.py review-to-action-preview "Pandora Weekly Review" --cadence weekly
python main.py review-to-action-preview "Pandora Weekly Review" --user-action ja
```

## API

```text
GET /api/cognitive/review-to-action/status
GET /api/cognitive/review-to-action/preview?query=...&user_action=ja
```
