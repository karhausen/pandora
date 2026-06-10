# Pandora Überblick

Pandora ist ein lokaler, modularer KI-Agent mit kontrolliertem Wachstum.
Der Core ist die stabile Schaltzentrale. Tools, Skills, Knowledge und Memory dürfen wachsen, aber nur nachvollziehbar und kontrolliert.

## Hauptkomponenten

```text
User
↓
User-GUI / CLI / API
↓
Coordinator Agent
↓
Model Router / Planner / Worker
↓
Tools / Skills / Knowledge / Memory
↓
Maintenance / Review / Approval
```

## Core-Aufgabe

Der Core soll:

- Aufgaben entgegennehmen
- zwischen Chat, Planung, Tools und Skills routen
- LLM-Profile nutzen
- Wissen und Kontext sicher bereitstellen
- Vorschläge erzeugen
- Risiken sichtbar machen
- User-Freigaben erzwingen

Der Core soll nicht:

- sich selbst ungeprüft überschreiben
- Sicherheitsregeln umgehen
- private Daten an Cloud-LLMs senden
- Runtime-Artefakte in Releases packen

## Wachstumsmodell

Pandora wächst über Vorschläge:

```text
Beobachtung
→ Analyse
→ Proposal
→ Review
→ Tests
→ Approval
→ Installation/Aktivierung
```

Night Mode und Maintenance erzeugen prüfbare Vorschläge, führen aber keine kritischen Änderungen ungefragt aus.
