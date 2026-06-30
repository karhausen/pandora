# MVP 26.6 – GUI Decision Inbox

MVP 26.6 macht die Central Decision Engine in der GUI sichtbar.

Ziel ist eine einfache Benutzerführung:

```text
Pandora erkennt einen notwendigen Schritt
↓
Decision Card in der GUI
↓
User wählt:
- Vorschlag ausarbeiten
- Später prüfen
- Ablehnen
- sicher fortfahren
```

## Grundregel

Die GUI Decision Inbox ist nur eine Anzeige- und Preview-Schicht.

Sie führt nichts aus:

- keine Tool-Ausführung
- keine Code-Generierung
- keine Tool-Aktivierung
- keine Knowledge-Schreiboperation
- keine Core-Änderung
- kein Release-Build

## Neue Komponente

```text
core/gui_decision_inbox.py
```

Die Komponente nutzt die Central Decision Engine und erzeugt daraus eine oder mehrere Decision Cards.

## CLI

```bash
python main.py gui-decision-inbox-status
python main.py gui-decision-inbox-preview "Ich brauche ein Tool für Aktienkurse"
python main.py gui-decision-inbox-preview "Ich brauche ein Tool für Aktienkurse" --user-action ja
```

## API

```text
GET /api/cognitive/gui-decision-inbox/status
GET /api/cognitive/gui-decision-inbox/preview?query=...
GET /api/cognitive/gui-decision-inbox/preview?query=...&user_action=ja
```

## GUI

```text
/decision-inbox
```

Die Seite zeigt:

- Anfragefeld
- Decision Cards
- einfache Aktionen
- Decision Trace
- Handoff für den nächsten kontrollierten Schritt

## Sicherheitsmodell

Auch wenn der Benutzer „Vorschlag ausarbeiten“ auswählt, bedeutet das nur:

```text
Proposal-Vorbereitung erlaubt
```

Nicht:

```text
Tool aktivieren
Core ändern
Knowledge schreiben
Release bauen
```

Diese Schritte bleiben hinter Review, Tests, Audit und finaler Freigabe.
