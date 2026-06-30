# MVP 26.0 – Working Memory Foundation

Working Memory ist Pandoras temporärer Denkraum für eine aktive Aufgabe.

Sie speichert während eines Tasks:

- Ziele
- Hypothesen
- Zwischenergebnisse
- offene Fragen
- Prioritäten
- Entscheidungen
- nächste Aktionen

## Sicherheitsregel

Working Memory ist standardmäßig flüchtig. Sie schreibt nicht automatisch in:

- Long-Term Memory
- Obsidian
- User Knowledge Base
- Core-Dateien
- Tools oder Skills

Ein Export ist immer ein Review-Schritt und benötigt Freigabe.

## CLI

```bash
python main.py working-memory-status
python main.py working-memory-preview "Was war meine letzte Notiz?"
```

## API

```text
GET /api/cognitive/working-memory/status
GET /api/cognitive/working-memory/preview?query=...
```
