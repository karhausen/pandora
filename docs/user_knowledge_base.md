# MVP 22.7 – User Knowledge Base

Die User Knowledge Base ist ein dateibasierter Wissensbereich für vom Nutzer gepflegte Notizen.

## Ziel

Pandora trennt jetzt eigenes Runtime-Memory von dauerhaftem Nutzerwissen.

## Verzeichnisstruktur

```text
user_knowledge/
├── public/
├── restricted_cloud_allowed/
└── private_local_only/
```

## Regeln

- `public`: darf lokal und in Cloud-LLMs verwendet werden.
- `restricted_cloud_allowed`: darf nur nach Policy-Prüfung in Cloud-/Company-Kontext verwendet werden.
- `private_local_only`: darf niemals in Cloud-Kontext verwendet werden.

## Unterstützte Formate

- Markdown (`.md`)
- Text (`.txt`)
- JSON (`.json`)

## GUI

Start über:

```text
/knowledge-base
```

## CLI

```bash
python main.py knowledge-ensure
python main.py knowledge-dashboard
python main.py knowledge-search "suchbegriff"
python main.py knowledge-search "suchbegriff" --cloud-context
python main.py knowledge-context-preview "suchbegriff" --target cloud
```

## Sicherheit

Die API ist read-only. Pandora kann die Struktur anlegen, verändert aber keine Wissensdateien.
Private Dateien aus `private_local_only` werden bei Cloud-/Company-Kontextvorschau blockiert.
