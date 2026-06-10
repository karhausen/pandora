# MVP 22.10 – Knowledge Editor GUI

Der Knowledge Editor macht die dateibasierte `user_knowledge/`-Wissensbasis direkt in der Browser-GUI pflegbar.

## Funktionen

- Markdown-Dateien anlegen und bearbeiten
- Zielbereich auswählen: `public`, `restricted_cloud_allowed`, `private_local_only`
- Ordner im jeweiligen Bereich anlegen
- Metadaten per Formular pflegen
- Dateien speichern, verschieben und löschen
- Governance-Prüfung nach dem Speichern anzeigen
- Schutz gegen Pfadausbruch (`../`) und Schreibzugriff außerhalb `user_knowledge/`

## Policy-Regeln

`private_local_only` erzwingt immer:

```yaml
visibility: private_local_only
cloud_allowed: false
```

Für `public` und `restricted_cloud_allowed` kann `cloud_allowed` gesetzt werden, wird aber weiterhin durch Governance und Context Injection geprüft.

## Start

```bash
python main.py api --host 127.0.0.1 --port 8000
```

Dann öffnen:

```text
http://127.0.0.1:8000/knowledge-editor
```

## API

```text
GET  /api/gui/knowledge/editor/status
GET  /api/gui/knowledge/editor/tree
GET  /api/gui/knowledge/editor/template
GET  /api/gui/knowledge/editor/files/{area}/{relative_path}
POST /api/gui/knowledge/editor/files
POST /api/gui/knowledge/editor/folders
POST /api/gui/knowledge/editor/move
POST /api/gui/knowledge/editor/delete
```
