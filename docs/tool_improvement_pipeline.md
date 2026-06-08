# MVP 21.4 – Tool Improvement Pipeline

Die Tool Improvement Pipeline ist ein Wartungsbaustein für Pandora.

Ziel: Pandora erkennt Tools mit auffälliger Fehlerhistorie und erzeugt daraus einen prüfbaren Verbesserungsvorschlag.

Wichtig: Die Pipeline verändert keine aktiven Tools.

## Erlaubt

- Tool Registry lesen
- Tool-Statistiken lesen
- Fehlerhäufigkeit bewerten
- schwache Tools markieren
- Review-Paket unter `proposals/tool_improvements/` erzeugen

## Verboten

- Tool-Code automatisch überschreiben
- Tools automatisch deaktivieren
- Ersatztools automatisch installieren
- Registry-Status automatisch ändern
- Netzwerkzugriffe ausführen

## CLI

Status:

```bash
python main.py tool-improvement-status
```

Dry Run:

```bash
python main.py tool-improvement-run --dry-run --force
```

Echter Vorschlagslauf:

```bash
python main.py tool-improvement-run --force --limit 200
```

## Integration in Maintenance Manager

`maintenance-run` führt die Pipeline mit aus und nimmt erzeugte Vorschläge in den Wartungsbericht auf.

## Ergebnis

Ein Vorschlag sieht so aus:

```text
proposals/tool_improvements/tool_improvement_<tool_id>_<timestamp>/proposal.json
```

Der Vorschlag enthält:

- betroffener Tool-Name
- Nutzungsstatistiken
- Fehlergründe
- Risikoeinstufung
- empfohlene Reparaturschritte
- klare Sperre gegen Auto-Aktivierung
```
