# MVP 25.1.1 – GUI Chat Cognitive Context Fix

## Ziel

Der GUI-Chat nutzt denselben Cognitive-Context-Pfad wie die CLI-Vorschau.

## Fix

Fragen wie `Was sind die Topics in meinem Vault?` werden als Obsidian-Topic-Abfrage erkannt. Wenn die Policy den Zugriff erlaubt, beantwortet Pandora diese Frage direkt aus dem Vault-Index. Dadurch hängt die Antwort nicht davon ab, ob das ausgewählte LLM lokalen Dateizugriff vermutet.

## Sicherheit

- Lokales LLM: Obsidian erlaubt.
- Company LLM: nur mit `OBSIDIAN_COMPANY_ALLOWED=true`.
- Public Cloud: nur mit `OBSIDIAN_CLOUD_ALLOWED=true`.
- Bei blockierter Policy wird eine klare Diagnose ausgegeben.

## Test

```bash
python main.py cognitive-context-preview "Was sind die Topics in meinem Vault?"
python main.py api
# dann im GUI-Chat dieselbe Frage stellen
```
