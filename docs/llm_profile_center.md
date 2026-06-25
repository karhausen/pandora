# MVP 22.6 – LLM & Profile Center

Ziel: Pandora zeigt Profile, Provider und Routing in der User-GUI, ohne Secrets offenzulegen.

## Seite

```text
/llm-profiles
```

## API

```text
GET  /api/gui/llm-profiles/dashboard
GET  /api/gui/llm-profiles/profiles
POST /api/gui/llm-profiles/profile
GET  /api/gui/llm-profiles/providers
GET  /api/gui/llm-profiles/routes
POST /api/gui/llm-profiles/smoke-preview
```

## Sicherheitsregeln

- API-Keys werden nicht angezeigt.
- Environment-Werte werden nicht angezeigt.
- Profile-Umschaltung speichert nur den Profilnamen in der lokalen Override-Datei.
- Live-Smoke-Tests werden nicht automatisch ausgeführt.
- Company-LLM wird nur über Environment-Variablen konfiguriert.

## Bedienung

```bash
python main.py api --host 127.0.0.1 --port 8000
```

Dann öffnen:

```text
http://127.0.0.1:8000/llm-profiles
```

CLI:

```bash
python main.py llm-profile-center-dashboard
python main.py llm-profile-center-profiles
python main.py llm-profile-center-providers
python main.py llm-profile-center-routes
```
