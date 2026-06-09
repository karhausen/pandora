# MVP 22.6.2 – User GUI Routing Sync

Die User-GUI nutzt keine eigene Provider-Auswahl mehr. Der aktive Chat-Provider wird aus der zentralen LLM-Routing-Konfiguration gelesen und nur noch angezeigt.

## Warum

Vorher konnte die User-GUI einen lokalen Provider aus `localStorage` verwenden. Dadurch stimmte der Chat-Provider nicht zwingend mit dem LLM Routing Editor überein.

## Neues Verhalten

- `/user/status` liefert `active_chat_route` aus dem zentralen `ModelRouter`.
- Die User-GUI zeigt `Chat-Route`, Provider, Modell und Routing-Quelle an.
- Änderungen erfolgen über das LLM & Profile Center.
- Chat-Aufrufe senden keinen lokalen Provider-Override mehr.

Damit ist der LLM Routing Editor die einzige Steuerstelle für den Chat-Provider.
