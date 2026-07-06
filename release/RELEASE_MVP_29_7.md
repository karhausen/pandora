# MVP 29.7 – Evolution Dashboard

## Ziel

MVP 29.7 bündelt die Controlled-Evolution-Komponenten in einer gemeinsamen, read-only Übersicht.

Das Dashboard aktiviert keine Änderungen und trifft keine Freigaben. Es aggregiert ausschließlich Fakten aus bestehenden Komponenten.

## Enthalten

- Neues Paket `core/evolution_dashboard`
- `EvolutionDashboardManager`
- Gesamtstatus aller Evolution-Subsysteme
- Health Score
- Summary API
- Timeline API
- Statistik API
- GUI-Seite `Evolution Dashboard`
- CLI-Befehle:
  - `python main.py evolution-dashboard status`
  - `python main.py evolution-dashboard summary`
  - `python main.py evolution-dashboard health`
  - `python main.py evolution-dashboard timeline`
  - `python main.py evolution-dashboard statistics`
- API-Endpunkte:
  - `/api/evolution-dashboard/status`
  - `/api/evolution-dashboard/health`
  - `/api/evolution-dashboard/summary`
  - `/api/evolution-dashboard/statistics`
  - `/api/evolution-dashboard/timeline`
  - `/api/evolution-dashboard/overview`

## Architekturprinzip

Python sammelt Fakten.  
Python aggregiert.  
LLM entscheidet hier nichts.  
Benutzerfreigabe bleibt Pflicht für jede Änderung.

## Release Checks

- CLI Import: OK
- API Import: OK
- Evolution Dashboard Status: OK
- CLI Selftest: OK
- API Selftest: OK
- Release bereinigt: OK
