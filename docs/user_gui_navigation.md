# MVP 22.1 – User GUI Navigation

Ziel: Die User-GUI bleibt der einfache Einstiegspunkt für Pandora, zeigt aber klare Wege zu den wichtigen Steuerflächen.

## Neu

Die Startseite `/` enthält jetzt direkte Schnellzugriffe auf:

- Operations Dashboard (`/operations`)
- Approval Center (`/approval`)
- Admin Dashboard (`/admin`)

Der wichtigste Punkt ist der sichtbare Button zum Operations Dashboard. Damit ist Maintenance, Nightly Review, Systemstatus und sichere Wartung direkt aus der User-GUI erreichbar.

## Sicherheitsregel

Die User-GUI startet weiterhin keine Core-Änderungen direkt. Sie verlinkt nur auf die kontrollierten Dashboards. Entscheidungen und Wartung laufen weiter über die bestehenden API- und Governance-Schichten.

## Bedienlogik

- `/` bleibt die normale Chat-/Task-Oberfläche.
- `/operations` ist die Betriebs- und Wartungszentrale.
- `/approval` ist die menschliche Freigabestelle.
- `/admin` bleibt für technische Details.
