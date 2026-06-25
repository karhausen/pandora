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


## MVP 22.1.1 – Badge Links

Die User-GUI nutzt für die Hauptnavigation nun dieselben Badge-Link-Buttons wie Operations-, Admin- und Approval-Seiten.

Betroffen:

- `/` User-GUI Header Navigation
- Schnellzugriffskarten als `badge-card link`
- Operations Dashboard bleibt prominent erreichbar

Ziel: einheitliche Bedienung und sichtbare, klare Navigationspunkte.


## MVP 22.1.2 – Shared Badge Styles

Die User-GUI nutzt jetzt dieselbe zentrale Badge-Link-Basis wie Approval, Operations und Admin.

Wichtig:

- `web/shared.css` enthält die gemeinsame `.badge.link` Darstellung.
- `index.html`, `admin.html`, `approval.html` und `operations.html` laden dieses Stylesheet.
- Die User-Startseite nutzt keine abweichenden `badge-card` Schnellzugriffe mehr, sondern echte `badge link` Elemente.
- Der Operations-Dashboard-Link bleibt als `badge link primary` hervorgehoben.
