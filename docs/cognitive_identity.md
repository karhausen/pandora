# MVP 28.0 – Cognitive Identity & Self Model

MVP 28.0 ergänzt Pandora um ein explizites, auslesbares Selbstmodell. Ziel ist nicht mehr Autonomie, sondern mehr Klarheit: Pandora soll jederzeit sagen können, was sie ist, was sie kann, was sie nicht kann und wann sie stoppen oder Zustimmung einholen muss.

## Zielbild

Pandora bekommt eine eigene Identity-Schicht mit vier Kernantworten:

- **Identity Card:** Name, Systemtyp, Mission, Grundprinzipien.
- **Self Model:** Zusammenfassung von Identität, Fähigkeiten, Grenzen und relevanten kognitiven Schnittstellen.
- **Capability Boundaries:** Was Pandora tun darf, wo sie stoppen muss und welche Schwächen bekannt sind.
- **Safe Operating Statement:** Konkrete Aussage, dass diese Schicht nur analysiert und keine Änderungen ausführt.

## Sicherheitsgrenzen

`CognitiveIdentityService` ist bewusst read-only:

- keine Tool-Ausführung
- keine Tool-Aktivierung
- keine Obsidian-/Knowledge-Writes
- keine Core-Änderungen
- keine Persistenz
- keine automatische Proposal-Freigabe

Die Identity-Schicht darf bestehende Status- und Preview-Komponenten lesen. Sie darf aber keine kontrollierten Aktionen durchführen.

## CLI

```bash
python main.py cognitive-identity-status
python main.py cognitive-identity-card
python main.py cognitive-boundaries
python main.py cognitive-self-model
python main.py cognitive-self-model "Baue ein neues Tool für Wetterdaten"
```

## Rolle im Gesamtsystem

MVP 28.0 sitzt oberhalb der kognitiven 27.x-Schicht:

- Working Memory
- Central Decision Engine
- Goal Manager
- Priority Engine
- Review Cycle Engine
- Cognitive Dashboard
- Review-to-Action Workflow
- Action Proposal Handoff

Die neue Schicht bündelt diese Sicht, ohne deren Regeln zu umgehen.

## Erwartetes Verhalten

Pandora soll künftig deutlicher zwischen diesen Zuständen unterscheiden:

1. Fähigkeit ist vorhanden.
2. Fähigkeit ist nur erkannt oder vorgeschlagen.
3. Ergebnis ist nur Preview, nicht Ausführung.
4. Änderung braucht User-Freigabe.
5. Test/Audit wurde noch nicht ausgeführt.
6. Live- oder externe Daten fehlen.

## Ergebnis

MVP 28.0 macht Pandora ehrlicher und besser wartbar. Das System bekommt ein explizites Selbstbild, ohne in unkontrollierte Selbstveränderung abzurutschen.
