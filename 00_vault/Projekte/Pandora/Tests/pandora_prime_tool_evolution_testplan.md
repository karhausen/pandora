# Pandora Testplan – MVP 29.4 Tool Evolution  
## Beispiel: Primzahlen-Tool über Chat → Proposal → Tool → Observation → Evolution

Stand: MVP 29.4 – Tool Evolution  
Ziel: Den kompletten kontrollierten Prozess testen, ohne fiktive Tools wie `weather_lookup`.

---

## 1. Ausgangslage

Testidee im Chat:

```text
Ich brauche ein Tool, das Prim-Zahlen berechnet.
```

Erwartung:

Pandora soll nicht einfach nur antworten, sondern erkennen:

- Es wird eine berechenbare Fähigkeit benötigt.
- Ein passendes Tool existiert vermutlich noch nicht.
- Es entsteht ein Capability Gap.
- Daraus wird ein Proposal oder Tool-Entwurf.
- Das Tool wird nach Freigabe erzeugt.
- Das Tool wird getestet.
- Danach wird es beobachtet.
- Tool Evolution bewertet später Qualität und Verbesserungsbedarf.

---

## 2. Vorbereitungscheck

Im Projektordner ausführen:

```powershell
python main.py selftest cli
python main.py selftest api
python main.py selftest integration
```

Erwartung:

```text
OK
```

Zusätzlich prüfen:

```powershell
python main.py genome status
python main.py evolution status
python main.py evolution-factory status
python main.py proposal-queue status
python main.py tools evolution status
```

Wenn hier ein Kommando unbekannt ist:  
**Stopp. Dann ist zuerst die CLI-Integration defekt.**

---

## 3. GUI starten

```powershell
uvicorn api.main:app --reload
```

Dann öffnen:

```text
http://127.0.0.1:8000
```

Prüfen:

- User-GUI lädt
- Chat ist sichtbar
- Maintenance Center erreichbar
- Proposal Queue erreichbar
- Tool Evolution erreichbar
- Tool Center erreichbar, falls vorhanden

---

## 4. Testfall A – Chat-Anfrage stellen

In der User-GUI eingeben:

```text
Ich brauche ein Tool, das Prim-Zahlen berechnet.
```

Erwartung:

Pandora sollte sinngemäß erkennen:

- Tool benötigt
- kein vorhandenes Tool gefunden
- Vorschlag/Proposal wird erstellt oder vorbereitet

Nicht ideal wäre:

```text
Hier ist Python-Code...
```

oder nur:

```text
Primzahlen sind Zahlen...
```

Das wäre kein vollständiger Agentenprozess.

---

## 5. Erwartete Zwischenergebnisse prüfen

### 5.1 Capability Gap

In Maintenance prüfen:

```text
Maintenance → Evolution / Observation / Capability Gaps
```

Erwartung:

Ein Eintrag ähnlich:

```text
Missing capability: prime number calculation
```

oder:

```text
Tool needed: prime_number_tool
```

Wenn kein Gap sichtbar ist:  
**Problem in Intent-/Capability-Erkennung.**

---

### 5.2 Proposal Queue

Prüfen:

```text
Maintenance → Proposal Queue
```

Oder CLI:

```powershell
python main.py proposal-queue list
```

Erwartung:

Ein Proposal ähnlich:

```text
Type: TOOL
Title: Prime Number Tool
Status: DRAFT / REVIEW / PROPOSED
Priority: MEDIUM
```

Wenn kein Proposal entsteht:  
**Problem zwischen Capability Gap → Evolution Factory → Proposal Queue.**

---

### 5.3 Proposal Details

Proposal öffnen.

Prüfen:

- Zweck ist klar beschrieben.
- Input ist klar.
- Output ist klar.
- Risiko ist niedrig.
- Tests sind vorgesehen.
- Keine automatische Aktivierung ohne Freigabe.

Beispiel-Erwartung:

```text
Tool calculates prime numbers or checks whether a number is prime.
Input: integer or range
Output: prime status / list of primes
Risk: low
```

---

## 6. Freigabe / Tool-Erzeugung testen

Falls GUI einen Review-/Approve-Button hat:

```text
Approve / Generate Tool
```

Falls CLI nötig ist:

```powershell
python main.py proposal-queue approve <proposal_id>
```

oder entsprechendes vorhandenes Kommando verwenden.

Erwartung:

- Tool-Datei wird erzeugt.
- Tool wird registriert.
- Tool erscheint im Tool Center.
- Tool ist nicht nur Code im Text, sondern Teil des Systems.

Wenn hier nichts passiert:  
**Problem im Handoff Proposal → Tool Factory / Evolution Factory.**

---

## 7. Tool direkt testen

Mögliche Testeingaben:

```text
Ist 7 eine Primzahl?
```

Erwartung:

```text
Ja, 7 ist eine Primzahl.
```

```text
Ist 12 eine Primzahl?
```

Erwartung:

```text
Nein, 12 ist keine Primzahl.
```

```text
Gib mir alle Primzahlen bis 30.
```

Erwartung:

```text
2, 3, 5, 7, 11, 13, 17, 19, 23, 29
```

Wenn Pandora wieder nur allgemein antwortet und das Tool nicht nutzt:  
**Problem in Tool Selection / Tool Registry / Routing.**

---

## 8. Observation prüfen

Nach mehreren Tool-Aufrufen prüfen:

```powershell
python main.py observation status
python main.py observation events
```

Oder GUI:

```text
Maintenance → Observation → Events
```

Erwartung:

Events wie:

```text
tool_invoked
tool_success
tool_failed
tool_runtime
```

mit Bezug auf das Primzahlen-Tool.

Wenn keine Events entstehen:  
**Problem in Tool-Ausführung → Observation Engine.**

---

## 9. Pattern Recognition prüfen

Mehrere erfolgreiche und ggf. fehlerhafte Aufrufe erzeugen.

Dann prüfen:

```powershell
python main.py pattern status
python main.py pattern list
```

Erwartung bei normalen Aufrufen:

```text
Pattern: tool frequently used
Confidence: low/medium
```

Bei simulierten Fehlern:

```text
Pattern: repeated tool failure
Confidence: medium/high
```

Wenn keine Patterns entstehen:  
**Problem Observation → Pattern Recognition.**

---

## 10. Tool Evolution prüfen

GUI:

```text
Maintenance → Tool Evolution
```

Oder CLI:

```powershell
python main.py tools evolution status
python main.py tools health
python main.py tools review
```

Erwartung:

Das neue Tool erscheint mit:

- Name
- Status
- Health Score
- Anzahl Aufrufe
- Erfolgsquote
- Fehlerrate
- letzte Nutzung
- Review-Status

Beispiel:

```text
prime_number_tool
Status: ACTIVE / EXPERIMENTAL
Health: 95
Runs: 6
Success: 6
Failures: 0
Recommendation: none
```

Wenn das Tool hier nicht erscheint:  
**Problem Tool Registry → Tool Evolution.**

---

## 11. Fehler absichtlich provozieren

Im Chat testen:

```text
Ist -5 eine Primzahl?
```

```text
Gib mir alle Primzahlen bis abc.
```

```text
Gib mir alle Primzahlen bis 1000000000.
```

Erwartung:

Das Tool soll sauber reagieren:

- keine Exception im User-Chat
- klare Fehlermeldung
- Event wird geloggt
- Health sinkt ggf. leicht
- Tool Review erkennt schlechte Eingabe/Performance-Grenze

Wenn die App abstürzt:  
**Problem in Tool-Safety / Input Validation.**

---

## 12. Refactoring Candidate prüfen

Nach absichtlich schlechten Tests prüfen:

```text
Maintenance → Tool Evolution → Refactoring Candidates
```

Oder CLI:

```powershell
python main.py tools review
```

Mögliche erwartete Empfehlung:

```text
Add input validation for non-integer values.
Add upper range limit.
Improve error messages.
```

Wichtig:

Es darf **nicht automatisch** geändert werden.

Erwartung:

```text
Recommendation only
Proposal optional
User approval required
```

---

## 13. Proposal aus Tool Evolution erzeugen

Falls Button vorhanden:

```text
Generate Improvement Proposal
```

Erwartung in Proposal Queue:

```text
Type: TOOL
Title: Improve Prime Number Tool Input Validation
Source: Tool Evolution
Status: DRAFT / REVIEW
```

Wenn direkt Code geändert wird:  
**Falsch. Controlled Evolution verletzt.**

---

## 14. Erfolgreicher Endzustand

Der komplette Prozess gilt als bestanden, wenn:

- Chat erkennt Tool-Bedarf.
- Capability Gap wird erkannt.
- Proposal wird erzeugt.
- Proposal landet in der Queue.
- Benutzer kann freigeben.
- Tool wird erzeugt und registriert.
- Tool wird im Chat genutzt.
- Tool-Aufrufe werden beobachtet.
- Tool Evolution zeigt Health/Review.
- Fehler führen zu Empfehlungen.
- Verbesserungen werden nur als Proposal vorgeschlagen.
- Keine automatische Core-/Tool-Änderung ohne Freigabe.

---

## 15. Kritische Abbruchpunkte

### Abbruchpunkt 1

```text
Chat antwortet nur allgemein.
```

Ursache wahrscheinlich:

- Intent Detection
- Tool Need Detection
- Capability Gap Routing

---

### Abbruchpunkt 2

```text
Proposal entsteht nicht.
```

Ursache wahrscheinlich:

- Evolution Factory
- Proposal Generator
- Proposal Queue Integration

---

### Abbruchpunkt 3

```text
Tool wird nicht erzeugt.
```

Ursache wahrscheinlich:

- Approval Workflow
- Tool Factory
- Evolution Factory Handoff

---

### Abbruchpunkt 4

```text
Tool existiert, wird aber nicht genutzt.
```

Ursache wahrscheinlich:

- Tool Registry
- Adaptive Tool Selection
- Router

---

### Abbruchpunkt 5

```text
Tool wird genutzt, aber nicht beobachtet.
```

Ursache wahrscheinlich:

- Observation Hooks fehlen in Tool Execution

---

### Abbruchpunkt 6

```text
Observation vorhanden, aber keine Tool Evolution.
```

Ursache wahrscheinlich:

- Tool Evolution liest Events/Registry nicht korrekt
- Tool Health Engine nicht verbunden

---

## 16. Minimaler CLI-Test ohne GUI

Falls GUI unklar ist:

```powershell
python main.py selftest integration
python main.py proposal-queue list
python main.py evolution-factory preview --type TOOL --title "Prime Number Tool"
python main.py proposal-queue list
python main.py tools evolution status
python main.py tools health
```

Danach im Chat testen:

```text
Ich brauche ein Tool, das Prim-Zahlen berechnet.
```

---

## 17. Meine Einschätzung

Dieser Test ist bewusst hart.

Wenn Pandora hier „auf Grund läuft“, ist das gut sichtbar und wertvoll.

Dann wissen wir genau, welcher Übergang fehlt:

```text
Chat
↓
Capability Gap
↓
Proposal
↓
Queue
↓
Approval
↓
Tool Generation
↓
Tool Registry
↓
Tool Usage
↓
Observation
↓
Pattern
↓
Priority
↓
Tool Evolution
↓
Improvement Proposal
```

Das ist der wichtigste Integrationstest für die gesamte Evolution Architecture.
