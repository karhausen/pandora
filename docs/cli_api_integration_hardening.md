# MVP 28.9.2 – CLI & API Integration Hardening

## Ziel

Dieser Release stabilisiert die Integration der neuen Evolution-Komponenten in CLI und API.

Der Auslöser war, dass dokumentierte Befehle wie

```bash
python main.py genome status
python main.py proposal-queue add --type TOOL --title "Test Tool Proposal" --priority MEDIUM
```

noch nicht vollständig in `main.py` eingehangen waren.

## Neu

### Saubere CLI-Kommandos

Folgende dokumentierte Schreibweisen sind jetzt unterstützt:

```bash
python main.py genome status
python main.py evolution status
python main.py evolution-factory status
python main.py observation status
python main.py pattern status
python main.py priority status
python main.py proposal-queue status
python main.py proposal-queue list
python main.py proposal-queue add --type TOOL --title "Test Tool Proposal" --priority MEDIUM
python main.py proposal-queue from-factory "Tool verbessern" --type TOOL
python main.py proposal-queue decide <queue_id> --decision deferred
```

Die alten flachen Kommandos wie `proposal-queue-status` bleiben weiterhin kompatibel.

### Proposal Queue Add

`proposal-queue add` erzeugt ein normales `EvolutionProposal` über die Evolution Factory und legt es danach in die Unified Proposal Queue.

Wichtig:

- keine automatische Aktivierung
- keine Runtime-Dateiänderung
- keine Core-Änderung
- Benutzerfreigabe bleibt Pflicht

### Selbsttests

Neue Selftest-Kommandos:

```bash
python main.py selftest cli
python main.py selftest api
python main.py selftest integration
```

Diese prüfen, ob dokumentierte CLI-Kommandos und wichtige API-Routen registriert sind.

### API-Selbsttest

Neue API-Endpunkte:

```text
/api/integration/status
/api/selftest/api
```

## Architekturprinzip

Ab diesem Release müssen neue MVPs ihre CLI/API-Verträge testbar machen. Ein MVP gilt erst als sauber integriert, wenn der Integration-Selftest grün ist.
