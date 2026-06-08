# MVP 21.7 – Proposal Approval Workflow

Der Proposal Approval Workflow macht aus der Review Inbox einen echten Human-in-the-loop Prozess.

## Ziel

Pandora darf Vorschläge sammeln, bewerten und zur Entscheidung vorlegen. Eine Entscheidung bedeutet aber **keine automatische Ausführung**.

## Neue Komponente

```text
core/proposal_approval_workflow.py
```

## Unterstützte Entscheidungen

```text
approve_next_step  Vorschlag darf in den nächsten kontrollierten Schritt gehen
needs_work         Vorschlag muss überarbeitet werden
reject             Vorschlag wird abgelehnt
defer              später prüfen
reviewed           geprüft, ohne Folgeaktion
```

## Sicherheitsregel

Auch bei `approve_next_step` gilt:

```text
activation_performed = false
execution_allowed = false
auto_changes_made = false
```

Eine spätere Tool-, Skill- oder Core-Aktivierung bleibt ein separater, kontrollierter Prozess.

## CLI

```bash
python main.py approval-status
python main.py approval-pending
python main.py approval-decide <item_id> --decision approve_next_step --note "freigegeben für nächsten Schritt"
python main.py approval-audit
```

## Besonderheit bei hohem Risiko

High- oder Critical-Risk-Vorschläge können nicht ohne Begründung für den nächsten Schritt freigegeben werden.

```bash
python main.py approval-decide <item_id> --decision approve_next_step
```

wird blockiert, wenn kein `--note` gesetzt ist.
