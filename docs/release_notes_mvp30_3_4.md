# MVP 30.3.4 – Vault First Stabilization

## Ziel

Dieser Hotfix stoppt den Feature-Ausbau und stabilisiert zuerst die beiden Basiswege:

1. sichere Vault-/Knowledge-Nutzung für Fragen zu gespeichertem Wissen
2. normaler LLM-Chat für allgemeine Fragen

Vorhandene Tools bleiben nutzbar, werden aber durch einfache Contract Guards geschützt. Capability-Gap und Tool-Entwicklung bleiben bewusst nachgelagert.

## Änderungen

- `ChatService._build_guarded_knowledge_context(...)`
  - lädt bei explizitem Kontextbedarf weiterhin Vault/Knowledge
  - führt bei `answer_directly` eine begrenzte, policy-sichere Knowledge-Suche als Safety-Net aus
  - wertet relevante Treffer als `answer_with_context`
  - lädt weiterhin keinen Kontext für Non-Chat-Routen

- `ActionPlanner`
  - schützt die Calculator-Capability vor natürlicher Sprache als `expression`
  - führt Calculator nur aus, wenn ein echter arithmetischer Ausdruck vorliegt
  - gibt bei ungeeignetem Calculator-Einsatz eine sichere Antwort statt SyntaxError

## Bewusst nicht geändert

- keine neue Capability-Gap-Logik
- keine neue Tool-Factory-Logik
- keine Sidebar / Live Console
- kein neues Feature-MVP

## Tests

`21 passed`

Neue Regressionstests:

- `answer_directly` wird bei relevanten Knowledge-Treffern zu `answer_with_context`
- `answer_directly` bleibt normaler Chat, wenn keine relevanten Treffer gefunden werden
- Calculator lehnt freie Sprache als Payload ab
- Calculator akzeptiert echte Ausdrücke wie `2+3*4`
