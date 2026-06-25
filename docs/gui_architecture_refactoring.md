# MVP 23.2.1 – GUI Architecture Refactoring

Ziel: Pandora bekommt eine klare GUI-Struktur, bevor weitere Feature-Seiten entstehen.

## Hauptnavigation

Maximal fünf Hauptbereiche:

- Chat
- Knowledge
- Capabilities
- Operations
- Profiles

## Regeln

1. Keine doppelten Navigationsbuttons im Header und Body.
2. Pro Seite ist maximal ein Hauptbereich als `primary` markiert.
3. Detailseiten verwenden neutrale Unterbereich-Links.
4. Neue Funktionen werden zuerst einem Hauptbereich zugeordnet.
5. Warnungen, offene Reviews oder kritische Zustände dürfen hervorgehoben werden – normale Navigation nicht.

## Bereichszuordnung

### Knowledge

- Knowledge Base
- Knowledge Editor
- Memory Explorer

### Capabilities

- Capability Explorer
- Tool Center
- Skill Center
- Approval Center

### Operations

- Operations Dashboard
- Night Mode
- Maintenance / Governance / Reports

### Profiles

- LLM & Profile Center
- Routing Editor
- Providerstatus

## Zweck

Dieses Refactoring reduziert Verwirrung, verhindert Navigationswildwuchs und schafft eine saubere Struktur für kommende MVPs.
