# Rollback

Rollback bleibt ein geschützter Core-Bereich. Neue Versionen dürfen nicht ungeprüft aktiv werden.


## MVP 17 – Core Versioning & Rollback

MVP 17 führt Core-Snapshots, Smoke-Tests, Versionierungsindex und Rollback-Markierungen ein.

Wichtig:
- Neue Versionen werden als Snapshot gespeichert.
- Smoke-Tests prüfen Heartbeat, ToolRegistry, ToolExecutor und SkillRegistry.
- Aktivierung setzt den Status auf ACTIVE, ersetzt aber keine Dateien unkontrolliert.
- Rollback markiert eine stabile Zielversion. Physische Wiederherstellung bleibt bewusst manuell.
