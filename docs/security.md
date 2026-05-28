# Sicherheit

Der aktive Core wird nicht unkontrolliert überschrieben. Kritische Änderungen benötigen explizite Freigabe.


## MVP 15 – Sandbox & Isolation

Tools laufen standardmäßig über den Sandbox-Layer:

- Ausführung in separatem Python-Subprozess
- Timeout je Execution Policy
- Sandbox-Logging
- Policy-Stufen: trusted, restricted, isolated, dangerous
- Netzwerk und Shell bleiben standardmäßig deaktiviert

Noch offen:

- harte CPU-/RAM-Limits
- Betriebssystem-spezifische Benutzer-/Container-Isolation
- feinere Dateisystem-Policies
