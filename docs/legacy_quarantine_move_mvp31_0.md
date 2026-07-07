# MVP 31.0 – Legacy Quarantine Move

Status: BUILD-MVP.

## Ergebnis
- Verschobene Dateien: **8**
- Fehlende Dateien: **0**
- Gelöschte Dateien: **0**

## Verschoben nach `legacy/core/`

- `core/chat_response_router.py` → `legacy/core/chat_response_router.py`
- `core/observation/detectors/capability_detector.py` → `legacy/core/observation/detectors/capability_detector.py`
- `core/observation/detectors/gui_detector.py` → `legacy/core/observation/detectors/gui_detector.py`
- `core/observation/detectors/memory_detector.py` → `legacy/core/observation/detectors/memory_detector.py`
- `core/observation/detectors/review_detector.py` → `legacy/core/observation/detectors/review_detector.py`
- `core/observation/detectors/runtime_detector.py` → `legacy/core/observation/detectors/runtime_detector.py`
- `core/observation/detectors/tool_detector.py` → `legacy/core/observation/detectors/tool_detector.py`
- `core/observation/detectors/workflow_detector.py` → `legacy/core/observation/detectors/workflow_detector.py`

## Regeln

- Nur Kategorie-D-Kandidaten aus MVP 30.11 wurden verschoben.
- Keine fachliche Router-Entscheidung wurde eingeführt.
- Tools/Skills/Capability-Gap/Evolution bleiben deaktiviert.
- Bei Problemen können die Dateien aus `legacy/core/` zurückkopiert werden.
