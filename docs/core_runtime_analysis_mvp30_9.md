# MVP 30.9 – Core Runtime Analysis

Status: **ANALYZE-MVP**. Dieser Bericht verändert keine Core-Runtime-Dateien.

## Regeln

- ANALYZE-MVP: no core runtime files are modified by this report.
- Static analysis only: do not delete or move files based solely on this report.
- Router must remain dispatcher-only; this report does not re-enable tools/capability gap/evolution.

## Entry Points

- `main`
- `core.api`

## Ergebnisübersicht

- Python-Dateien gesamt: **291**
- Core-Module gesamt: **266**
- Von Entry Points statisch erreichbar: **181**
- Nicht von Entry Points statisch erreichbar: **85**
- Legacy-Kandidaten, statisch/konservativ: **26**

## Wichtige Einschränkung

Diese Analyse ist statisch. Sie erkennt normale `import`/`from ... import ...`-Beziehungen, aber keine dynamischen Imports, CLI-Pfade, Plugin-Loader oder String-basierte Modulaufrufe. Ein Legacy-Kandidat darf deshalb **nicht automatisch gelöscht** werden.

## Legacy-Kandidaten – statisch, nicht importiert und nicht erreichbar

- `core/action_proposal_engine.py`
- `core/capability_registry.py`
- `core/capability_relationships.py`
- `core/chat_response_router.py`
- `core/execution_context.py`
- `core/observation/detectors/capability_detector.py`
- `core/observation/detectors/gui_detector.py`
- `core/observation/detectors/memory_detector.py`
- `core/observation/detectors/review_detector.py`
- `core/observation/detectors/runtime_detector.py`
- `core/observation/detectors/tool_detector.py`
- `core/observation/detectors/workflow_detector.py`
- `core/obsidian_export.py`
- `core/obsidian_indexer.py`
- `core/obsidian_search.py`
- `core/operations_issue_workflow.py`
- `core/prioritization/evaluators/benefit.py`
- `core/prioritization/evaluators/confidence.py`
- `core/prioritization/evaluators/effort.py`
- `core/prioritization/evaluators/frequency.py`
- `core/prioritization/evaluators/risk.py`
- `core/prioritization/evaluators/urgency.py`
- `core/prioritization/evaluators/user_value.py`
- `core/recovery.py`
- `core/resource_monitor.py`
- `core/security.py`

## Nicht erreichbare Core-Module – statischer Graph

- `core/__init__.py` — imported_by: —
- `core/action_proposal_engine.py` — imported_by: —
- `core/adaptive_goals/__init__.py` — imported_by: —
- `core/adaptive_goals/adaptive_goal_manager.py` — imported_by: `core.adaptive_goals.__init__`
- `core/capability_registry.py` — imported_by: —
- `core/capability_relationships.py` — imported_by: —
- `core/chat_response_router.py` — imported_by: —
- `core/core_evolution/__init__.py` — imported_by: —
- `core/core_evolution/core_evolution_manager.py` — imported_by: `core.core_evolution.__init__`
- `core/decision_learning/__init__.py` — imported_by: —
- `core/decision_learning/decision_learning_manager.py` — imported_by: `core.decision_learning.__init__`
- `core/decision_learning/decision_storage.py` — imported_by: `core.decision_learning.decision_learning_manager`
- `core/evolution_dashboard/__init__.py` — imported_by: —
- `core/evolution_dashboard/evolution_dashboard_manager.py` — imported_by: `core.evolution_dashboard.__init__`
- `core/execution_context.py` — imported_by: —
- `core/genome/__init__.py` — imported_by: —
- `core/genome/evolution_factory.py` — imported_by: `core.genome.__init__`, `core.genome.evolution_service`
- `core/genome/evolution_lifecycle.py` — imported_by: `core.genome.__init__`, `core.genome.evolution_proposal`, `core.genome.genome_manager`, `core.genome.genome_validator`
- `core/genome/evolution_proposal.py` — imported_by: `core.genome.__init__`, `core.genome.evolution_factory`, `core.genome.evolution_service`, `core.genome.genome_manager`
- `core/genome/evolution_service.py` — imported_by: `core.genome.__init__`
- `core/genome/genome.py` — imported_by: `core.genome.__init__`, `core.genome.genome_loader`, `core.genome.genome_manager`, `core.genome.genome_validator`
- `core/genome/genome_loader.py` — imported_by: `core.genome.genome_manager`
- `core/genome/genome_manager.py` — imported_by: `core.genome.__init__`, `core.genome.evolution_factory`, `core.genome.evolution_service`
- `core/genome/genome_rules.py` — imported_by: `core.genome.genome_manager`, `core.genome.genome_validator`
- `core/genome/genome_schema.py` — imported_by: `core.genome.genome_validator`
- `core/genome/genome_validator.py` — imported_by: `core.genome.__init__`, `core.genome.genome_manager`
- `core/knowledge_evolution/__init__.py` — imported_by: —
- `core/knowledge_evolution/knowledge_evolution_manager.py` — imported_by: `core.knowledge_evolution.__init__`
- `core/llm_clients/__init__.py` — imported_by: —
- `core/observation/__init__.py` — imported_by: —
- `core/observation/detectors/__init__.py` — imported_by: —
- `core/observation/detectors/capability_detector.py` — imported_by: —
- `core/observation/detectors/gui_detector.py` — imported_by: —
- `core/observation/detectors/memory_detector.py` — imported_by: —
- `core/observation/detectors/review_detector.py` — imported_by: —
- `core/observation/detectors/runtime_detector.py` — imported_by: —
- `core/observation/detectors/tool_detector.py` — imported_by: —
- `core/observation/detectors/workflow_detector.py` — imported_by: —
- `core/observation/event_bus.py` — imported_by: `core.observation.__init__`, `core.observation.observation_engine`
- `core/observation/event_logger.py` — imported_by: `core.observation.__init__`, `core.observation.event_bus`
- `core/observation/observation_engine.py` — imported_by: `core.observation.__init__`, `core.observation.observation_manager`
- `core/observation/observation_manager.py` — imported_by: `core.observation.__init__`
- `core/observation/observation_schema.py` — imported_by: `core.observation.__init__`, `core.observation.event_logger`, `core.observation.observation_storage`
- `core/observation/observation_storage.py` — imported_by: `core.observation.event_logger`, `core.observation.observation_engine`, `core.pattern.pattern_engine`
- `core/obsidian_export.py` — imported_by: —
- `core/obsidian_indexer.py` — imported_by: —
- `core/obsidian_search.py` — imported_by: —
- `core/operations_issue_workflow.py` — imported_by: —
- `core/pattern/__init__.py` — imported_by: —
- `core/pattern/pattern_detector.py` — imported_by: `core.pattern.pattern_engine`
- `core/pattern/pattern_engine.py` — imported_by: `core.pattern.__init__`, `core.pattern.pattern_manager`
- `core/pattern/pattern_manager.py` — imported_by: `core.pattern.__init__`
- `core/pattern/pattern_schema.py` — imported_by: `core.pattern.__init__`, `core.pattern.pattern_detector`, `core.pattern.pattern_storage`
- `core/pattern/pattern_storage.py` — imported_by: `core.pattern.pattern_engine`
- `core/prioritization/__init__.py` — imported_by: —
- `core/prioritization/evaluators/benefit.py` — imported_by: —
- `core/prioritization/evaluators/confidence.py` — imported_by: —
- `core/prioritization/evaluators/effort.py` — imported_by: —
- `core/prioritization/evaluators/frequency.py` — imported_by: —
- `core/prioritization/evaluators/risk.py` — imported_by: —
- `core/prioritization/evaluators/urgency.py` — imported_by: —
- `core/prioritization/evaluators/user_value.py` — imported_by: —
- `core/prioritization/prioritization_engine.py` — imported_by: `core.prioritization.__init__`, `core.prioritization.priority_manager`
- `core/prioritization/priority_manager.py` — imported_by: `core.prioritization.__init__`
- `core/prioritization/priority_schema.py` — imported_by: `core.prioritization.__init__`, `core.prioritization.prioritization_engine`, `core.prioritization.priority_storage`, `core.prioritization.scoring_engine`
- `core/prioritization/priority_storage.py` — imported_by: `core.prioritization.prioritization_engine`
- `core/prioritization/scoring_engine.py` — imported_by: `core.prioritization.prioritization_engine`
- `core/prioritization/scoring_models.py` — imported_by: `core.prioritization.prioritization_engine`, `core.prioritization.scoring_engine`
- `core/proposal_evolution/__init__.py` — imported_by: —
- `core/proposal_evolution/proposal_evolution.py` — imported_by: `core.proposal_evolution.__init__`, `core.proposal_evolution.proposal_evolution_manager`
- `core/proposal_evolution/proposal_evolution_manager.py` — imported_by: `core.proposal_evolution.__init__`
- `core/proposal_evolution/proposal_evolution_storage.py` — imported_by: `core.proposal_evolution.proposal_evolution`
- `core/proposal_generator/__init__.py` — imported_by: —
- `core/proposal_generator/proposal_generator.py` — imported_by: `core.proposal_generator.__init__`, `core.proposal_generator.proposal_generator_manager`
- `core/proposal_generator/proposal_generator_manager.py` — imported_by: `core.proposal_generator.__init__`
- `core/proposal_generator/proposal_prompt.py` — imported_by: `core.proposal_generator.proposal_generator`
- `core/proposal_queue/__init__.py` — imported_by: —
- `core/proposal_queue/queue_manager.py` — imported_by: `core.proposal_queue.__init__`
- `core/proposal_queue/queue_schema.py` — imported_by: `core.proposal_queue.__init__`, `core.proposal_queue.queue_manager`, `core.proposal_queue.queue_storage`
- `core/proposal_queue/queue_storage.py` — imported_by: `core.proposal_queue.__init__`, `core.proposal_queue.queue_manager`
- `core/recovery.py` — imported_by: —
- `core/resource_monitor.py` — imported_by: —
- `core/security.py` — imported_by: —
- `core/tool_evolution/__init__.py` — imported_by: —
- `core/tool_evolution/tool_evolution_manager.py` — imported_by: `core.tool_evolution.__init__`

## Nächste sinnvolle Aktion

1. Bericht prüfen.
2. Legacy-Kandidaten manuell klassifizieren: `active`, `cli_only`, `api_only`, `deprecated`, `unknown`.
3. Erst danach Dateien nach `legacy/` verschieben.
