# Minimum Viable Agent (MVA) — Bootcamp QA Día 01

Agente que asiste a un ingeniero de QA: analiza un Work Item (WI) a partir de 3 fuentes que se contradicen y genera un plan de pruebas validado con Pydantic. Equipo: Darío Tabares, David Buitrago, Diego Villarraga (responsabilidad colectiva, sin silos).

## Comandos

- `uv sync` — instalar dependencias (Python 3.12, gestionado por uv)
- `uv run pytest -v` — correr la suite (debe pasar al 100%)
- `uv run python src/main_mva.py` — ejecutar el agente
- Tras `uv add <pkg>`, regenerar: `uv export --no-hashes --no-header --no-emit-project --format requirements-txt > requirements.txt`

## Estructura (proyecto uv bare, plano según la guía)

- `src/` sin paquete: los imports son directos (`from schemas import QATestPlanOutput`); pytest tiene `pythonpath = ["src"]`.
- `src/schemas.py` — `TestCaseItem` y `QATestPlanOutput`
- `src/logger.py` — Audit Trail: trace_id, timestamp, reasoning_trace, tools, output
- `src/tools_and_rag.py` — `fetch_tfs_and_git_data(wi_id) -> str` y `search_meet_transcript_rag(wi_id, query) -> str`
- `src/main_mva.py` — agente con Instruction Contract (IC) y bucle FCoT (Recap → Reason → Verify)
- `mocks/wi_<id>_*.json` — datos de prueba. **Nunca quemar datos en el código**: siempre leerlos de `mocks/`.
- `tests/test_mva_day1.py` — solo las 5 pruebas de aceptación, contra el LLM real (`uv run pytest -v -m llm`).
- `tests/test_unitarias.py` — tests unitarios de cada pieza, con LLM simulado y embeddings falsos (sin API key).

## Contratos (no cambiar sin acuerdo del equipo)

- `TestCaseItem`: `test_id: str`, `category: Literal["functional","ui_visual","edge_case","bug_hunt"]`, `given_context`, `when_action`, `then_expected: str`, `grounding_source: Literal["TFS","MEET_PO_DEMO","GIT_DIFF_DISCREPANCY"]`
- `QATestPlanOutput`: `wi_id: str`, `status: Literal["READY_FOR_QA","ESCALATE_TO_HUMAN_MISSING_DATA"]`, `confidence_score: float = Field(ge=0.0, le=1.0)`, `detected_discrepancies: List[str]`, `human_escalation_reason: Optional[str]`, `test_cases: List[TestCaseItem]`
- `fetch_tfs_and_git_data` devuelve `{"error": "Work Item not found"}` si no existe el archivo, nunca un traceback.

## Reglas de negocio

1. Lo que dice la PO en Meet tiene prioridad sobre TFS, pero la contradicción se registra en `detected_discrepancies`.
2. Si el git diff contradice Meet o la regla de negocio → discrepancia + caso `bug_hunt` o `edge_case`.
3. Datos faltantes o corruptos → `confidence_score < 0.85`, `ESCALATE_TO_HUMAN_MISSING_DATA`, sin inventar pruebas.
4. Si algo no está en ninguna fuente, decirlo explícitamente; nunca alucinar.

## Pruebas de aceptación (deben cubrirse en `tests/test_mva_day1.py`)

1. WI-4052 → `READY_FOR_QA`, confianza ≥ 0.85, 3 trampas: recargo 8.5% (Meet) vs 5% (TFS) con `MEET_PO_DEMO`; `days_overdue >= 30` en vez de `> 30` (probar día 30 exacto); `disablePayButton={false}` vs bloqueo de 3 s pedido en Meet.
2. WI-4099 → `ESCALATE_TO_HUMAN_MISSING_DATA`, confianza < 0.85 (audio de Meet corrupto + commit null).
3. "¿Cuántos segundos dura el spinner al pagar con Visa?" (WI-4052) → responder que no existe en las fuentes.
4. WI-9999 → error controlado, sin traceback.
5. Logs muestran Recap → Reason → Verify y `pytest -v` en verde.

## Progreso

Estado por nivel de madurez, decisiones y pendientes: @.claude/progreso.md
