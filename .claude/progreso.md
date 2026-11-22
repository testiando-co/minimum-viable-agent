# Progreso del MVA por nivel de madurez

Referencia: **GenAI Maturity Model** (niveles 0–6), capítulo 1 de *Agentic Architectural Patterns
for Building Multi-Agent Systems*. Cada nivel alcanzado tiene un tag anotado en git
(`git show nivel-N` explica qué cubre).

| Nivel | Tag | Qué cubre en el repo |
|---|---|---|
| 0 — Data foundation | `nivel-0` | Mocks válidos; esquemas Pydantic de entrada (`WorkItemFile`, `extra="forbid"`); `missing_or_corrupt_sources()`; `fetch_tfs_and_git_data` con error controlado |
| 1 — Modelo + prompt + servicio | `nivel-1` | `init_chat_model`, Instruction Contract, `with_structured_output(method="json_schema")`, Audit Trail |
| 2 — RAG | `nivel-2` | WI descubiertos desde `mocks/`; RAG híbrido (FAISS + BM25, reranking con fecha, k=5, chunks de 200 con 20 de solapamiento, metadata `source_file`/`context`/`date`) |
| 3 — Tuning | — | Omitido a propósito (no aplica a este MVA) |
| 4 — Grounding y evaluación | `nivel-4` | Citas verificadas contra la fuente; confianza por reglas; FCoT RECAP → REASON → VERIFY; validadores del plan; `evals/golden_set.json` + `src/evaluation.py`; proveedor elegible desde `.env` |

## Arquitectura actual

- `src/schemas.py`: entrada (mocks), contratos de salida con validadores y borradores del LLM con
  `evidence_quote`. `CONFIDENCE_THRESHOLD = 0.85`.
- `src/tools_and_rag.py`: carga de mocks, `list_work_item_ids`, RAG híbrido
  (`search_meet_transcript_rag`, `search_work_item_rag`), `get_embeddings` (FastEmbed u Ollama).
- `src/grounding.py`: `quote_in_source` (acepta `...`, `…`, `[...]`, `(...)`; ignora `[hh:mm:ss]`),
  `verify_test_cases`, `compute_confidence = 0.5·fuentes válidas/3 + 0.5·casos con respaldo/propuestos`.
- `src/main_mva.py`: Instruction Contract, `run_test_plan`/`generate_test_plan`,
  `run_question`/`answer_question`, selección de proveedor (`build_llm`, `llm_available`, `llm_errors`), CLI.
- `src/evaluation.py`: métricas `accuracy`, `check_pass_rate`, `grounding_rate`, `hallucination_guard`.
- `src/logger.py`: Audit Trail en `logs/audit.jsonl` (entradas y salidas completas de herramientas).

`grounding.py`, `evaluation.py` y `evals/` no están en la estructura obligatoria de la guía; se
agregaron para el Nivel 4.

## Decisiones clave

- El LLM **no** decide `status` ni `confidence_score`: los calculan reglas verificables.
- Datos faltantes o corruptos (WI-4099) escalan **sin invocar al LLM**.
- Un caso sin cita verificable se descarta; si queda confianza < 0.85, se escala sin casos.
- Pregunta sin respaldo verificable → `UNVERIFIED_ANSWER` ("No pude verificar… no estoy seguro de
  que exista"), sin agregar el texto del LLM.
- Anthropic con Opus 5.5 requiere `method="json_schema"` (rechaza el *forced tool calling*).
- Tests deterministas con LLM simulado (`FakeStructuredLLM`) y embeddings falsos; las variantes
  con el modelo real llevan `@pytest.mark.llm` y se saltan si el proveedor no está configurado.

## Proveedores probados (5 criterios de aceptación con el modelo real)

| Modelo | Resultado | Notas |
|---|---|---|
| `google_genai:gemini-3.5-flash-lite` | 5/5, evaluación 1.0 en todas las métricas | El plan gratuito a veces responde 503 por alta demanda |
| `ollama:gemma4` | 4/5 | Falló la trampa 1 por `[...]` en la cita; ya corregido en el verificador |
| `ollama:gemma3` | 3/5 | No llena discrepancias; respondió el spinner con un dato de otro tema |

## Pendientes conocidos

- La trampa 3 depende del comentario "BUG OCULTO": el mock no contiene `disablePayButton={false}`.
- La evaluación no comprueba que el caso del día 30 diga que **no** se aplica recargo.
- La fecha de la metadata del RAG es la de modificación del archivo; tras un `git clone` todas
  coinciden y el factor de fecha no influye hasta ingresarlas a mano.
- `langchain-community` (FAISS, FastEmbed) emite un aviso de *sunset*.
- Falta la explicación por archivo del concepto del Día 01 que aplica (pedida en `INSTRUCTIONS.md`
  para el video).
- PR `dev` → `main` pendiente.
