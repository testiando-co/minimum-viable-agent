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

Modelo en uso: **`google_genai:gemini-3.6-flash`** (desde el 2026-11-23, en `.env` y `.env.example`).

| Modelo | Resultado | Notas |
|---|---|---|
| `google_genai:gemini-3.6-flash` | 5/5; trampas con criterio estricto 3/3 | **Recomendado.** Día 30 → "NO se aplica recargo" y trampa 3 como `bug_hunt` desde el diff en todas las corridas. ~15 s por plan |
| `google_genai:gemini-4-hydrogen` | Trampas con criterio estricto 3/3 | Igual de bien, pero ~45 s por plan y parece experimental (límites más estrictos) |
| `google_genai:gemini-3.5-flash` | Trampas con criterio estricto 2/3 | Una vez redactó mal el día 30 |
| `google_genai:gemini-3.5-flash-lite` | 5/5 en pytest, pero día 30 correcto 0/3 | Escribe "se aplica el recargo" el día 30 (valida el bug); a veces copia "seja" por "sea" y el verificador descarta el caso del banner (1 de 12 corridas escaló). Ignora `temperature` |
| `ollama:gemma4` | 4/5 | Falló la trampa 1 por `[...]` en la cita; ya corregido en el verificador |
| `ollama:gemma3` | 3/5 | No llena discrepancias; respondió el spinner con un dato de otro tema |

Criterio estricto (2026-11-23, 3 corridas por modelo): además de lo que exige `golden_set.json`, el
caso del día 30 debe decir que **no** se aplica recargo y la trampa 3 debe venir del git diff
(`GIT_DIFF_DISCREPANCY`). Con la key gratuita: `gemini-3.7-flash` y `3.8-flash` respondieron 503,
`gemini-4-argon`/`carbon`/`neon` 429 (fuera del plan gratuito) y `gemini-2.5-*` 404 (retirados).

## Pendientes conocidos

- La trampa 3 depende del comentario "BUG OCULTO": el mock no contiene `disablePayButton={false}`.
- La evaluación es permisiva: no comprueba que el caso del día 30 diga que **no** se aplica recargo
  ni que la trampa 3 venga del git diff (busca palabras, no razonamiento). Con gemini-3.6-flash
  sale bien igual, pero pytest no detectaría una corrida mala.
- `temperature` solo se fija con Ollama; con Gemini y Anthropic los resultados varían entre corridas.
- La fecha de la metadata del RAG es la de modificación del archivo; tras un `git clone` todas
  coinciden y el factor de fecha no influye hasta ingresarlas a mano.
- `langchain-community` (FAISS, FastEmbed) emite un aviso de *sunset*.
- Falta la explicación por archivo del concepto del Día 01 que aplica (pedida en `INSTRUCTIONS.md`
  para el video).
- PR `dev` → `main` pendiente.
