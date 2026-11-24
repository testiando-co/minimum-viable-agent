# Progreso del MVA por nivel de madurez

Referencia: **GenAI Maturity Model** (niveles 0–6), capítulo 1 de *Agentic Architectural Patterns
for Building Multi-Agent Systems*. Cada nivel alcanzado tiene un tag anotado en git
(`git show nivel-N` explica qué cubre).

| Nivel | Tag | Qué cubre en el repo |
|---|---|---|
| 0 — Data foundation | `nivel-0` | Mocks válidos; esquemas Pydantic de entrada (`WorkItemFile`, `extra="forbid"`); `missing_or_corrupt_sources()`; `fetch_tfs_and_git_data` con error controlado |
| 1 — Modelo + prompt + servicio | `nivel-1` | `init_chat_model`, Instruction Contract, `with_structured_output(method="json_schema")`, Audit Trail |
| 2 — RAG | `nivel-2` | WI descubiertos desde `mocks/`; RAG híbrido (FAISS — hoy Chroma — + BM25, reranking con fecha, k=5, chunks de 200 con 20 de solapamiento, metadata `source_file`/`context`/`date`) |
| 3 — Tuning | — | Omitido a propósito (no aplica a este MVA) |
| 4 — Grounding y evaluación | `nivel-4` | Citas verificadas contra la fuente; confianza por reglas; FCoT RECAP → REASON → VERIFY; validadores del plan; `evals/golden_set.json` + `src/evaluation.py`; proveedor elegible desde `.env` |
| 5 — Agente único | (en desarrollo, sin tag) | `src/agent_loop/` con LangGraph: bucle sense → reason (LLM) → plan (LLM) → act ↺ hasta el objetivo, FCoT fractal; memoria de corto plazo (InMemorySaver) y largo plazo (Chroma); RAG en Chroma persistente; una traza de LangSmith por ejecución |

## Arquitectura actual

- `src/agent_loop/`: bucle de LangGraph. `sense` (RECAP: WI, fuentes, memoria, observaciones y
  feedback) → `reason` (LLM: `Assessment`) → `plan` (LLM: `ActionPlan` con herramientas de
  `tools.py`) → `act` (ejecuta cada paso con RECAP → REASON → VERIFY; redacta con el LLM;
  `verify.check_goal` decide con reglas si termina o vuelve a `sense`). Máx. `MVA_MAX_ITERATIONS`
  (3). `memory.py`: corto plazo con `InMemorySaver` (campo `history`, un thread por sesión) y
  largo plazo en Chroma (`mva_long_term_memory`). `prompts.py`: IC. `llm.py`: proveedor.
- `src/main_mva.py`: CLI y API pública; la CLI es una sola traza (`MVA CLI`).
- `src/schemas.py`: entrada (mocks), contratos con validadores y borradores del LLM con
  `evidence_quote`. `CONFIDENCE_THRESHOLD = 0.85`.
- `src/tools_and_rag.py`: carga de mocks, `list_work_item_ids`, RAG híbrido con Chroma persistente
  (`chroma.db/`, se reindexa solo si cambia la huella) + BM25; herramientas con `@traceable`.
- `src/grounding.py`: `quote_in_source`, `verify_test_cases`, `compute_confidence`.
- `src/evaluation.py`: métricas; cada caso en su propia sesión; una sola traza (`MVA evaluación`).
- `src/logger.py`: Audit Trail en `logs/audit.jsonl`.

`agent_loop/`, `grounding.py`, `evaluation.py` y `evals/` no están en la estructura obligatoria de
la guía.

## Decisiones clave

- El LLM **no** decide `status` ni `confidence_score`: los calculan reglas verificables.
- El bucle sigue la anatomía del cap. 4 (Reason y Plan con el LLM, Act ejecuta herramientas y
  su resultado se percibe en la siguiente vuelta) y el FCoT del cap. 13 (IC en cada llamada y
  RECAP → REASON → VERIFY por vuelta y por paso, con `level` en el Audit Trail). ACT solo llama
  al LLM en los pasos de redacción: 3 llamadas por vuelta en el camino feliz.
- ACT ejecuta un paso del plan por nodo (`act` ↺ `act`): el checkpointer guarda `step_index` y
  `completed_steps` tras cada paso. Ante un error del LLM, `run_agent` espera
  `MVA_PLAN_RETRY_DELAY` (10 s por defecto, luego el doble; 2 reintentos) y retoma con
  `graph.invoke(None, config)` desde el paso pendiente, sin repetir herramientas ni llamadas.
- Las unitarias fijan `MVA_MAX_ITERATIONS=3` y `MVA_PLAN_RETRY_DELAY=0` en `conftest.py`: no
  dependen del `.env`.
- Que el objetivo se alcanzó lo deciden reglas (grounding y confianza), no el LLM; si no, el
  feedback de VERIFY (casos descartados y por qué) vuelve a REASON y PLAN en la vuelta siguiente.
- Escalar a un humano lo decide el agente (LLM con `escalate_to_human`); las reglas son una
  barrera: si la confianza no llega a 0.85 se escala igual, y una fuente corrupta nunca sirve de
  respaldo para un caso.
- Un caso sin cita verificable se descarta; si queda confianza < 0.85, se escala sin casos.
- Pregunta sin respaldo verificable → `UNVERIFIED_ANSWER` ("No pude verificar… no estoy seguro de
  que exista"), sin agregar el texto del LLM.
- La memoria es solo contexto: nunca es fuente ni se acepta como cita en el grounding.
- Los nodos que llaman al LLM pasan el `config` del grafo; si no, LangSmith abre otra traza.
- El estado del grafo guarda solo dicts/listas (serializables por el checkpointer).
- Anthropic con Opus 5.5 requiere `method="json_schema"` (rechaza el *forced tool calling*).
- Tests unitarios (`tests/test_unitarias.py`) con LLM simulado (`FakeStructuredLLM`), embeddings
  falsos y Chroma en carpeta temporal (`tests/conftest.py`); las 5 de aceptación
  (`tests/test_mva_day1.py`) usan el modelo real, llevan `@pytest.mark.llm` y se saltan si el
  proveedor no está configurado.
- En LangSmith, cada prueba de aceptación es una traza `pytest <test>` (tags `pytest`/`aceptacion`)
  con el agente anidado; las unitarias nunca envían trazas (lo comprueba un test).

## Proveedores probados (5 criterios de aceptación con el modelo real)

Modelo por defecto: **`openrouter:openrouter/auto`** (en el código y en `.env.example`, con
`OPENROUTER_API_KEY`). OpenRouter elige el modelo en cada llamada y, con
`openrouter_provider={"require_parameters": True}`, solo enruta a modelos que soportan
`response_format` (json_schema). Como el modelo puede variar entre llamadas, los resultados
también. Aún sin corrida de las 5 de aceptación con este modelo.

| Modelo | Resultado | Notas |
|---|---|---|
| `google_genai:gemini-3.6-flash` | 5/5; trampas con criterio estricto 3/3 | **Recomendado.** Día 30 → "NO se aplica recargo" y trampa 3 como `bug_hunt` desde el diff en todas las corridas. ~15 s por plan |
| `google_genai:gemini-4-hydrogen` | Trampas con criterio estricto 3/3 | Igual de bien, pero ~45 s por plan y parece experimental (límites más estrictos) |
| `google_genai:gemini-3.5-flash` | Trampas con criterio estricto 2/3 | Una vez redactó mal el día 30 |
| `google_genai:gemini-3.5-flash-lite` | 5/5 en pytest, pero día 30 correcto 0/3 | Escribe "se aplica el recargo" el día 30 (valida el bug); a veces copia "seja" por "sea" y el verificador descarta el caso del banner (1 de 12 corridas escaló). Ignora `temperature` |
| `ollama:gemma4` | 4/5 | Falló la trampa 1 por `[...]` en la cita; ya corregido en el verificador |
| `ollama:gemma3` | 3/5 | No llena discrepancias; respondió el spinner con un dato de otro tema |

Con el bucle FCoT (2026-11-23, `gemini-3.5-flash-lite`): 5/5 en ~56 s, todas en **1 vuelta**.
WI-4052 → `READY_FOR_QA` con 4/4 citas verificadas (confianza 1.0) y la trampa 3 como `bug_hunt`
desde el git diff; WI-4099 → el agente decidió escalar; spinner → "no pude verificar". Lo que
aporta el FCoT: PLAN eligió reunir TFS/Git y Meet antes de redactar, y cada paso quedó verificado
en el Audit Trail. Lo que **no** corrigió: el caso del día 30 sigue diciendo "se aplica el
recargo" (las reglas verifican la cita, no el sentido del `then_expected`). Una cita mal copiada
ahora da otra vuelta con feedback en vez de escalar de inmediato (cubierto por tests unitarios; no
ocurrió en esta corrida).

**Límites del plan gratuito**: los modelos gratis limitan las solicitudes por minuto y por día
(p. ej. `gemini-3.5-flash-lite` permite unas 15 RPM) y pueden responder 429/503. Con el bucle, cada
WI usa ≥ 3 llamadas por vuelta: las 5 de aceptación son ~12 llamadas en ~1 min, cerca del límite.
Una corrida lenta o un 429 suele ser el límite, no un fallo del agente; el límite no cambia la
calidad de las respuestas (el error del día 30 es del modelo).

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
- `langchain-community` (FastEmbed) emite un aviso de *sunset*.
- Datos quemados en los tests (citas copiadas de los mocks): se dejó para después.
- La memoria de largo plazo crece sin límite y puede influir en evaluaciones repetidas.
- Falta la explicación por archivo del concepto del Día 01 que aplica (pedida en `INSTRUCTIONS.md`
  para el video).
- El checkpointer es en memoria: un reintento retoma dentro del mismo proceso, pero no después
  de cerrar la CLI (haría falta un checkpointer persistente, p. ej. SQLite).
- PR `dev` → `main` pendiente.
