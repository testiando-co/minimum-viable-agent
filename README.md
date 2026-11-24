# Minimum Viable Agent (MVA) — QA

Agente que analiza un Work Item a partir de TFS, la demo de Meet con la PO y el git diff,
y genera un plan de pruebas validado con Pydantic (`QATestPlanOutput`).

## Nivel de madurez (GenAI Maturity Model)

| Nivel | Qué exige | Dónde está |
|---|---|---|
| 0 — Data foundation | Datos curados y accesibles | `mocks/*.json`, `src/tools_and_rag.py` |
| 1 — Modelo + prompt + servicio | Modelo elegido, prompt diseñado, servido vía API | `src/agent_loop/llm.py`, `src/agent_loop/prompts.py` (Instruction Contract) |
| 2 — RAG | Recuperar contexto relevante | `src/tools_and_rag.py` (`search_meet_transcript_rag`, `search_work_item_rag`) |
| 3 — Tuning | Fine-tuning / PEFT | Fuera de alcance (no aplica a este MVA) |
| 4 — Grounding y evaluación | Respuestas respaldadas por las fuentes y métricas | `src/grounding.py`, `src/evaluation.py`, `evals/golden_set.json` |
| 5 — Agente único | Agente orquestado, memoria, AgentOps (trazas) | `src/agent_loop/` (LangGraph), memoria en memoria + Chroma, LangSmith |

## Instalación

```bash
uv sync
cp .env.example .env   # completa la API key del proveedor que vayas a usar
```

### Proveedor del LLM

Se elige solo con `.env`:

| Proveedor | Configuración en `.env` |
|---|---|
| Google Gemini (recomendado, gratis) | `MVA_MODEL=google_genai:gemini-3.6-flash` + `GOOGLE_API_KEY` |
| Anthropic | `MVA_MODEL=anthropic:claude-opus-5-5` + `ANTHROPIC_API_KEY` (es el modelo por defecto si no se define `MVA_MODEL`) |
| Ollama local | `LOCAL_LLM=1` + `OLLAMA_MODEL` (p. ej. `gemma4`) y `OLLAMA_EMBEDDINGS_MODEL` (p. ej. `embeddinggemma`) |

Con `LOCAL_LLM=1`, Ollama también genera los embeddings del RAG; si no, se usa FastEmbed local.

**Los modelos gratuitos tienen límites.** El plan gratuito de Gemini limita las solicitudes por
minuto (RPM) y por día, y en horas de alta demanda puede responder `429` (límite alcanzado) o
`503` (sobrecarga). Con el bucle del agente cada WI hace al menos 3 llamadas al LLM por vuelta
(reason, plan y redacción), así que `uv run pytest -v -m llm` hace unas 12 llamadas en un minuto y
puede rozar el límite; una vuelta extra de autocorrección suma 3 más. Si aparece un `429`, espera
un minuto o cambia de modelo/API key en `.env`. La CLI muestra el error de forma controlada.
Los tests `@pytest.mark.llm` corren solo si el proveedor elegido está configurado.

Resultado de las 5 pruebas de aceptación con el modelo real (`uv run pytest -v`):

| Modelo | Resultado |
|---|---|
| `google_genai:gemini-3.6-flash` | 5/5, y las 3 trampas bien razonadas en todas las corridas (~15 s por plan) |
| `google_genai:gemini-3.5-flash-lite` | 5/5, pero redacta al revés el caso del día 30 ("se aplica el recargo") y a veces una cita mal copiada hace escalar el plan |
| `ollama:gemma4` | 4/5 (antes de aceptar `[...]` en las citas) |
| `ollama:gemma3` | 3/5 |

## Uso

```bash
uv run python src/main_mva.py                     # plan de pruebas de todos los WI en mocks/
uv run python src/main_mva.py WI-4052 WI-9999     # solo los WI indicados
uv run python src/main_mva.py WI-4052 -q "¿Cuántos segundos dura la animación del spinner al pagar con Visa?"
uv run pytest -v                                  # todo: unitarias + las 5 de aceptación
uv run pytest -v -m llm                           # solo las 5 pruebas de aceptación (tests/test_mva_day1.py)
uv run pytest -v tests/test_unitarias.py          # solo las unitarias (LLM simulado, sin API key)
uv run python src/evaluation.py                   # evaluación contra el conjunto dorado (llama al LLM)
```

Con `LANGSMITH_TRACING=true` en `.env`, cada prueba de aceptación queda como una traza en
LangSmith (`pytest <nombre del test>`, tags `pytest` y `aceptacion`) con la ejecución del agente
(`run_agent`: sense → reason → plan → act, en una o más vueltas) anidada y el resultado `passed`/`failed`. Las
pruebas unitarias nunca envían trazas (`tests/conftest.py`).

Los Work Items se descubren desde `mocks/`: cada archivo `wi_<id>_*.json` es el `WI-<id>`.
Para agregar uno, basta con poner su JSON en esa carpeta.

## Arquitectura del agente (Nivel 5)

El agente es un **bucle** de LangGraph en `src/agent_loop/`, organizado según su anatomía
(Sense → Reason → Plan → Act, cap. 4): ACT vuelve a SENSE hasta alcanzar el objetivo.

```
START → sense ─┬─ reason[LLM] → plan[LLM] → act ─┬─ objetivo alcanzado → END
               │                                 └─ falta ─→ sense (otra vuelta)
               └─ WI no existe ──────────────→ act → END (error controlado)
```

| Módulo | Rol | Fase FCoT | LLM |
|---|---|---|---|
| `sense.py` | Percibir: WI, fuentes faltantes/corruptas, memoria y lo observado en la vuelta anterior | RECAP | no |
| `reason.py` | Interpretar la situación frente al objetivo (`Assessment`: qué se sabe, qué falta) | REASON | sí |
| `plan.py` | Convertir la evaluación en pasos con herramientas del catálogo (`ActionPlan`) | REASON | sí |
| `act.py` | Ejecutar paso a paso; cada paso hace RECAP → REASON → VERIFY; entregar o repetir | RECAP/REASON/VERIFY | solo al redactar |
| `verify.py` | Grounding, confianza y ¿objetivo alcanzado? (reglas) | VERIFY | no |
| `tools.py` | Catálogo: `fetch_tfs_and_git_data`, `search_meet_transcript_rag`, `search_work_item_rag`, `recall_memory`, `draft_test_plan`, `draft_answer` | — | — |
| `state.py` / `memory.py` / `prompts.py` / `llm.py` / `graph.py` | Estado y objetivo / memoria / IC / modelo / grafo | — | — |

**Fractal Chain of Thought (cap. 13)**: el Instruction Contract va en cada llamada al LLM y el
mismo ciclo RECAP → REASON → VERIFY se repite en dos niveles: en cada vuelta (sense → reason/plan
→ comprobación del objetivo) y en cada paso que ejecuta ACT. El Audit Trail lo registra con el
campo `level` (`vuelta 1`, `vuelta 1 · paso 2`, …).

**Cuándo termina**: el objetivo lo comprueban reglas, no el LLM. En un plan, cuando el agente
decide escalar o el plan tiene respaldo suficiente (confianza ≥ 0.85); en una pregunta, cuando
todas las citas se verifican o el agente dice que no lo encontró. Si no, VERIFY deja feedback
(p. ej. qué casos se descartaron y por qué) y el agente da otra vuelta para corregirse. Tras
`MVA_MAX_ITERATIONS` vueltas (3 por defecto) se escala a un humano o se responde que no se pudo
verificar. En el camino feliz son 3 llamadas al LLM por vuelta (reason, plan y redacción).

**Memoria**
- *Corto plazo* (en memoria): el checkpointer `InMemorySaver` de LangGraph guarda el estado de la
  sesión; el campo `history` acumula los turnos y el LLM ve los últimos 5. Se pierde al salir.
- *Largo plazo* (Chroma, colección `mva_long_term_memory` en `chroma.db/`): al terminar cada
  ejecución se guarda un resumen; en SENSE se recuperan las 3 ejecuciones pasadas más parecidas.

La memoria es solo contexto: no es una fuente y el grounding nunca la acepta como cita.

**Trazas (LangSmith)**: cada ejecución del agente es un solo `graph.invoke`, así que queda en
**una sola traza** con los nodos, las herramientas (`@traceable`), el RAG y la llamada al LLM
anidados. La CLI (`MVA CLI`) y la evaluación (`MVA evaluación`) agrupan todas sus ejecuciones en
una traza raíz. El `trace_id` del Audit Trail va en la metadata de la traza.

## RAG híbrido (Nivel 2)

Se indexan **todos** los archivos de `mocks/`. Cada fuente del WI (TFS, Meet, Git) se parte
con `RecursiveCharacterTextSplitter` en chunks de 200 caracteres con **20 de solapamiento**.
Cada chunk lleva metadata:

| Campo | Contenido | Por defecto |
|---|---|---|
| `source_file` | archivo de origen | nombre del JSON |
| `context` | de dónde proviene | `WI — título · tipo de fuente` |
| `date` | fecha para relevancia temporal | fecha de modificación del archivo |
| `source`, `wi_id`, `corrupt`, `chunk` | fuente, WI, si está corrupta, posición | — |

`context` y `date` se pueden ingresar a mano:

```python
from tools_and_rag import build_index
build_index(metadata_overrides={"wi_4052_happy_path.json": {"context": "Export TFS sprint 12", "date": "2026-11-20"}})
```

Búsqueda (`k = 5`):
1. **Candidatos** por embeddings locales en un vector store persistente (Chroma en `chroma.db/`,
   con FastEmbed `paraphrase-multilingual-MiniLM-L12-v2`, sin API key) y por términos (BM25).
   Chroma solo recalcula los embeddings si cambian los mocks, el split o el modelo de embeddings.
2. **Reranking**: `0.5 · embeddings + 0.5 · BM25` (cada señal normalizada contra su máximo),
   multiplicado por un factor de fecha que puede restar hasta 20 % a los documentos más antiguos
   (vida media de 30 días respecto al más reciente).
3. Se devuelven los 5 mejores con sus puntajes (`score`, `score_embeddings`, `score_bm25`,
   `score_recency`). Las fuentes corruptas se excluyen y se reportan como advertencia.

El modelo de embeddings se descarga la primera vez (~220 MB) en `.cache/fastembed/`; se cambia con
`MVA_EMBEDDINGS_MODEL`. Los tests usan embeddings falsos deterministas; para probar con los reales:
`MVA_TEST_EMBEDDINGS=1 uv run pytest -v`.

## Grounding y evaluación (Nivel 4)

**Grounding** (`src/grounding.py`): el agente decide; las reglas verifican.

1. **RECAP**: se detectan las fuentes faltantes o corruptas y el agente reúne la evidencia con
   sus herramientas (TFS y Git, fragmentos de Meet vía RAG); las advertencias llegan al LLM.
2. **REASON**: al redactar, el agente decide si escala a un humano (`escalate_to_human` +
   `escalation_reason`) o propone discrepancias y casos; cada caso trae `evidence_quote`, una
   cita literal de la fuente que indica su `grounding_source`.
3. **VERIFY**: si el agente escaló, se escala sin casos. Si propuso un plan, cada cita se busca en
   el texto real de esa fuente (ignorando mayúsculas, tildes y espacios; se permiten elipsis); una
   fuente corrupta nunca sirve de respaldo. Los casos sin respaldo se descartan, quedan en el
   Audit Trail y se le devuelven al agente como feedback para la siguiente vuelta.

`confidence_score = 0.5 · fuentes válidas / 3 + 0.5 · casos con respaldo / casos propuestos`.
Con confianza ≥ 0.85 el plan queda `READY_FOR_QA`; si no, el agente da otra vuelta y, si se
agotan las vueltas, se escala sin casos aunque no lo haya pedido (barrera de seguridad). Los validadores
de `QATestPlanOutput` impiden planes incoherentes (p. ej. un escalado con casos de prueba).

En las preguntas, una respuesta solo se acepta si todas sus citas aparecen en las fuentes; si
no, el agente lo intenta de nuevo y, al agotar las vueltas, responde que *no pudo verificar esa información en las fuentes del WI y no está
seguro de que exista*, sin agregar el texto no verificado del LLM.

**Evaluación** (`src/evaluation.py`): corre el agente contra `evals/golden_set.json` (las 3
trampas de WI-4052, el escalado de WI-4099, WI-9999 y dos preguntas) y reporta:

| Métrica | Qué mide |
|---|---|
| `accuracy` | casos del conjunto dorado que pasan todas sus verificaciones |
| `check_pass_rate` | verificaciones individuales que pasan |
| `grounding_rate` | casos propuestos por el LLM con cita verificada |
| `hallucination_guard` | preguntas sin respuesta en las fuentes que el agente rechaza |

El reporte queda en `logs/eval_<fecha>.json`. Termina con código 0 solo si `accuracy` = 1.0.

Cada ejecución deja su Audit Trail (trace_id, timestamp, reasoning_trace, tools, output)
en `logs/audit.jsonl`.
