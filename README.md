# Minimum Viable Agent (MVA) — QA

Agente que analiza un Work Item a partir de TFS, la demo de Meet con la PO y el git diff,
y genera un plan de pruebas validado con Pydantic (`QATestPlanOutput`).

## Nivel de madurez (GenAI Maturity Model)

| Nivel | Qué exige | Dónde está |
|---|---|---|
| 0 — Data foundation | Datos curados y accesibles | `mocks/*.json`, `src/tools_and_rag.py` |
| 1 — Modelo + prompt + servicio | Modelo elegido, prompt diseñado, servido vía API | `src/main_mva.py` (`init_chat_model`, Instruction Contract, `with_structured_output`) |
| 2 — RAG | Recuperar contexto relevante | `src/tools_and_rag.py` (`search_meet_transcript_rag`, `search_work_item_rag`) |
| 3 — Tuning | Fine-tuning / PEFT | Fuera de alcance (no aplica a este MVA) |
| 4 — Grounding y evaluación | Respuestas respaldadas por las fuentes y métricas | `src/grounding.py`, `src/evaluation.py`, `evals/golden_set.json` |

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

Los Work Items se descubren desde `mocks/`: cada archivo `wi_<id>_*.json` es el `WI-<id>`.
Para agregar uno, basta con poner su JSON en esa carpeta.

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
1. **Candidatos** por embeddings locales (FAISS + FastEmbed `paraphrase-multilingual-MiniLM-L12-v2`,
   sin API key) y por términos (BM25).
2. **Reranking**: `0.5 · embeddings + 0.5 · BM25` (cada señal normalizada contra su máximo),
   multiplicado por un factor de fecha que puede restar hasta 20 % a los documentos más antiguos
   (vida media de 30 días respecto al más reciente).
3. Se devuelven los 5 mejores con sus puntajes (`score`, `score_embeddings`, `score_bm25`,
   `score_recency`). Las fuentes corruptas se excluyen y se reportan como advertencia.

El modelo de embeddings se descarga la primera vez (~220 MB) en `.cache/fastembed/`; se cambia con
`MVA_EMBEDDINGS_MODEL`. Los tests usan embeddings falsos deterministas; para probar con los reales:
`MVA_TEST_EMBEDDINGS=1 uv run pytest -v`.

## Grounding y evaluación (Nivel 4)

**Grounding** (`src/grounding.py`): el LLM ya no decide el estado ni la confianza.

1. **RECAP**: se cargan TFS y Git y se recuperan los fragmentos de Meet vía RAG. Si falta una
   fuente o está corrupta, se escala de inmediato **sin llamar al LLM**.
2. **REASON**: el LLM propone discrepancias y casos; cada caso trae `evidence_quote`, una cita
   literal de la fuente que indica su `grounding_source`.
3. **VERIFY**: cada cita se busca en el texto real de esa fuente (ignorando mayúsculas, tildes y
   espacios; se permiten elipsis). Los casos sin respaldo se descartan y quedan en el Audit Trail.

`confidence_score = 0.5 · fuentes válidas / 3 + 0.5 · casos con respaldo / casos propuestos`.
Con confianza ≥ 0.85 el plan queda `READY_FOR_QA`; si no, se escala sin casos. Los validadores
de `QATestPlanOutput` impiden planes incoherentes (p. ej. un escalado con casos de prueba).

En las preguntas, una respuesta solo se acepta si todas sus citas aparecen en las fuentes; si
no, el agente responde que *no pudo verificar esa información en las fuentes del WI y no está
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
