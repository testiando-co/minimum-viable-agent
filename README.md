# Minimum Viable Agent (MVA) — QA

Agente que analiza un Work Item a partir de TFS, la demo de Meet con la PO y el git diff,
y genera un plan de pruebas validado con Pydantic (`QATestPlanOutput`).

## Nivel de madurez (GenAI Maturity Model)

| Nivel | Qué exige | Dónde está |
|---|---|---|
| 0 — Data foundation | Datos curados y accesibles | `mocks/*.json`, `src/tools_and_rag.py` |
| 1 — Modelo + prompt + servicio | Modelo elegido, prompt diseñado, servido vía API | `src/main_mva.py` (`init_chat_model`, Instruction Contract, `with_structured_output`) |
| 2 — RAG | Recuperar contexto relevante | `src/tools_and_rag.py` (`search_meet_transcript_rag`, `search_work_item_rag`) |

## Instalación

```bash
uv sync
cp .env.example .env   # completa ANTHROPIC_API_KEY
```

`MVA_MODEL` elige el modelo en formato LangChain `proveedor:modelo`
(por defecto `anthropic:claude-opus-5-5`). Para otro proveedor, instala su paquete
(`uv add langchain-openai`, etc.) y cambia la variable.

## Uso

```bash
uv run python src/main_mva.py                     # plan de pruebas de todos los WI en mocks/
uv run python src/main_mva.py WI-4052 WI-9999     # solo los WI indicados
uv run python src/main_mva.py WI-4052 -q "¿Cuántos segundos dura la animación del spinner al pagar con Visa?"
uv run pytest -v                                  # los tests que llaman al LLM se saltan sin ANTHROPIC_API_KEY
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

Cada ejecución deja su Audit Trail (trace_id, timestamp, reasoning_trace, tools, output)
en `logs/audit.jsonl`.
