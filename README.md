# Minimum Viable Agent (MVA) — QA

Agente que analiza un Work Item a partir de TFS, la demo de Meet con la PO y el git diff,
y genera un plan de pruebas validado con Pydantic (`QATestPlanOutput`).

## Nivel de madurez (GenAI Maturity Model)

| Nivel | Qué exige | Dónde está |
|---|---|---|
| 0 — Data foundation | Datos curados y accesibles | `mocks/*.json`, `src/tools_and_rag.py` |
| 1 — Modelo + prompt + servicio | Modelo elegido, prompt diseñado, servido vía API | `src/main_mva.py` (`init_chat_model`, Instruction Contract, `with_structured_output`) |
| 2 — RAG | Recuperar contexto relevante | Pendiente: `search_meet_transcript_rag` |

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

Cada ejecución deja su Audit Trail (trace_id, timestamp, reasoning_trace, tools, output)
en `logs/audit.jsonl`.
