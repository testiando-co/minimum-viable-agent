"""Memoria del agente.

- Corto plazo (en memoria): el checkpointer `InMemorySaver` de LangGraph guarda el estado
  de cada sesión (thread). El campo `history` del estado acumula los turnos de la sesión, así
  el agente recuerda lo que hizo antes en la misma ejecución del programa. Se pierde al salir.
- Largo plazo (Chroma): al terminar cada ejecución se guarda un resumen en la colección
  `mva_long_term_memory` de chroma.db/; en SENSE se recuperan las ejecuciones pasadas más
  parecidas. Es solo contexto: nunca cuenta como fuente para el grounding.

En LangSmith cada operación de memoria es un run dentro de la traza del agente:
`memoria corto plazo` (turnos de la sesión), `memoria largo plazo: recuperar` (retriever) y
`memoria largo plazo: guardar`.
"""

import os
import uuid
from functools import lru_cache
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import traceable

from tools_and_rag import chroma_client, chroma_dir, chroma_where, get_embeddings

LONG_TERM_COLLECTION = "mva_long_term_memory"
SHORT_TERM_TURNS = 5      # turnos de la sesión que se le muestran al LLM
LONG_TERM_K = 3           # ejecuciones pasadas que se recuperan

# --- Corto plazo ------------------------------------------------------------------------
SHORT_TERM_CHECKPOINTER = InMemorySaver()
SESSION_ID = str(uuid.uuid4())  # una sesión por proceso (p. ej. una ejecución del CLI)


def _without_self(inputs: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in inputs.items() if k != "self"}


@traceable(name="memoria corto plazo", run_type="chain", tags=["memoria"])
def recent_turns(history: list[dict[str, Any]], n: int = SHORT_TERM_TURNS) -> list[dict[str, Any]]:
    return history[-n:]


def turn_summary(state: dict[str, Any]) -> dict[str, Any]:
    """Resumen de la ejecución para la memoria de corto plazo (y el texto del largo plazo)."""
    result = state.get("result")
    if state["mode"] == "plan" and isinstance(result, dict) and "status" in result:
        outcome = (f"{result['status']} (confianza {result['confidence_score']}); "
                   f"discrepancias: {'; '.join(result['detected_discrepancies']) or 'ninguna'}; "
                   f"casos: {', '.join(c['test_id'] for c in result['test_cases']) or 'ninguno'}")
        if result.get("human_escalation_reason"):
            outcome += f"; motivo: {result['human_escalation_reason']}"
    else:
        outcome = str(result)[:400]
    request = state.get("question") or "plan de pruebas"
    return {"wi_id": state["wi_id"], "mode": state["mode"], "request": request, "outcome": outcome,
            "trace_id": state["trace_id"], "timestamp": state["timestamp"]}


# --- Largo plazo --------------------------------------------------------------------------
class LongTermMemory:
    """Ejecuciones pasadas en Chroma, recuperables por similitud."""

    def __init__(self, embeddings: Embeddings, persist_directory: str):
        from langchain_chroma import Chroma

        self.store = Chroma(
            collection_name=LONG_TERM_COLLECTION, embedding_function=embeddings,
            client=chroma_client(persist_directory),
        )

    @traceable(name="memoria largo plazo: guardar", run_type="tool", tags=["memoria"],
               process_inputs=_without_self)
    def save(self, turn: dict[str, Any]) -> str:
        text = (f"{turn['mode']} sobre {turn['wi_id']}: {turn['request']}. "
                f"Resultado: {turn['outcome']}")
        metadata = {key: str(turn[key]) for key in ("wi_id", "mode", "request", "trace_id", "timestamp")}
        self.store.add_documents([Document(page_content=text, metadata=metadata)], ids=[turn["trace_id"]])
        return turn["trace_id"]

    @traceable(name="memoria largo plazo: recuperar", run_type="retriever", tags=["memoria"],
               process_inputs=_without_self,
               process_outputs=lambda recalled: {"documents": [
                   {"page_content": m["memory"], "type": "Document",
                    "metadata": {k: v for k, v in m.items() if k != "memory"}}
                   for m in recalled]})
    def recall(self, query: str, k: int = LONG_TERM_K, where: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        if self.store._collection.count() == 0:
            return []
        hits = self.store.similarity_search_with_score(query, k=k, filter=chroma_where(where or {}))
        return [
            {"memory": doc.page_content, **doc.metadata, "similarity": round(1 / (1 + float(dist)), 4)}
            for doc, dist in hits
        ]

    def count(self) -> int:
        return self.store._collection.count()


@lru_cache(maxsize=4)
def _long_term_memory(persist_directory: str) -> LongTermMemory:
    return LongTermMemory(get_embeddings(), persist_directory)


def get_long_term_memory() -> LongTermMemory:
    """Memoria de largo plazo de la carpeta configurada (MVA_CHROMA_DIR o chroma.db/)."""
    return _long_term_memory(os.fspath(chroma_dir()))
