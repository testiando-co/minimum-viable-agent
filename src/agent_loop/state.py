"""Estado del agente en LangGraph.

Cada nodo (sense, plan, reason, verify, act) lee el estado y devuelve solo los campos que
cambia. Todo lo que se guarda es serializable (dicts y listas), porque el checkpointer en
memoria lo conserva entre ejecuciones de la misma sesión.

Los campos "por ejecución" se reinician en cada llamada (`new_run`); `history` es la
memoria de corto plazo y se acumula durante la sesión gracias a su reducer (`operator.add`).
"""

import operator
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Optional, TypedDict

Mode = Literal["plan", "question"]
Route = Literal["not_found", "llm"]


class AgentState(TypedDict, total=False):
    # --- Entrada -------------------------------------------------------------------
    mode: Mode
    wi_id: str
    question: Optional[str]
    trace_id: str
    timestamp: str
    model: str
    session_id: str

    # --- SENSE: lo que el agente percibe ------------------------------------------------
    work_item: Optional[dict[str, Any]]       # WorkItemData.model_dump() o None si no existe
    problems: list[str]                       # fuentes faltantes o corruptas
    prompt_vars: dict[str, str]               # fuentes formateadas para el prompt
    long_term_memory: list[dict[str, Any]]    # ejecuciones pasadas parecidas (Chroma)

    # --- PLAN: qué camino tomar ---------------------------------------------------------
    route: Route

    # --- REASON: lo que propone el LLM --------------------------------------------------
    draft: Optional[dict[str, Any]]           # LLMTestPlanDraft o GroundedAnswer (model_dump)

    # --- VERIFY: grounding ----------------------------------------------------------------
    grounding: Optional[dict[str, Any]]
    grounded_cases: list[dict[str, Any]]

    # --- ACT: resultado final ------------------------------------------------------------
    result: Any                               # plan (dict), respuesta (str) o error (dict)
    details: dict[str, Any]

    # --- Audit Trail ------------------------------------------------------------------------
    reasoning_trace: list[dict[str, Any]]
    tools: list[dict[str, Any]]

    # --- Memoria de corto plazo (se acumula en la sesión) --------------------------------------
    history: Annotated[list[dict[str, Any]], operator.add]


def new_run(mode: Mode, wi_id: str, question: Optional[str], model: str, session_id: str) -> AgentState:
    """Estado inicial de una ejecución: reinicia todo menos `history`."""
    return AgentState(
        mode=mode, wi_id=wi_id, question=question, model=model, session_id=session_id,
        trace_id=str(uuid.uuid4()), timestamp=datetime.now(timezone.utc).isoformat(),
        work_item=None, problems=[], prompt_vars={}, long_term_memory=[],
        route="llm", draft=None, grounding=None, grounded_cases=[],
        result=None, details={}, reasoning_trace=[], tools=[],
    )
