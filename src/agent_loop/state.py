"""Estado del agente en LangGraph.

Cada nodo (sense, reason, plan, act) lee el estado y devuelve solo los campos que cambia.
Todo lo que se guarda es serializable (dicts y listas), porque el checkpointer en memoria lo
conserva entre ejecuciones de la misma sesión.

Los campos "por ejecución" se reinician en cada llamada (`new_run`); `history` es la
memoria de corto plazo y se acumula durante la sesión gracias a su reducer (`operator.add`).
"""

import operator
import os
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Optional, TypedDict

Mode = Literal["plan", "question"]
Route = Literal["not_found", "llm"]

DEFAULT_MAX_ITERATIONS = 3
MAX_STEPS_PER_PLAN = 8  # tope de pasos por plan (cada paso es un nodo del grafo)


def max_iterations() -> int:
    """Vueltas máximas del bucle (MVA_MAX_ITERATIONS, por defecto 3)."""
    try:
        return max(1, int(os.getenv("MVA_MAX_ITERATIONS", DEFAULT_MAX_ITERATIONS)))
    except ValueError:
        return DEFAULT_MAX_ITERATIONS


def goal_for(mode: Mode, wi_id: str, question: Optional[str]) -> str:
    if mode == "plan":
        return (f"Entregar un plan de pruebas del {wi_id} con respaldo verificable en las fuentes "
                "(READY_FOR_QA), o escalar a un humano explicando el motivo si los datos no alcanzan.")
    return (f"Responder la pregunta «{question}» sobre el {wi_id} solo con citas verificables de "
            "las fuentes, o decir que no se pudo verificar.")


class AgentState(TypedDict, total=False):
    # --- Entrada y objetivo ------------------------------------------------------------
    mode: Mode
    wi_id: str
    question: Optional[str]
    goal: str
    trace_id: str
    timestamp: str
    model: str
    session_id: str

    # --- Bucle --------------------------------------------------------------------------
    iteration: int                            # vuelta actual (empieza en 1 en SENSE)
    max_iterations: int
    goal_reached: bool
    end_reason: Optional[str]                 # por qué terminó el bucle

    # --- SENSE: lo que el agente percibe ------------------------------------------------
    route: Route                              # not_found si el WI no existe
    work_item: Optional[dict[str, Any]]       # WorkItemData.model_dump() o None si no existe
    problems: list[str]                       # fuentes faltantes o corruptas
    long_term_memory: list[dict[str, Any]]    # ejecuciones pasadas parecidas (Chroma)
    memory_context: str                       # memoria formateada para los prompts
    recap: str                                # resumen de la situación para REASON

    # --- REASON y PLAN (LLM) ------------------------------------------------------------
    assessment: Optional[dict[str, Any]]      # Assessment
    action_plan: Optional[dict[str, Any]]     # ActionPlan

    # --- Avance dentro de la vuelta (el checkpointer lo guarda después de cada paso) ---------
    step_index: int                           # siguiente paso del plan a ejecutar
    completed_steps: list[dict[str, Any]]     # {step, tool, query, status, verify} ya hechos

    # --- ACT: lo que se ejecutó y observó -----------------------------------------------
    observations: list[dict[str, Any]]        # {iteration, tool, query, result} de esta ejecución
    draft: Optional[dict[str, Any]]           # último LLMTestPlanDraft o GroundedAnswer
    feedback: list[str]                       # por qué no se alcanzó el objetivo (para la siguiente vuelta)

    # --- VERIFY: grounding ----------------------------------------------------------------
    grounding: Optional[dict[str, Any]]
    grounded_cases: list[dict[str, Any]]

    # --- Resultado final ------------------------------------------------------------------
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
        mode=mode, wi_id=wi_id, question=question, goal=goal_for(mode, wi_id, question),
        model=model, session_id=session_id,
        trace_id=str(uuid.uuid4()), timestamp=datetime.now(timezone.utc).isoformat(),
        iteration=0, max_iterations=max_iterations(), goal_reached=False, end_reason=None,
        route="llm", work_item=None, problems=[], long_term_memory=[], memory_context="", recap="",
        assessment=None, action_plan=None, step_index=0, completed_steps=[], observations=[], draft=None, feedback=[],
        grounding=None, grounded_cases=[], result=None, details={}, reasoning_trace=[], tools=[],
    )
