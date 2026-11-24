"""El bucle del agente como grafo de LangGraph (anatomía Sense → Reason → Plan → Act ↺).

    START → sense ─┬─ reason[LLM] → plan[LLM] → act ─┬─ quedan pasos ──────→ act (un paso por nodo)
                   │                                 ├─ objetivo alcanzado → END
                   │                                 └─ falta ─────────────→ sense (otra vuelta)
                   └─ WI no existe ──────────────→ act → END (error controlado)

Cada vuelta aplica el Fractal Chain of Thought: RECAP (sense) → REASON (reason, plan) →
VERIFY (act, por paso y al final de la vuelta). ACT vuelve a SENSE con el feedback de la
verificación hasta alcanzar el objetivo o agotar `MVA_MAX_ITERATIONS` vueltas.

Checkpoint por paso: ACT ejecuta un paso del plan por invocación, así que el checkpointer guarda
el avance (`step_index`, `completed_steps`) después de cada paso. Si el LLM falla (429/503 del
plan gratuito, conexión), `run_agent` espera `MVA_PLAN_RETRY_DELAY` segundos y retoma con
`graph.invoke(None, config)` desde el paso pendiente, sin repetir los que ya se hicieron.

Una ejecución = un `graph.invoke` = una sola traza en LangSmith, con cada vuelta, nodo,
herramienta, recuperación del RAG y llamada al LLM anidados dentro. La memoria de corto plazo
vive en el checkpointer en memoria (un thread por sesión).
"""

import logging
import os
import time
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from agent_loop.act import make_act, route_after_act
from agent_loop.llm import llm_errors, model_name
from agent_loop.memory import SESSION_ID, SHORT_TERM_CHECKPOINTER
from agent_loop.plan import make_plan
from agent_loop.reason import make_reason
from agent_loop.sense import route_after_sense, sense
from agent_loop.state import MAX_STEPS_PER_PLAN, AgentState, Mode, new_run

NODES_PER_ITERATION = 3 + MAX_STEPS_PER_PLAN  # sense, reason, plan y un act por paso
DEFAULT_RETRY_DELAY = 10  # segundos
MAX_RETRIES = 2

_log = logging.getLogger("mva")


def retry_delay() -> float:
    """Espera antes de retomar tras un error del LLM (MVA_PLAN_RETRY_DELAY, por defecto 10 s)."""
    try:
        return max(0.0, float(os.getenv("MVA_PLAN_RETRY_DELAY", DEFAULT_RETRY_DELAY)))
    except ValueError:
        return DEFAULT_RETRY_DELAY


def build_graph(llm: Optional[Any] = None, checkpointer: Any = SHORT_TERM_CHECKPOINTER):
    graph = StateGraph(AgentState)
    graph.add_node("sense", sense)
    graph.add_node("reason", make_reason(llm))
    graph.add_node("plan", make_plan(llm))
    graph.add_node("act", make_act(llm))
    graph.add_edge(START, "sense")
    graph.add_conditional_edges("sense", route_after_sense, ["reason", "act"])
    graph.add_edge("reason", "plan")
    graph.add_edge("plan", "act")
    graph.add_conditional_edges("act", route_after_act, {"act": "act", "sense": "sense", "end": END})
    return graph.compile(checkpointer=checkpointer)


def run_agent(
    mode: Mode,
    wi_id: str,
    question: Optional[str] = None,
    llm: Optional[Any] = None,
    session_id: Optional[str] = None,
) -> AgentState:
    """Ejecuta el agente hasta alcanzar el objetivo y devuelve el estado final.

    Si el LLM falla, reintenta hasta MAX_RETRIES veces retomando desde el último checkpoint
    (el paso pendiente); si se agotan los reintentos, propaga el error.
    """
    session_id = session_id or SESSION_ID
    state = new_run(mode, wi_id, question, model_name(), session_id)
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": NODES_PER_ITERATION * state["max_iterations"] + 5,
        "run_name": f"MVA {'plan de pruebas' if mode == 'plan' else 'pregunta'} {wi_id}",
        "tags": ["mva", mode],
        "metadata": {"wi_id": wi_id, "mode": mode, "model": state["model"],
                     "trace_id": state["trace_id"], "session_id": session_id},
    }
    graph = build_graph(llm)
    pending: Optional[AgentState] = state
    for attempt in range(MAX_RETRIES + 1):
        try:
            return graph.invoke(pending, config)
        except llm_errors() as error:
            if attempt == MAX_RETRIES:
                raise
            snapshot = graph.get_state(config).values
            delay = retry_delay() * (attempt + 1)
            _log.warning("[%s] error del LLM (%s); reintento %d/%d en %.0f s desde la vuelta %s, paso %s",
                         state["trace_id"], type(error).__name__, attempt + 1, MAX_RETRIES, delay,
                         snapshot.get("iteration"), snapshot.get("step_index", 0) + 1)
            time.sleep(delay)
            pending = None  # None = continuar desde el último checkpoint del thread
    raise AssertionError("inalcanzable")
