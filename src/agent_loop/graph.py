"""El bucle del agente como grafo de LangGraph (anatomía Sense → Reason → Plan → Act ↺).

    START → sense ─┬─ reason[LLM] → plan[LLM] → act ─┬─ objetivo alcanzado → END
                   │                                 └─ falta ─→ sense (otra vuelta)
                   └─ WI no existe ──────────────→ act → END (error controlado)

Cada vuelta aplica el Fractal Chain of Thought: RECAP (sense) → REASON (reason, plan) →
VERIFY (act, por paso y al final de la vuelta). ACT vuelve a SENSE con el feedback de la
verificación hasta alcanzar el objetivo o agotar `MVA_MAX_ITERATIONS` vueltas.

Una ejecución = un `graph.invoke` = una sola traza en LangSmith, con cada vuelta, nodo,
herramienta, recuperación del RAG y llamada al LLM anidados dentro. La memoria de corto plazo
vive en el checkpointer en memoria (un thread por sesión).
"""

from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from agent_loop.act import make_act, route_after_act
from agent_loop.llm import model_name
from agent_loop.memory import SESSION_ID, SHORT_TERM_CHECKPOINTER
from agent_loop.plan import make_plan
from agent_loop.reason import make_reason
from agent_loop.sense import route_after_sense, sense
from agent_loop.state import AgentState, Mode, new_run

NODES_PER_ITERATION = 4  # sense, reason, plan, act


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
    graph.add_conditional_edges("act", route_after_act, {"sense": "sense", "end": END})
    return graph.compile(checkpointer=checkpointer)


def run_agent(
    mode: Mode,
    wi_id: str,
    question: Optional[str] = None,
    llm: Optional[Any] = None,
    session_id: Optional[str] = None,
) -> AgentState:
    """Ejecuta el agente hasta alcanzar el objetivo y devuelve el estado final."""
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
    return build_graph(llm).invoke(state, config)
