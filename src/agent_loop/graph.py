"""El bucle del agente como grafo de LangGraph (anatomía Sense → Plan → Reason → Verify → Act).

    START → sense → plan ─┬─ llm ───────→ reason → verify → act → END
                          └─ not_found ─────────────────────→ act

Una ejecución = un `graph.invoke` = una sola traza en LangSmith, con cada nodo, herramienta,
recuperación del RAG y llamada al LLM anidados dentro. La memoria de corto plazo vive en el
checkpointer en memoria (un thread por sesión).
"""

from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from agent_loop.act import act
from agent_loop.llm import model_name
from agent_loop.memory import SESSION_ID, SHORT_TERM_CHECKPOINTER
from agent_loop.plan import plan, route_after_plan
from agent_loop.reason import make_reason
from agent_loop.sense import sense
from agent_loop.state import AgentState, Mode, new_run
from agent_loop.verify import verify


def build_graph(llm: Optional[Any] = None, checkpointer: Any = SHORT_TERM_CHECKPOINTER):
    graph = StateGraph(AgentState)
    graph.add_node("sense", sense)
    graph.add_node("plan", plan)
    graph.add_node("reason", make_reason(llm))
    graph.add_node("verify", verify)
    graph.add_node("act", act)
    graph.add_edge(START, "sense")
    graph.add_edge("sense", "plan")
    graph.add_conditional_edges("plan", route_after_plan, ["reason", "act"])
    graph.add_edge("reason", "verify")
    graph.add_edge("verify", "act")
    graph.add_edge("act", END)
    return graph.compile(checkpointer=checkpointer)


def run_agent(
    mode: Mode,
    wi_id: str,
    question: Optional[str] = None,
    llm: Optional[Any] = None,
    session_id: Optional[str] = None,
) -> AgentState:
    """Ejecuta el agente y devuelve el estado final."""
    session_id = session_id or SESSION_ID
    state = new_run(mode, wi_id, question, model_name(), session_id)
    config = {
        "configurable": {"thread_id": session_id},
        "run_name": f"MVA {'plan de pruebas' if mode == 'plan' else 'pregunta'} {wi_id}",
        "tags": ["mva", mode],
        "metadata": {"wi_id": wi_id, "mode": mode, "model": state["model"],
                     "trace_id": state["trace_id"], "session_id": session_id},
    }
    return build_graph(llm).invoke(state, config)
