"""ACT — entregar el resultado y dejar registro.

1. Si el WI no existe, el resultado es el error controlado.
2. Escribe el Audit Trail (logs/audit.jsonl) con el mismo trace_id de la traza de LangSmith.
3. Guarda la ejecución en la memoria de largo plazo (Chroma) y agrega el turno a la de
   corto plazo (`history`, que el checkpointer conserva durante la sesión).
"""

from typing import Any

import logger
from agent_loop.memory import get_long_term_memory, turn_summary
from agent_loop.state import AgentState
from tools_and_rag import NOT_FOUND


def act(state: AgentState) -> dict[str, Any]:
    update: dict[str, Any] = {}
    if state["route"] == "not_found":
        result = dict(NOT_FOUND) if state["mode"] == "plan" else NOT_FOUND["error"]
        update.update(result=result, details={"found": False, "grounded": False, "citations": []})
    else:
        result = state["result"]

    logger.write_record({
        "trace_id": state["trace_id"], "timestamp": state["timestamp"], "session_id": state["session_id"],
        "wi_id": state["wi_id"], "request": state.get("question") or "plan de pruebas",
        "model": state["model"], "reasoning_trace": state["reasoning_trace"], "tools": state["tools"],
        "output": {"output": result, "grounding": state.get("grounding")},
    })

    if state["route"] != "not_found":
        turn = turn_summary({**state, "result": result})
        get_long_term_memory().save(turn)
        update["history"] = [turn]  # el reducer lo agrega a la memoria de corto plazo
    return update
