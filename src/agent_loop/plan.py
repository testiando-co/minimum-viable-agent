"""PLAN (FCoT: REASON) — decidir qué camino toma el agente.

- `not_found`: el WI no existe -> error controlado (no hay nada que analizar).
- `llm`: el agente razona con el LLM sobre las fuentes (incluida la decisión de escalar a un
  humano) y luego VERIFY comprueba el respaldo.
"""

from typing import Any

import logger
from agent_loop.state import AgentState


def plan(state: AgentState) -> dict[str, Any]:
    if state["work_item"] is None:
        return {"route": "not_found"}

    if state["mode"] == "plan":
        detail = "plan: pedir al LLM que analice las fuentes y decida plan de pruebas o escalado"
        if state["problems"]:
            detail += f" (fuentes con problemas: {len(state['problems'])})"
    else:
        detail = "plan: pedir al LLM una respuesta con citas bajo el Instruction Contract"
    return {
        "route": "llm",
        "reasoning_trace": state["reasoning_trace"] + [logger.step(state["trace_id"], "REASON", detail)],
    }


def route_after_plan(state: AgentState) -> str:
    return {"not_found": "act", "llm": "reason"}[state["route"]]
