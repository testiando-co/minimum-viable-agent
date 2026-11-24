"""PLAN (FCoT: REASON) — la capa táctica: el LLM convierte la evaluación en pasos.

Con el Assessment de REASON, el catálogo de herramientas del modo y el feedback de la vuelta
anterior, el LLM produce un `ActionPlan`: pasos ordenados `{tool, query, purpose}` que ACT
ejecuta. El último paso redacta el entregable (`draft_test_plan` o `draft_answer`).
"""

import json
from typing import Any, Callable, Optional

from langchain_core.runnables import RunnableConfig

import logger
from agent_loop.llm import structured
from agent_loop.prompts import PLAN_PROMPT
from agent_loop.state import AgentState
from agent_loop.tools import DRAFT_TOOL, catalog_for
from schemas import ActionPlan


def make_plan(llm: Optional[Any] = None) -> Callable[[AgentState, RunnableConfig], dict[str, Any]]:
    """Crea el nodo PLAN con el LLM indicado (o el configurado en .env)."""

    def plan(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        mode = state["mode"]
        action_plan: ActionPlan = (PLAN_PROMPT | structured(llm, ActionPlan)).invoke({
            "goal": state["goal"], "iteration": state["iteration"], "max_iterations": state["max_iterations"],
            "assessment": json.dumps(state["assessment"], ensure_ascii=False, indent=2),
            "catalog": catalog_for(mode), "draft_tool": DRAFT_TOOL[mode],
            "feedback": "\n".join(state["feedback"]) or "(ninguno)",
        }, config)
        steps = "; ".join(
            f"{i}. {s.tool}" + (f"({s.query})" if s.query else "") for i, s in enumerate(action_plan.steps, 1)
        )
        return {
            "action_plan": action_plan.model_dump(),
            "reasoning_trace": state["reasoning_trace"] + [
                logger.step(state["trace_id"], "REASON", f"plan de acción: {steps or '(sin pasos)'}",
                            f"vuelta {state['iteration']}")],
        }

    return plan
