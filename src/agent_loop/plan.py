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
from agent_loop.state import MAX_STEPS_PER_PLAN, AgentState
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
        level = f"vuelta {state['iteration']}"
        trail = list(state["reasoning_trace"])
        if len(action_plan.steps) > MAX_STEPS_PER_PLAN:
            trail.append(logger.step(state["trace_id"], "VERIFY",
                                     f"el plan trae {len(action_plan.steps)} pasos; se recorta a {MAX_STEPS_PER_PLAN}", level))
            action_plan.steps = action_plan.steps[:MAX_STEPS_PER_PLAN]
        steps = "; ".join(
            f"{i}. {s.tool}" + (f"({s.query})" if s.query else "") for i, s in enumerate(action_plan.steps, 1)
        )
        trail.append(logger.step(state["trace_id"], "REASON", f"plan de acción: {steps or '(sin pasos)'}", level))
        # Vuelta nueva: el avance de ACT empieza en el paso 1 y sin borrador.
        return {"action_plan": action_plan.model_dump(), "step_index": 0, "completed_steps": [],
                "draft": None, "reasoning_trace": trail}

    return plan
