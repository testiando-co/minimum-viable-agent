"""ACT — ejecutar el plan paso a paso, verificar y decidir si el bucle termina.

ACT ejecuta UN paso por invocación y vuelve a sí mismo mientras queden pasos: así el
checkpointer guarda el avance después de cada paso (`step_index`, `completed_steps`) y, si un
paso falla, el grafo se retoma desde ahí sin repetir los anteriores.

Cada paso repite el mismo ciclo del FCoT a menor escala (lo "fractal"):
- RECAP: qué paso es y para qué sirve.
- REASON: ejecutarlo. Las herramientas de percepción son deterministas; los pasos de
  redacción (`draft_test_plan`, `draft_answer`) llaman al LLM.
- VERIFY: revisar el resultado contra el Instruction Contract.

Al final de la vuelta, VERIFY decide con reglas si se alcanzó el objetivo (`check_goal`):
- Sí (o se agotaron las vueltas, o el WI no existe): entrega el resultado, escribe el Audit
  Trail y guarda la memoria; el grafo termina.
- No: deja el feedback y el grafo vuelve a SENSE.
"""

from typing import Any, Callable, Optional

from langchain_core.runnables import RunnableConfig

import logger
from agent_loop.llm import structured
from agent_loop.memory import get_long_term_memory, turn_summary
from agent_loop.prompts import QUESTION_PROMPT, TEST_PLAN_PROMPT
from agent_loop.state import AgentState
from agent_loop.tools import invalid_step, run_tool
from agent_loop.verify import check_goal, verify_draft, verify_tool_result
from schemas import ActionPlan, GroundedAnswer, LLMTestPlanDraft, WorkItemData
from tools_and_rag import NOT_FOUND


def _observations_text(observations: list[dict[str, Any]]) -> str:
    if not observations:
        return "(no se ejecutó ninguna herramienta)"
    return "\n\n".join(
        f"### {o['tool']}" + (f" (query: {o['query']})" if o["query"] else "") + f"\n{o['result']}"
        for o in observations
    )


def _draft(llm: Optional[Any], state: AgentState, observations: list[dict[str, Any]], config: RunnableConfig):
    common = {
        "wi_id": state["wi_id"], "observations": _observations_text(observations),
        "feedback": "\n".join(state["feedback"]) or "(ninguno)", "memory": state["memory_context"],
    }
    if state["mode"] == "plan":
        draft: LLMTestPlanDraft = (TEST_PLAN_PROMPT | structured(llm, LLMTestPlanDraft)).invoke(
            {**common, "source_warnings": "\n".join(state["problems"]) or "(ninguna)"}, config)
        detail = (f"decisión del agente: {'ESCALAR' if draft.escalate_to_human else 'plan de pruebas'}"
                  + (f" ({draft.escalation_reason})" if draft.escalate_to_human else "")
                  + f"; {len(draft.detected_discrepancies)} discrepancias: {draft.detected_discrepancies}; "
                  f"{len(draft.test_cases)} casos propuestos")
    else:
        draft = (QUESTION_PROMPT | structured(llm, GroundedAnswer)).invoke(
            {**common, "question": state["question"]}, config)
        detail = f"respuesta redactada: found={draft.found}; {len(draft.citations)} citas"
    return draft.model_dump(), detail


def _finish(state: AgentState, update: dict[str, Any]) -> dict[str, Any]:
    """Entrega: Audit Trail y memoria (largo plazo en Chroma, corto plazo en `history`)."""
    final = {**state, **update}
    logger.write_record({
        "trace_id": final["trace_id"], "timestamp": final["timestamp"], "session_id": final["session_id"],
        "wi_id": final["wi_id"], "request": final.get("question") or "plan de pruebas",
        "model": final["model"], "iterations": final["iteration"], "end_reason": final["end_reason"],
        "reasoning_trace": final["reasoning_trace"], "tools": final["tools"],
        "output": {"output": final["result"], "grounding": final.get("grounding")},
    })
    if final["route"] != "not_found":
        turn = turn_summary(final)
        get_long_term_memory().save(turn)
        update["history"] = [turn]  # el reducer lo agrega a la memoria de corto plazo
    return update


def _run_step(llm: Optional[Any], state: AgentState, wi: WorkItemData, config: RunnableConfig) -> dict[str, Any]:
    """Ejecuta el paso `step_index` con su RECAP → REASON → VERIFY y devuelve los cambios."""
    trace_id, mode, iteration = state["trace_id"], state["mode"], state["iteration"]
    index = state["step_index"]
    step = ActionPlan.model_validate(state["action_plan"]).steps[index]
    level = f"vuelta {iteration} · paso {index + 1}"
    trail, tools = list(state["reasoning_trace"]), list(state["tools"])
    observations, draft = list(state["observations"]), state["draft"]

    trail.append(logger.step(trace_id, "RECAP",
                             f"{step.tool}" + (f"({step.query})" if step.query else "") + f": {step.purpose}", level))
    problem = invalid_step(mode, step)
    if problem:
        status, check = "rechazado", f"paso rechazado: {problem}"
        trail.append(logger.step(trace_id, "VERIFY", check, level))
    elif step.tool in ("draft_test_plan", "draft_answer"):
        draft, detail = _draft(llm, state, observations, config)
        status, check = "ok", verify_draft(mode, wi, draft)
        trail += [logger.step(trace_id, "REASON", detail, level), logger.step(trace_id, "VERIFY", check, level)]
    else:
        args, result = run_tool(step.tool, state["wi_id"], step.query)
        tools.append(logger.tool(trace_id, step.tool, args, result))
        observations.append({"iteration": iteration, "tool": step.tool, "query": step.query, "result": result})
        status, check = "ok", verify_tool_result(step.tool, result)
        trail += [logger.step(trace_id, "REASON", f"se ejecutó {step.tool}", level),
                  logger.step(trace_id, "VERIFY", check, level)]

    done = {"step": index + 1, "tool": step.tool, "query": step.query, "status": status, "verify": check}
    return {
        "step_index": index + 1, "completed_steps": state["completed_steps"] + [done],
        "observations": observations, "draft": draft, "tools": tools, "reasoning_trace": trail,
    }


def make_act(llm: Optional[Any] = None) -> Callable[[AgentState, RunnableConfig], dict[str, Any]]:
    """Crea el nodo ACT: ejecuta UN paso del plan por invocación.

    Al terminar el nodo, el checkpointer guarda qué pasos se hicieron (`completed_steps`) y cuál
    sigue (`step_index`). Si un paso falla (p. ej. un 429 del LLM), el grafo se retoma desde ese
    paso sin repetir los anteriores (ver `graph.run_agent`). En el último paso, ACT comprueba el
    objetivo y entrega el resultado o deja el feedback para la siguiente vuelta.
    """

    def act(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        if state["route"] == "not_found":
            result = dict(NOT_FOUND) if state["mode"] == "plan" else NOT_FOUND["error"]
            return _finish(state, {
                "result": result, "goal_reached": True, "end_reason": "el WI no existe",
                "details": {"found": False, "grounded": False, "citations": []},
            })

        wi = WorkItemData.model_validate(state["work_item"])
        total = len(ActionPlan.model_validate(state["action_plan"]).steps)
        update: dict[str, Any] = _run_step(llm, state, wi, config) if state["step_index"] < total else {}
        current = {**state, **update}
        if current["step_index"] < total:
            return {**update, "goal_reached": False}  # quedan pasos: ACT vuelve a ACT

        # Último paso: VERIFY a nivel de vuelta (¿se alcanzó el objetivo?).
        trace_id = state["trace_id"]
        outcome = check_goal(current, current["draft"])
        level = f"vuelta {state['iteration']}"
        trail = current["reasoning_trace"] + [logger.step(trace_id, "VERIFY", r, level) for r in outcome["rejected"]]
        trail.append(logger.step(trace_id, "VERIFY", outcome["detail"], level))
        update.update(reasoning_trace=trail, goal_reached=outcome["goal_reached"], feedback=outcome["feedback"])
        if not outcome["goal_reached"]:
            return update
        for key in ("end_reason", "result", "grounding", "grounded_cases", "details"):
            if key in outcome:
                update[key] = outcome[key]
        return _finish(state, update)

    return act


def route_after_act(state: AgentState) -> str:
    """Quedan pasos → act; objetivo alcanzado (o WI inexistente, o vueltas agotadas) → fin;
    si no, otra vuelta → sense."""
    if state["goal_reached"]:
        return "end"
    if state["route"] != "not_found" and state["step_index"] < len(state["action_plan"]["steps"]):
        return "act"
    return "sense"
