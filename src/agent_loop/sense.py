"""SENSE (FCoT: RECAP) — percibir el entorno al inicio de cada vuelta del bucle.

- Vuelta 1: ¿existe el WI?, ¿qué fuentes faltan o están corruptas?, y qué recuerda el agente
  (largo plazo en Chroma y turnos recientes de la sesión).
- Vueltas siguientes: lo que se observó al ejecutar el plan anterior (salidas de las
  herramientas) y el feedback de la verificación, para que REASON corrija el rumbo.

No llama al LLM: solo percibe y resume.
"""

import json
from typing import Any

import logger
from agent_loop.memory import get_long_term_memory, recent_turns
from agent_loop.state import AgentState
from tools_and_rag import load_work_item

OBSERVATION_PREVIEW = 1200  # caracteres de cada observación que se muestran en el recap


def _memory_context(long_term: list[dict[str, Any]], short_term: list[dict[str, Any]]) -> str:
    if not long_term and not short_term:
        return "(sin memoria previa)"
    return json.dumps(
        {"ejecuciones_anteriores_parecidas": long_term, "conversacion_reciente": short_term},
        ensure_ascii=False, indent=2,
    )


def _preview(text: str) -> str:
    return text if len(text) <= OBSERVATION_PREVIEW else text[:OBSERVATION_PREVIEW] + " …(recortado)"


def build_recap(state: AgentState, problems: list[str]) -> str:
    wi = state["work_item"] or {}
    recap = {
        "wi_id": state["wi_id"],
        "titulo": wi.get("title"),
        "pregunta": state.get("question"),
        "estado_de_las_fuentes": problems or ["TFS, MEET_PO_DEMO y GIT_DIFF presentes"],
        "observaciones": [
            {"vuelta": o["iteration"], "herramienta": o["tool"], "query": o["query"], "resultado": _preview(o["result"])}
            for o in state["observations"]
        ] or "(todavía no se ha ejecutado ninguna herramienta)",
        "feedback_de_la_verificacion": state["feedback"] or "(ninguno)",
    }
    return json.dumps(recap, ensure_ascii=False, indent=2)


def sense(state: AgentState) -> dict[str, Any]:
    trace_id, wi_id = state["trace_id"], state["wi_id"]
    iteration = state["iteration"] + 1
    level = f"vuelta {iteration}"

    if iteration == 1:
        wi = load_work_item(wi_id)
        if wi is None:
            return {
                "iteration": iteration, "route": "not_found", "work_item": None,
                "reasoning_trace": state["reasoning_trace"] + [
                    logger.step(trace_id, "RECAP", f"{wi_id} no existe en mocks/", level)],
            }
        problems = wi.missing_or_corrupt_sources()
        query = f"{wi_id} {wi.title} {state.get('question') or 'plan de pruebas'}"
        long_term = get_long_term_memory().recall(query)
        short_term = recent_turns(state.get("history", []))
        update: dict[str, Any] = {
            "route": "llm", "work_item": wi.model_dump(), "problems": problems,
            "long_term_memory": long_term, "memory_context": _memory_context(long_term, short_term),
        }
        detail = (f"objetivo: {state['goal']} | fuentes válidas {3 - len(problems)}/3"
                  + (f" (problemas: {'; '.join(problems)})" if problems else "")
                  + f"; memoria: {len(long_term)} ejecuciones pasadas, {len(short_term)} turnos de la sesión")
    else:
        problems = state["problems"]
        update = {}
        detail = (f"{len(state['observations'])} observaciones acumuladas; feedback de la vuelta "
                  f"anterior: {'; '.join(state['feedback']) or 'ninguno'}")

    merged = {**state, **update}
    return {
        **update, "iteration": iteration, "recap": build_recap(merged, problems),
        "reasoning_trace": state["reasoning_trace"] + [logger.step(trace_id, "RECAP", detail, level)],
    }


def route_after_sense(state: AgentState) -> str:
    """Si el WI no existe no hay nada que razonar: ACT entrega el error controlado."""
    return "act" if state["route"] == "not_found" else "reason"
