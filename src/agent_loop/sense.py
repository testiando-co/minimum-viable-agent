"""SENSE (FCoT: RECAP) — percibir el entorno: qué dicen las fuentes y qué recuerda el agente.

1. Herramienta `fetch_tfs_and_git_data`: TFS y git diff del WI.
2. RAG: fragmentos de Meet (plan) o de todas las fuentes válidas (pregunta).
3. Memoria: ejecuciones pasadas parecidas (largo plazo, Chroma) y turnos recientes de la
   sesión (corto plazo, estado en memoria).
"""

import json
from typing import Any

import logger
from agent_loop.memory import get_long_term_memory, recent_turns
from agent_loop.state import AgentState
from tools_and_rag import (
    fetch_tfs_and_git_data,
    load_work_item,
    search_meet_transcript_rag,
    search_work_item_rag,
)


def _memory_context(long_term: list[dict[str, Any]], short_term: list[dict[str, Any]]) -> str:
    if not long_term and not short_term:
        return "(sin memoria previa)"
    return json.dumps(
        {"ejecuciones_anteriores_parecidas": long_term, "conversacion_reciente": short_term},
        ensure_ascii=False, indent=2,
    )


def sense(state: AgentState) -> dict[str, Any]:
    trace_id, wi_id = state["trace_id"], state["wi_id"]
    tools = list(state["tools"])

    tfs_and_git = fetch_tfs_and_git_data(wi_id)
    tools.append(logger.tool(trace_id, "fetch_tfs_and_git_data", {"wi_id": wi_id}, tfs_and_git))
    wi = load_work_item(wi_id)
    if wi is None:
        return {
            "work_item": None, "tools": tools,
            "reasoning_trace": state["reasoning_trace"] + [logger.step(trace_id, "RECAP", f"{wi_id} no existe en mocks/")],
        }

    if state["mode"] == "plan":
        # La consulta al RAG sale de la propia especificación de TFS.
        query = " ".join(filter(None, (wi.tfs_specification.business_rule, wi.tfs_specification.ui_requirement)))
        rag = search_meet_transcript_rag(wi_id, query)
        tools.append(logger.tool(trace_id, "search_meet_transcript_rag", {"wi_id": wi_id, "query": query}, rag))
        prompt_vars = {"wi_id": wi_id, "tfs_and_git": tfs_and_git, "meet_fragments": rag}
    else:
        query = state["question"]
        rag = search_work_item_rag(wi_id, query)
        tools.append(logger.tool(trace_id, "search_work_item_rag", {"wi_id": wi_id, "query": query}, rag))
        prompt_vars = {"wi_id": wi_id, "question": query, "fragments": rag}

    long_term = get_long_term_memory().recall(f"{wi_id} {wi.title} {query}")
    short_term = recent_turns(state.get("history", []))
    prompt_vars["memory"] = _memory_context(long_term, short_term)

    problems = wi.missing_or_corrupt_sources()
    if state["mode"] == "plan":
        prompt_vars["source_warnings"] = "\n".join(problems) or "(ninguna)"
    detail = (f"TFS presente; {len(json.loads(rag)['fragments'])} fragmentos vía RAG; "
              f"fuentes válidas {3 - len(problems)}/3; memoria: {len(long_term)} ejecuciones "
              f"pasadas, {len(short_term)} turnos de la sesión")
    if problems:
        detail += f"; problemas: {'; '.join(problems)}"
    return {
        "work_item": wi.model_dump(), "problems": problems, "prompt_vars": prompt_vars,
        "long_term_memory": long_term, "tools": tools,
        "reasoning_trace": state["reasoning_trace"] + [logger.step(trace_id, "RECAP", detail)],
    }
