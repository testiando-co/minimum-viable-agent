"""Catálogo de herramientas (toolbelt) que PLAN puede pedir y ACT ejecuta.

Las herramientas de percepción son deterministas (no usan el LLM); las de redacción
(`draft_test_plan`, `draft_answer`) las ejecuta ACT con el LLM.
"""

import json
from typing import Optional

from agent_loop.memory import get_long_term_memory
from agent_loop.state import Mode
from schemas import PlanStep
from tools_and_rag import fetch_tfs_and_git_data, search_meet_transcript_rag, search_work_item_rag

# Nombre -> (modos en que sirve, si necesita `query`, descripción para el LLM)
CATALOG: dict[str, tuple[tuple[Mode, ...], bool, str]] = {
    "fetch_tfs_and_git_data": (
        ("plan", "question"), False,
        "Trae la especificación de TFS y el git diff del commit del WI (JSON). Sin query.",
    ),
    "search_meet_transcript_rag": (
        ("plan", "question"), True,
        "Busca en la transcripción de la demo con la PO (RAG, top 5 con puntajes). Avisa si está corrupta.",
    ),
    "search_work_item_rag": (
        ("plan", "question"), True,
        "Busca en las tres fuentes válidas del WI (TFS, Meet y git diff) con RAG.",
    ),
    "recall_memory": (
        ("plan", "question"), True,
        "Recupera ejecuciones pasadas parecidas (memoria de largo plazo). Solo contexto: no es fuente.",
    ),
    "draft_test_plan": (
        ("plan",), False,
        "Redacta el plan de pruebas (o decide escalar) con todo lo observado. Debe ser el último paso.",
    ),
    "draft_answer": (
        ("question",), False,
        "Redacta la respuesta con citas literales de lo observado. Debe ser el último paso.",
    ),
}

DRAFT_TOOL: dict[Mode, str] = {"plan": "draft_test_plan", "question": "draft_answer"}


def catalog_for(mode: Mode) -> str:
    """Catálogo en texto para el prompt de PLAN (solo las herramientas del modo)."""
    lines = []
    for name, (modes, needs_query, description) in CATALOG.items():
        if mode in modes:
            lines.append(f"- {name}{'(query)' if needs_query else ''}: {description}")
    return "\n".join(lines)


def invalid_step(mode: Mode, step: PlanStep) -> Optional[str]:
    """Motivo por el que un paso no se puede ejecutar en este modo, o None si es válido."""
    if step.tool not in CATALOG:
        return f"la herramienta {step.tool} no existe"
    modes, needs_query, _ = CATALOG[step.tool]
    if mode not in modes:
        return f"{step.tool} no sirve para el objetivo de este modo ({mode})"
    if needs_query and not (step.query or "").strip():
        return f"{step.tool} necesita una consulta (query)"
    return None


def run_tool(name: str, wi_id: str, query: Optional[str]) -> tuple[dict[str, str], str]:
    """Ejecuta una herramienta determinista y devuelve (argumentos, resultado en JSON)."""
    if name == "fetch_tfs_and_git_data":
        return {"wi_id": wi_id}, fetch_tfs_and_git_data(wi_id)
    args = {"wi_id": wi_id, "query": query or ""}
    if name == "search_meet_transcript_rag":
        return args, search_meet_transcript_rag(wi_id, args["query"])
    if name == "search_work_item_rag":
        return args, search_work_item_rag(wi_id, args["query"])
    if name == "recall_memory":
        recalled = get_long_term_memory().recall(f"{wi_id} {args['query']}")
        return args, json.dumps({"memory": recalled}, ensure_ascii=False, indent=2)
    raise ValueError(f"{name} no es una herramienta determinista")
