"""REASON (FCoT: REASON) — el núcleo cognitivo: el LLM interpreta la situación.

Con el objetivo, el Instruction Contract, el recap de SENSE y la memoria, el LLM produce un
`Assessment`: qué se sabe, qué falta, qué problemas hay en las fuentes y qué debe lograr la
vuelta. No decide pasos (eso es PLAN) ni la confianza (eso lo calculan reglas en VERIFY).
"""

from typing import Any, Callable, Optional

from langchain_core.runnables import RunnableConfig

import logger
from agent_loop.llm import structured
from agent_loop.prompts import REASON_PROMPT
from agent_loop.state import AgentState
from schemas import Assessment


def make_reason(llm: Optional[Any] = None) -> Callable[[AgentState, RunnableConfig], dict[str, Any]]:
    """Crea el nodo REASON con el LLM indicado (o el configurado en .env)."""

    def reason(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        # `config` propaga el contexto del grafo: la llamada al LLM queda dentro de la misma
        # traza de LangSmith en vez de abrir una traza nueva.
        assessment: Assessment = (REASON_PROMPT | structured(llm, Assessment)).invoke({
            "goal": state["goal"], "iteration": state["iteration"], "max_iterations": state["max_iterations"],
            "recap": state["recap"], "memory": state["memory_context"],
        }, config)
        detail = (f"evaluación: {assessment.recap} | falta: {'; '.join(assessment.gaps) or 'nada'}"
                  + (f" | problemas en las fuentes: {'; '.join(assessment.source_problems)}"
                     if assessment.source_problems else "")
                  + f" | objetivo de la vuelta: {assessment.next_objective}")
        return {
            "assessment": assessment.model_dump(),
            "reasoning_trace": state["reasoning_trace"] + [
                logger.step(state["trace_id"], "REASON", detail, f"vuelta {state['iteration']}")],
        }

    return reason
