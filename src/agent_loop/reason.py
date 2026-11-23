"""REASON (FCoT: REASON) — el LLM propone, bajo el Instruction Contract.

- Plan: decide si escala a un humano; si no, discrepancias y casos de prueba, cada uno con
  `evidence_quote` (LLMTestPlanDraft).
- Pregunta: respuesta con citas literales (GroundedAnswer).
La confianza no la decide el LLM: la calcula VERIFY con reglas.
"""

from typing import Any, Callable, Optional

from langchain_core.runnables import RunnableConfig

import logger
from agent_loop.llm import structured
from agent_loop.prompts import PLAN_PROMPT, QUESTION_PROMPT
from agent_loop.state import AgentState
from schemas import GroundedAnswer, LLMTestPlanDraft


def make_reason(llm: Optional[Any] = None) -> Callable[[AgentState, RunnableConfig], dict[str, Any]]:
    """Crea el nodo REASON con el LLM indicado (o el configurado en .env)."""

    def reason(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        # `config` propaga el contexto del grafo: la llamada al LLM queda dentro de la misma
        # traza de LangSmith en vez de abrir una traza nueva.
        trace_id = state["trace_id"]
        if state["mode"] == "plan":
            draft: LLMTestPlanDraft = (PLAN_PROMPT | structured(llm, LLMTestPlanDraft)).invoke(state["prompt_vars"], config)
            detail = (f"decisión del agente: {'ESCALAR' if draft.escalate_to_human else 'plan de pruebas'}"
                      + (f" ({draft.escalation_reason})" if draft.escalate_to_human else "")
                      + f"; {len(draft.detected_discrepancies)} discrepancias: {draft.detected_discrepancies}; "
                      f"{len(draft.test_cases)} casos propuestos")
        else:
            draft: GroundedAnswer = (QUESTION_PROMPT | structured(llm, GroundedAnswer)).invoke(state["prompt_vars"], config)
            detail = f"found={draft.found}; {len(draft.citations)} citas"
        return {
            "draft": draft.model_dump(),
            "reasoning_trace": state["reasoning_trace"] + [logger.step(trace_id, "REASON", detail)],
        }

    return reason
