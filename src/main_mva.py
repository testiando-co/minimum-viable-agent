"""MVA de QA — punto de entrada (CLI) y API pública del agente.

El agente vive en `agent_loop/` como un bucle de LangGraph con la anatomía
Sense → Reason → Plan → Act ↺ (vuelve a Sense hasta alcanzar el objetivo; ver agent_loop/__init__.py):

- Nivel 1: modelo elegido desde .env e Instruction Contract (agent_loop/llm.py, prompts.py).
- Nivel 2: RAG híbrido sobre mocks/ con Chroma persistente + BM25 (tools_and_rag.py).
- Nivel 4: grounding verificable y confianza por reglas (grounding.py, agent_loop/verify.py).
- Nivel 5: agente único orquestado con LangGraph (REASON y PLAN con el LLM, ACT ejecuta las
  herramientas), memoria de corto plazo (en memoria) y de largo plazo (Chroma), y una sola
  traza de LangSmith por ejecución.

Fractal Chain of Thought en el Audit Trail: RECAP → REASON → VERIFY en cada vuelta y en cada
paso que ejecuta ACT.

El modelo se elige con MVA_MODEL en formato `proveedor:modelo` de LangChain (ver
agent_loop/llm.py; por defecto `anthropic:claude-opus-5-5`). Recomendado:
`google_genai:gemini-3.6-flash` (gratis). Con LOCAL_LLM=1 se usa Ollama local.
"""

import argparse
import json
import sys
from typing import Any, Optional, Union

from dotenv import load_dotenv

load_dotenv()

from langchain_core.language_models import BaseChatModel  # noqa: E402
from langsmith import traceable  # noqa: E402

from agent_loop.graph import run_agent  # noqa: E402
from agent_loop.llm import (  # noqa: E402,F401  (se re-exportan para tests y evaluation.py)
    DEFAULT_MODEL,
    build_llm,
    llm_available,
    llm_errors,
    model_name,
)
from agent_loop.prompts import (  # noqa: E402,F401
    INSTRUCTION_CONTRACT,
    PLAN_PROMPT,
    QUESTION_PROMPT,
    REASON_PROMPT,
    TEST_PLAN_PROMPT,
    UNVERIFIED_ANSWER,
)
from grounding import GroundingReport  # noqa: E402
from schemas import QATestPlanOutput, TestCaseItem  # noqa: E402
from tools_and_rag import list_work_item_ids  # noqa: E402


# --- API del agente -----------------------------------------------------------------
def run_test_plan(
    wi_id: str, llm: Optional[BaseChatModel] = None, session_id: Optional[str] = None
) -> tuple[Union[QATestPlanOutput, dict[str, str]], Optional[GroundingReport]]:
    """Plan de pruebas + reporte de grounding (lo usa evaluation.py)."""
    state = run_agent("plan", wi_id, llm=llm, session_id=session_id)
    if state["route"] == "not_found":
        return state["result"], None
    g = state["grounding"]
    report = GroundingReport(
        valid_sources=g["valid_sources"], total_cases=g["total_cases"],
        grounded=[TestCaseItem.model_validate(c) for c in state["grounded_cases"]], rejected=g["rejected"],
    )
    return QATestPlanOutput.model_validate(state["result"]), report


def generate_test_plan(
    wi_id: str, llm: Optional[BaseChatModel] = None, session_id: Optional[str] = None
) -> Union[QATestPlanOutput, dict[str, str]]:
    """Plan de pruebas validado, o `{"error": "Work Item not found"}` si el WI no existe."""
    return run_test_plan(wi_id, llm, session_id)[0]


def run_question(
    wi_id: str, question: str, llm: Optional[BaseChatModel] = None, session_id: Optional[str] = None
) -> tuple[str, dict[str, Any]]:
    """Respuesta + detalle de grounding (lo usa evaluation.py)."""
    state = run_agent("question", wi_id, question=question, llm=llm, session_id=session_id)
    return state["result"], state["details"]


def answer_question(
    wi_id: str, question: str, llm: Optional[BaseChatModel] = None, session_id: Optional[str] = None
) -> str:
    """Respuesta con sus fuentes, o que no se pudo verificar si no hay respaldo verificable."""
    return run_question(wi_id, question, llm, session_id)[0]


# --- CLI --------------------------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVA de QA: plan de pruebas o preguntas sobre un WI.")
    parser.add_argument(
        "wi_ids", nargs="*", metavar="WI_ID",
        help="Work Items a analizar (p. ej. WI-4052). Sin valor: todos los de mocks/.",
    )
    parser.add_argument(
        "-q", "--pregunta",
        help="Hace esta pregunta sobre cada WI en vez de generar el plan de pruebas.",
    )
    return parser.parse_args(argv)


@traceable(name="MVA CLI", run_type="chain")
def _run_cli(wi_ids: list[str], pregunta: Optional[str]) -> None:
    """Toda la ejecución del CLI queda en una sola traza; cada WI es una rama del agente."""
    for wi_id in wi_ids:
        print(f"\n=== {wi_id} ===")
        if pregunta:
            print(answer_question(wi_id, pregunta))
        else:
            plan = generate_test_plan(wi_id)
            data = plan.model_dump() if isinstance(plan, QATestPlanOutput) else plan
            print(json.dumps(data, ensure_ascii=False, indent=2))


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    wi_ids = args.wi_ids or list_work_item_ids()
    if not wi_ids:
        print("No hay Work Items en mocks/ (se esperan archivos wi_<id>_*.json).", file=sys.stderr)
        return 1
    try:
        _run_cli(wi_ids, args.pregunta)
    except llm_errors() as e:
        print(f"\nError al llamar al modelo ({model_name()}): {e}", file=sys.stderr)
        print("Revisa la API key del proveedor en tu .env, o que Ollama esté corriendo si LOCAL_LLM=1.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
