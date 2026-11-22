"""Suite del MVA.

- Tests offline: contratos, herramientas y prompt. Siempre corren.
- Tests `live`: llaman al LLM real. Se saltan si no hay ANTHROPIC_API_KEY.
"""

import json
import os

import pytest
from pydantic import ValidationError

from main_mva import INSTRUCTION_CONTRACT, PLAN_PROMPT, answer_question, generate_test_plan
from schemas import QATestPlanOutput, TestCaseItem, WorkItemFile
from tools_and_rag import MOCKS_DIR, fetch_tfs_and_git_data, load_meet_transcript, load_work_item

live = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="requiere ANTHROPIC_API_KEY para llamar al LLM"
)


class ExplodingLLM:
    """Falla si alguien intenta usarla: prueba que no se llama al modelo sin necesidad."""

    def __getattr__(self, name):
        raise AssertionError("no se debía invocar al LLM")


# --- Contratos ------------------------------------------------------------------
def test_confidence_score_fuera_de_rango_es_rechazado():
    with pytest.raises(ValidationError):
        QATestPlanOutput(
            wi_id="WI-1", status="READY_FOR_QA", confidence_score=1.5,
            detected_discrepancies=[], human_escalation_reason=None, test_cases=[],
        )


def test_grounding_source_invalido_es_rechazado():
    with pytest.raises(ValidationError):
        TestCaseItem(
            test_id="TC-1", category="functional", given_context="g", when_action="w",
            then_expected="t", grounding_source="SLACK",
        )


# --- Esquemas de entrada (mocks/) ------------------------------------------------
@pytest.mark.parametrize("path", sorted(MOCKS_DIR.glob("wi_*.json")), ids=lambda p: p.name)
def test_cada_mock_cumple_el_esquema_de_entrada(path):
    WorkItemFile.model_validate_json(path.read_text(encoding="utf-8"))


def test_wi_4052_tiene_datos_completos():
    wi = load_work_item("WI-4052")
    assert wi.missing_or_corrupt_sources() == []
    assert "\n" in wi.git_commit_diff.diff_summary  # saltos de línea normalizados
    assert "\\n" not in wi.git_commit_diff.diff_summary


def test_wi_4099_detecta_meet_corrupto_y_commit_null():
    problems = load_work_item("WI-4099").missing_or_corrupt_sources()
    assert any(p.startswith("MEET_PO_DEMO") and "corrupto" in p for p in problems)
    assert any(p.startswith("GIT_DIFF") and "null" in p for p in problems)


def test_esquema_rechaza_campo_desconocido():
    raw = json.loads((MOCKS_DIR / "wi_4099_exception_path.json").read_text(encoding="utf-8"))
    raw["data"]["campo_inventado"] = 1
    with pytest.raises(ValidationError):
        WorkItemFile.model_validate(raw)


def test_esquema_rechaza_wi_id_mal_formado():
    raw = json.loads((MOCKS_DIR / "wi_4099_exception_path.json").read_text(encoding="utf-8"))
    raw["data"]["wi_id"] = "4099"
    with pytest.raises(ValidationError):
        WorkItemFile.model_validate(raw)


# --- Herramientas (datos leídos de mocks/) --------------------------------------
def test_fetch_wi_4052_devuelve_tfs_y_git():
    data = json.loads(fetch_tfs_and_git_data("WI-4052"))
    assert data["tfs_specification"]["business_rule"]
    assert data["git_commit_diff"]["commit_hash"] == "a8f92b1"
    assert "meet_transcript_po_demo" not in data


def test_fetch_wi_4099_tiene_commit_null():
    data = json.loads(fetch_tfs_and_git_data("WI-4099"))
    assert data["git_commit_diff"] is None


def test_meet_transcript_se_lee_de_mocks():
    assert "8.5%" in load_meet_transcript("WI-4052")


# --- Prueba 4: WI inexistente → error controlado --------------------------------
@pytest.mark.parametrize("wi_id", ["WI-9999", "basura", ""])
def test_wi_inexistente_devuelve_error_controlado(wi_id):
    assert json.loads(fetch_tfs_and_git_data(wi_id)) == {"error": "Work Item not found"}


def test_plan_wi_9999_no_llama_al_llm():
    assert generate_test_plan("WI-9999", llm=ExplodingLLM()) == {"error": "Work Item not found"}


def test_pregunta_wi_9999_no_llama_al_llm():
    assert answer_question("WI-9999", "¿lo que sea?", llm=ExplodingLLM()) == "Work Item not found"


# --- Instruction Contract --------------------------------------------------------
def test_prompt_incluye_las_tres_fuentes():
    rendered = PLAN_PROMPT.format(
        wi_id="WI-4052", tfs_and_git="TFS_GIT", meet_transcript="MEET", threshold=0.85
    )
    assert "TFS_GIT" in rendered and "MEET" in rendered and "0.85" in rendered


def test_ic_contiene_reglas_de_negocio():
    for rule in ("prioridad sobre TFS", "ESCALATE_TO_HUMAN_MISSING_DATA", "NO inventes"):
        assert rule in INSTRUCTION_CONTRACT


# --- Pruebas de aceptación contra el LLM real -----------------------------------
@live
def test_prueba1_happy_path_wi_4052():
    plan = generate_test_plan("WI-4052")
    assert plan["status"] == "READY_FOR_QA"
    assert plan["confidence_score"] >= 0.85
    cases = json.dumps(plan["test_cases"], ensure_ascii=False)
    assert "8.5" in cases
    assert any(c["grounding_source"] == "MEET_PO_DEMO" for c in plan["test_cases"])
    assert any(c["category"] in ("bug_hunt", "edge_case") for c in plan["test_cases"])
    assert len(plan["detected_discrepancies"]) >= 3


@live
def test_prueba2_exception_path_wi_4099():
    plan = generate_test_plan("WI-4099")
    assert plan["status"] == "ESCALATE_TO_HUMAN_MISSING_DATA"
    assert plan["confidence_score"] < 0.85
    assert plan["test_cases"] == []
    assert plan["human_escalation_reason"]


@live
def test_prueba3_pregunta_adversaria_spinner_visa():
    answer = answer_question(
        "WI-4052", "¿Cuántos segundos dura la animación del spinner al pagar con Visa?"
    ).lower()
    assert "no existe" in answer or "no aparece" in answer or "no se menciona" in answer
