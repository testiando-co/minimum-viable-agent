"""Pruebas de aceptación del Día 01 (las 5 que pide la guía), contra el LLM real.

Usan el modelo de MVA_MODEL (u Ollama con LOCAL_LLM=1) y se saltan si no hay ninguno
configurado. Solo estas: `uv run pytest -v -m llm`.

1. WI-4052 → READY_FOR_QA con las 3 trampas.
2. WI-4099 → ESCALATE_TO_HUMAN_MISSING_DATA sin inventar pruebas.
3. Pregunta del spinner → responde que no pudo verificarlo.
4. WI-9999 → error controlado, sin traceback.
5. Audit Trail con RECAP → REASON → VERIFY (y `pytest -v` en verde).

Los tests unitarios (cada pieza por separado, con LLM simulado) están en `test_unitarias.py`.
"""

import json

import pytest

from main_mva import answer_question, generate_test_plan, llm_available
from tools_and_rag import ROOT_DIR as ROOT

_needs_api_key = pytest.mark.skipif(
    not llm_available(), reason="requiere la API key del proveedor de MVA_MODEL o LOCAL_LLM=1"
)


def live(test):
    """Variante con el modelo real: `pytest -m llm` para correr solo estas."""
    return pytest.mark.llm(_needs_api_key(test))


@pytest.fixture
def real_index():
    """Índice con los embeddings reales (FastEmbed u Ollama), reconstruido por test."""
    import tools_and_rag

    tools_and_rag.get_index.cache_clear()
    yield
    tools_and_rag.get_index.cache_clear()


def _golden_plan(wi_id):
    golden = json.loads((ROOT / "evals" / "golden_set.json").read_text(encoding="utf-8"))
    return next(p for p in golden["plans"] if p["wi_id"] == wi_id)


def _failed(result):
    return [(c["name"], c["detail"]) for c in result["checks"] if not c["passed"]]


@live
def test_prueba1_happy_path_wi_4052_llm_real(real_index):
    from evaluation import evaluate_plan

    result = evaluate_plan(_golden_plan("WI-4052"))
    assert not _failed(result), _failed(result)


@live
def test_prueba2_exception_path_wi_4099_llm_real(real_index):
    from evaluation import evaluate_plan

    result = evaluate_plan(_golden_plan("WI-4099"))
    assert not _failed(result), _failed(result)


@live
def test_prueba3_pregunta_adversaria_spinner_visa_llm_real(real_index):
    answer = answer_question("WI-4052", "¿Cuántos segundos dura la animación del spinner al pagar con Visa?")
    assert answer.startswith("No pude verificar"), answer


@live
def test_prueba4_wi_9999_llm_real(real_index, capsys):
    import main_mva

    assert main_mva.main(["WI-9999"]) == 0
    assert "Work Item not found" in capsys.readouterr().out


@live
def test_prueba5_audit_trail_llm_real(real_index, tmp_path, monkeypatch):
    import logger

    monkeypatch.setattr(logger, "LOG_FILE", tmp_path / "audit.jsonl")
    generate_test_plan("WI-4052")
    record = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    phases = [step["phase"] for step in record["reasoning_trace"]]
    assert [p for i, p in enumerate(phases) if p not in phases[:i]] == ["RECAP", "REASON", "VERIFY"]
