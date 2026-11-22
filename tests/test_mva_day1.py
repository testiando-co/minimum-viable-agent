"""Suite del MVA.

- Tests deterministas: corren sin API key (LLM simulado y embeddings falsos).
- Tests `@pytest.mark.llm`: llaman al LLM real. Se saltan si no hay ANTHROPIC_API_KEY.
"""

import json
import os

import pytest
from pydantic import ValidationError

from main_mva import INSTRUCTION_CONTRACT, PLAN_PROMPT, answer_question, generate_test_plan
from schemas import QATestPlanOutput, TestCaseItem, WorkItemFile
from tools_and_rag import (
    MOCKS_DIR,
    ROOT_DIR as ROOT,
    fetch_tfs_and_git_data,
    list_work_item_ids,
    load_meet_transcript,
    load_work_item,
)

_needs_api_key = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="requiere ANTHROPIC_API_KEY para llamar al LLM"
)


def live(test):
    """Variante con el modelo real: `pytest -m llm` para correr solo estas."""
    return pytest.mark.llm(_needs_api_key(test))


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
def test_descubre_los_wi_desde_la_carpeta_mocks():
    ids = list_work_item_ids()
    assert len(ids) == len(list(MOCKS_DIR.glob("wi_*.json")))
    assert {"WI-4052", "WI-4099"} <= set(ids)
    assert all(load_work_item(wi_id) is not None for wi_id in ids)


def test_descubrimiento_ignora_archivos_que_no_son_wi(tmp_path, monkeypatch):
    import tools_and_rag

    (tmp_path / "wi_7_demo.json").write_text("{}", encoding="utf-8")
    (tmp_path / "wi_12_demo.json").write_text("{}", encoding="utf-8")
    (tmp_path / "notas.txt").write_text("x", encoding="utf-8")
    (tmp_path / "wi_sin_numero.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(tools_and_rag, "MOCKS_DIR", tmp_path)
    assert list_work_item_ids() == ["WI-7", "WI-12"]  # orden numérico, no alfabético


def test_descubrimiento_sin_carpeta_mocks_devuelve_lista_vacia(tmp_path, monkeypatch):
    import tools_and_rag

    monkeypatch.setattr(tools_and_rag, "MOCKS_DIR", tmp_path / "no_existe")
    assert list_work_item_ids() == []

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


# --- CLI (main) --------------------------------------------------------------------
def test_main_sin_argumentos_recorre_los_wi_de_mocks(monkeypatch):
    import main_mva

    calls = []
    monkeypatch.setattr(main_mva, "generate_test_plan", lambda wi_id: calls.append(wi_id) or {})
    assert main_mva.main([]) == 0
    assert calls == list_work_item_ids()


def test_main_con_pregunta_usa_answer_question(monkeypatch):
    import main_mva

    calls = []
    monkeypatch.setattr(main_mva, "answer_question", lambda wi_id, q: calls.append((wi_id, q)) or "ok")
    assert main_mva.main(["WI-4052", "-q", "¿spinner?"]) == 0
    assert calls == [("WI-4052", "¿spinner?")]


def test_main_wi_inexistente_no_rompe(capsys):
    import main_mva

    assert main_mva.main(["WI-9999"]) == 0
    assert "Work Item not found" in capsys.readouterr().out


def test_main_sin_mocks_devuelve_error(monkeypatch):
    import main_mva

    monkeypatch.setattr(main_mva, "list_work_item_ids", lambda: [])
    assert main_mva.main([]) == 1


# --- Instruction Contract --------------------------------------------------------
def test_prompt_incluye_las_tres_fuentes():
    rendered = PLAN_PROMPT.format(
        wi_id="WI-4052", tfs_and_git="TFS_GIT", meet_fragments="MEET", threshold=0.85
    )
    assert "TFS_GIT" in rendered and "MEET" in rendered


def test_ic_contiene_reglas_de_negocio():
    for rule in ("prioridad sobre TFS", "ESCALATE_TO_HUMAN_MISSING_DATA", "NO inventes", "evidence_quote"):
        assert rule in INSTRUCTION_CONTRACT


# --- Pruebas de aceptación contra el LLM real -----------------------------------
@live
def test_prueba1_happy_path_wi_4052_llm_real():
    from evaluation import evaluate_plan

    golden = json.loads((ROOT / "evals" / "golden_set.json").read_text(encoding="utf-8"))
    result = evaluate_plan(next(p for p in golden["plans"] if p["wi_id"] == "WI-4052"))
    assert all(c["passed"] for c in result["checks"]), result["checks"]


@live
def test_prueba3_pregunta_adversaria_spinner_visa_llm_real():
    answer = answer_question("WI-4052", "¿Cuántos segundos dura la animación del spinner al pagar con Visa?")
    assert answer.startswith("No existe en las fuentes")


# =================================================================================
# Nivel 2: RAG híbrido (embeddings falsos deterministas: sin red ni modelo)
# =================================================================================
import shutil

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.runnables import RunnableLambda

import tools_and_rag as rag


@pytest.fixture
def fake_index(monkeypatch):
    index = rag.build_index(embeddings=DeterministicFakeEmbedding(size=64))
    monkeypatch.setattr(rag, "get_index", lambda: index)
    return index


def _meet_chunks(wi_file="wi_4052_happy_path.json"):
    docs = rag.split_work_item_file(MOCKS_DIR / wi_file)
    return [d for d in docs if d.metadata["source"] == "MEET_PO_DEMO"]


def test_split_respeta_tamano_y_solapamiento_de_20_caracteres():
    chunks = _meet_chunks()
    assert rag.CHUNK_OVERLAP == 20 and len(chunks) > 1
    assert all(len(c.page_content) <= rag.CHUNK_SIZE for c in chunks)
    for prev, nxt in zip(chunks, chunks[1:]):
        prev_end = prev.metadata["start_index"] + len(prev.page_content)
        overlap = prev_end - nxt.metadata["start_index"]
        assert 0 < overlap <= rag.CHUNK_OVERLAP


def test_cada_chunk_trae_archivo_contexto_y_fecha():
    docs = rag.split_work_item_file(MOCKS_DIR / "wi_4052_happy_path.json")
    assert {d.metadata["source"] for d in docs} == {"TFS", "MEET_PO_DEMO", "GIT_DIFF"}
    for d in docs:
        assert d.metadata["source_file"] == "wi_4052_happy_path.json"
        assert d.metadata["context"].startswith("WI-4052")
        assert d.metadata["date"] == rag.date.fromisoformat(d.metadata["date"]).isoformat()


def test_metadata_se_puede_ingresar_a_mano():
    docs = rag.split_work_item_file(
        MOCKS_DIR / "wi_4052_happy_path.json", context="Export de TFS del sprint 12", date_="2026-01-15"
    )
    assert {(d.metadata["context"], d.metadata["date"]) for d in docs} == {
        ("Export de TFS del sprint 12", "2026-01-15")
    }


def test_wi_4099_sin_commit_no_genera_chunks_de_git_y_marca_meet_corrupto():
    docs = rag.split_work_item_file(MOCKS_DIR / "wi_4099_exception_path.json")
    assert "GIT_DIFF" not in {d.metadata["source"] for d in docs}
    assert all(d.metadata["corrupt"] for d in docs if d.metadata["source"] == "MEET_PO_DEMO")


def test_indice_cubre_todos_los_archivos_de_mocks(fake_index):
    files = {os.path.basename(p) for p in rag.list_mock_files()}
    assert {d.metadata["source_file"] for d in fake_index.documents} == files


def test_busqueda_devuelve_max_k_5_ordenados_y_filtrados(fake_index):
    hits = fake_index.search("recargo por mora", where={"wi_id": "WI-4052"})
    assert 0 < len(hits) <= rag.TOP_K == 5
    scores = [h.metadata["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(h.metadata["wi_id"] == "WI-4052" for h in hits)


def test_reranking_por_terminos_bm25(fake_index, monkeypatch):
    monkeypatch.setattr(rag, "EMBEDDINGS_WEIGHT", 0.0)  # solo BM25
    top = fake_index.search("SURCHARGE_RATE days_overdue", where={"wi_id": "WI-4052"})[0]
    assert top.metadata["source"] == "GIT_DIFF" and top.metadata["score_bm25"] == 1.0


def test_reranking_por_embeddings(fake_index, monkeypatch):
    monkeypatch.setattr(rag, "EMBEDDINGS_WEIGHT", 1.0)  # solo embeddings
    target = _meet_chunks()[1]
    top = fake_index.search(target.page_content, where={"wi_id": "WI-4052"})[0]
    assert top.metadata["chunk_id"] == target.metadata["chunk_id"]


def test_reranking_prefiere_el_documento_mas_reciente(tmp_path, monkeypatch):
    for name in ("wi_1_viejo.json", "wi_2_nuevo.json"):
        shutil.copy(MOCKS_DIR / "wi_4052_happy_path.json", tmp_path / name)
    monkeypatch.setattr(rag, "MOCKS_DIR", tmp_path)
    index = rag.build_index(
        embeddings=DeterministicFakeEmbedding(size=64),
        metadata_overrides={
            "wi_1_viejo.json": {"date": "2025-01-01"},
            "wi_2_nuevo.json": {"date": "2026-11-01"},
        },
    )
    top = index.search("recargo por mora del 8.5%", where={"source": "MEET_PO_DEMO"})[0]
    assert top.metadata["source_file"] == "wi_2_nuevo.json"
    assert top.metadata["score_recency"] == 1.0


def test_rag_meet_wi_4099_avisa_transcripcion_corrupta(fake_index):
    result = json.loads(rag.search_meet_transcript_rag("WI-4099", "reembolso"))
    assert result["fragments"] == [] and "corrupto" in result["warning"]


def test_rag_meet_wi_4052_devuelve_fragmentos_con_puntaje(fake_index):
    result = json.loads(rag.search_meet_transcript_rag("WI-4052", "recargo por mora"))
    assert result["fragments"] and "warning" not in result
    for f in result["fragments"]:
        assert f["source"] == "MEET_PO_DEMO" and 0 <= f["score"] <= 1
        assert {"source_file", "context", "date", "score_embeddings", "score_bm25"} <= f.keys()


def test_rag_wi_inexistente_error_controlado(fake_index):
    assert json.loads(rag.search_meet_transcript_rag("WI-9999", "x")) == {"error": "Work Item not found"}
    assert json.loads(rag.search_work_item_rag("WI-9999", "x")) == {"error": "Work Item not found"}


def test_rag_wi_4099_excluye_fuentes_corruptas(fake_index):
    result = json.loads(rag.search_work_item_rag("WI-4099", "reembolso"))
    assert {f["source"] for f in result["fragments"]} == {"TFS"}
    assert len(result["warnings"]) == 2


def test_answer_question_le_pasa_al_llm_los_fragmentos_rag(fake_index):
    llm = FakeStructuredLLM(lambda schema, prompt: GroundedAnswer(found=False, answer="", citations=[]))
    answer_question("WI-4052", "¿spinner con Visa?", llm=llm)
    assert "fragmentos_rag" in llm.prompts[0] and "source_file" in llm.prompts[0]


@pytest.mark.skipif(not os.getenv("MVA_TEST_EMBEDDINGS"), reason="define MVA_TEST_EMBEDDINGS=1 para usar FastEmbed")
def test_rag_con_embeddings_reales_encuentra_el_recargo_de_meet():
    rag.get_index.cache_clear()
    result = json.loads(rag.search_meet_transcript_rag("WI-4052", "porcentaje del recargo por mora"))
    assert "8.5%" in result["fragments"][0]["text"]


# =================================================================================
# Nivel 4: grounding, FCoT y evaluación (LLM simulado: deterministas, sin API key)
# =================================================================================
from evaluation import run_evaluation
from grounding import compute_confidence, quote_in_source, verify_test_cases
from main_mva import run_test_plan
from schemas import EvidencedTestCase, GroundedAnswer, LLMTestPlanDraft


class FakeStructuredLLM:
    """LLM simulado: `responder(schema, prompt)` decide qué objeto devuelve."""

    def __init__(self, responder):
        self.responder = responder
        self.prompts: list[str] = []

    def with_structured_output(self, schema, **_):
        def respond(prompt_value):
            self.prompts.append(prompt_value.to_string())
            return self.responder(schema, self.prompts[-1])

        return RunnableLambda(respond)


def _case(test_id, category, source, then, quote):
    return EvidencedTestCase(
        test_id=test_id, category=category, given_context="Factura vencida en el checkout PSE",
        when_action="El usuario paga la factura", then_expected=then,
        grounding_source=source, evidence_quote=quote,
    )


GOOD_CASES = [
    _case("TC-01", "functional", "MEET_PO_DEMO", "Se aplica un recargo por mora del 8.5% (no el 5% de TFS)",
          "el recargo por mora debe ser exactamente del 8.5% para facturas mayores a 30 días"),
    _case("TC-02", "edge_case", "GIT_DIFF_DISCREPANCY", "Con 30 días exactos NO se aplica recargo",
          "if days_overdue >= 30"),
    _case("TC-03", "bug_hunt", "GIT_DIFF_DISCREPANCY", "El botón Pagar queda bloqueado 3 segundos",
          "BUG OCULTO: No activó bloqueo de 3s"),
]
GOOD_DRAFT = LLMTestPlanDraft(
    detected_discrepancies=["Recargo 8.5% (Meet) vs 5% (TFS)", ">= 30 en el código vs > 30",
                            "No se activó el bloqueo de 3 s pedido en Meet"],
    test_cases=GOOD_CASES,
)
INVENTED = _case("TC-99", "functional", "TFS", "Recargo del 12%", "el recargo es del 12% para todas las facturas")


def _good_responder(schema, prompt):
    if schema is LLMTestPlanDraft:
        return GOOD_DRAFT
    if "spinner" in prompt:
        return GroundedAnswer(found=False, answer="El dato no aparece en los fragmentos.", citations=[])
    return GroundedAnswer(found=True, answer="La PO pidió un recargo del 8.5%.", citations=[
        {"source": "MEET_PO_DEMO", "quote": "el recargo por mora debe ser exactamente del 8.5%"}])


# --- Grounding ---------------------------------------------------------------------
def test_cita_literal_se_encuentra_ignorando_tildes_mayusculas_y_espacios():
    assert quote_in_source("RECARGO  por mora   debe ser exactamente del 8.5%",
                           "el recargo por mora debe ser exactamente del 8.5%")
    assert quote_in_source("Ya NO es el 5% … ahora el recargo", "ya NO es el 5% de recargo, ahora el recargo")
    assert not quote_in_source("el recargo es del 12%", "el recargo por mora es del 8.5%")
    assert not quote_in_source("8.5", "8.5%")  # citas demasiado cortas no cuentan


def test_casos_con_citas_inventadas_se_descartan():
    report = verify_test_cases(load_work_item("WI-4052"), GOOD_CASES + [INVENTED])
    assert [c.test_id for c in report.grounded] == ["TC-01", "TC-02", "TC-03"]
    assert [r["test_id"] for r in report.rejected] == ["TC-99"]
    assert "evidence_quote" not in report.grounded[0].model_dump()  # se entrega el contrato limpio


def test_cita_de_otra_fuente_no_cuenta():
    misattributed = _case("TC-X", "functional", "TFS", "8.5%",
                          "el recargo por mora debe ser exactamente del 8.5%")  # está en Meet, no en TFS
    assert verify_test_cases(load_work_item("WI-4052"), [misattributed]).grounded == []


@pytest.mark.parametrize("sources,total,grounded,expected", [
    (3, 3, 3, 1.0), (3, 4, 3, 0.875), (3, 3, 1, 0.6667), (1, 0, 0, 0.1667),
])
def test_confianza_se_calcula_con_reglas(sources, total, grounded, expected):
    assert compute_confidence(sources, total, grounded) == expected


# --- Validadores del contrato de salida ------------------------------------------------
@pytest.mark.parametrize("kwargs", [
    {"status": "ESCALATE_TO_HUMAN_MISSING_DATA", "confidence_score": 0.9, "human_escalation_reason": "x"},
    {"status": "ESCALATE_TO_HUMAN_MISSING_DATA", "confidence_score": 0.2, "human_escalation_reason": None},
    {"status": "ESCALATE_TO_HUMAN_MISSING_DATA", "confidence_score": 0.2, "human_escalation_reason": "x",
     "test_cases": [GOOD_CASES[0].model_dump(exclude={"evidence_quote"})]},
    {"status": "READY_FOR_QA", "confidence_score": 0.5, "human_escalation_reason": None},
    {"status": "READY_FOR_QA", "confidence_score": 0.9, "human_escalation_reason": "x"},
    {"status": "READY_FOR_QA", "confidence_score": 0.9, "human_escalation_reason": None, "test_cases": []},
])
def test_validadores_rechazan_planes_incoherentes(kwargs):
    base = {"wi_id": "WI-1", "detected_discrepancies": [],
            "test_cases": [GOOD_CASES[0].model_dump(exclude={"evidence_quote"})]}
    with pytest.raises(ValidationError):
        QATestPlanOutput(**{**base, **kwargs})


# --- Pruebas de aceptación deterministas ----------------------------------------------
def test_prueba1_wi_4052_ready_con_las_3_trampas(fake_index):
    plan = generate_test_plan("WI-4052", llm=FakeStructuredLLM(_good_responder))
    assert isinstance(plan, QATestPlanOutput)
    assert plan.status == "READY_FOR_QA" and plan.confidence_score >= 0.85
    by_id = {c.test_id: c for c in plan.test_cases}
    assert "8.5%" in by_id["TC-01"].then_expected and by_id["TC-01"].grounding_source == "MEET_PO_DEMO"
    assert by_id["TC-02"].category == "edge_case" and "30 días exactos" in by_id["TC-02"].then_expected
    assert by_id["TC-03"].category in ("ui_visual", "bug_hunt") and "3 segundos" in by_id["TC-03"].then_expected


def test_prueba1_caso_inventado_se_descarta_del_plan(fake_index):
    draft = GOOD_DRAFT.model_copy(update={"test_cases": GOOD_CASES + [INVENTED]})
    plan, report = run_test_plan("WI-4052", llm=FakeStructuredLLM(lambda schema, prompt: draft))
    assert plan.status == "READY_FOR_QA" and plan.confidence_score == 0.875
    assert "TC-99" not in {c.test_id for c in plan.test_cases}
    assert report.rejected[0]["test_id"] == "TC-99"


def test_plan_mayormente_inventado_escala_sin_casos(fake_index):
    draft = GOOD_DRAFT.model_copy(update={"test_cases": [GOOD_CASES[0], INVENTED, INVENTED]})
    plan = generate_test_plan("WI-4052", llm=FakeStructuredLLM(lambda schema, prompt: draft))
    assert plan.status == "ESCALATE_TO_HUMAN_MISSING_DATA" and plan.confidence_score < 0.85
    assert plan.test_cases == [] and "1 de 3" in plan.human_escalation_reason


def test_prueba2_wi_4099_escala_sin_invocar_al_llm():
    plan = generate_test_plan("WI-4099", llm=ExplodingLLM())
    assert plan.status == "ESCALATE_TO_HUMAN_MISSING_DATA"
    assert plan.confidence_score < 0.85 and plan.test_cases == []
    reason = plan.human_escalation_reason.lower()
    assert "meet" in reason and "corrupto" in reason and "git" in reason and "null" in reason


def test_prueba3_pregunta_sin_respuesta_en_las_fuentes(fake_index):
    answer = answer_question("WI-4052", "¿Cuántos segundos dura la animación del spinner al pagar con Visa?",
                             llm=FakeStructuredLLM(_good_responder))
    assert answer.startswith("No existe en las fuentes del WI-4052")


def test_prueba3_cita_inventada_se_convierte_en_no_existe(fake_index):
    liar = FakeStructuredLLM(lambda schema, prompt: GroundedAnswer(
        found=True, answer="El spinner dura 2 segundos.",
        citations=[{"source": "MEET_PO_DEMO", "quote": "el spinner dura 2 segundos con Visa"}]))
    assert answer_question("WI-4052", "¿spinner?", llm=liar).startswith("No existe en las fuentes")


def test_respuesta_con_cita_verificada_incluye_la_fuente(fake_index):
    answer = answer_question("WI-4052", "¿Qué recargo pidió la PO?", llm=FakeStructuredLLM(_good_responder))
    assert "8.5%" in answer and "[MEET_PO_DEMO]" in answer


def test_prueba5_audit_trail_registra_recap_reason_verify(fake_index, tmp_path, monkeypatch):
    import logger

    monkeypatch.setattr(logger, "LOG_FILE", tmp_path / "audit.jsonl")
    generate_test_plan("WI-4052", llm=FakeStructuredLLM(_good_responder))
    record = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    phases = [step["phase"] for step in record["reasoning_trace"]]
    assert [p for i, p in enumerate(phases) if p not in phases[:i]] == ["RECAP", "REASON", "VERIFY"]
    assert {"trace_id", "timestamp", "tools", "output"} <= record.keys()
    assert record["tools"][0]["input"] == {"wi_id": "WI-4052"} and record["tools"][0]["output"]
    assert record["output"]["grounding"]["grounded_cases"] == 3


# --- Evaluación ---------------------------------------------------------------------
def test_evaluacion_con_agente_correcto_pasa_todo(fake_index):
    summary = run_evaluation(llm=FakeStructuredLLM(_good_responder))["summary"]
    assert summary["accuracy"] == 1.0 and summary["check_pass_rate"] == 1.0
    assert summary["grounding_rate"] == 1.0 and summary["hallucination_guard"] == 1.0


def test_evaluacion_detecta_trampa_faltante_y_alucinacion(fake_index):
    def bad(schema, prompt):
        if schema is LLMTestPlanDraft:  # olvida la trampa 3
            return GOOD_DRAFT.model_copy(update={"test_cases": GOOD_CASES[:2]})
        return GroundedAnswer(found=True, answer="Dura 2 segundos.",
                              citations=[{"source": "MEET_PO_DEMO", "quote": "dura 2 segundos el spinner"}])

    evaluation = run_evaluation(llm=FakeStructuredLLM(bad))
    failed = {c["name"] for r in evaluation["results"] for c in r["checks"] if not c["passed"]}
    assert "Trampa 3: bloqueo de 3 segundos del botón Pagar" in failed
    assert "responde con citas verificadas" in failed  # la cita inventada no se acepta
    assert evaluation["summary"]["accuracy"] < 1.0
