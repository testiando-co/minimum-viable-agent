"""Nivel 4 — Grounding: verificar contra las fuentes lo que dice el LLM.

- Cada caso de prueba y cada respuesta traen una cita textual (`evidence_quote`).
  La cita se busca en el texto real de la fuente que dice citar; si no aparece, el
  caso se descarta (no se acepta nada sin respaldo verificable).
- El `confidence_score` no lo decide el LLM: se calcula con reglas verificables.
"""

import re
import unicodedata
from dataclasses import dataclass, field

from schemas import EvidencedTestCase, TestCaseItem, WorkItemData
from tools_and_rag import source_texts

TOTAL_SOURCES = 3  # TFS, MEET_PO_DEMO, GIT_DIFF
SOURCES_WEIGHT = 0.5  # el resto lo pesa la proporción de casos con respaldo
MIN_QUOTE_CHARS = 8

# grounding_source del contrato -> fuente donde se busca la cita
GROUNDING_TO_SOURCE = {"TFS": "TFS", "MEET_PO_DEMO": "MEET_PO_DEMO", "GIT_DIFF_DISCREPANCY": "GIT_DIFF"}


def normalize(text: str) -> str:
    """Minúsculas, sin tildes, sin comillas y con espacios colapsados."""
    plain = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    plain = re.sub(r"[\"'`“”‘’«»]", "", plain)
    return re.sub(r"\s+", " ", plain).strip()


def quote_in_source(quote: str, source_text: str) -> bool:
    """True si la cita aparece literalmente en la fuente.

    Se permiten elipsis ("..." o "…"): cada tramo debe aparecer, en orden.
    """
    haystack = normalize(source_text)
    segments = [normalize(s) for s in re.split(r"\.\.\.|…", quote)]
    segments = [s for s in segments if s]
    if not segments or sum(len(s) for s in segments) < MIN_QUOTE_CHARS:
        return False
    position = 0
    for segment in segments:
        found = haystack.find(segment, position)
        if found < 0:
            return False
        position = found + len(segment)
    return True


def compute_confidence(valid_sources: int, total_cases: int, grounded_cases: int) -> float:
    """0.5 · fuentes válidas / 3 + 0.5 · casos con respaldo / casos propuestos."""
    sources_ratio = valid_sources / TOTAL_SOURCES
    grounding_ratio = grounded_cases / total_cases if total_cases else 0.0
    return round(SOURCES_WEIGHT * sources_ratio + (1 - SOURCES_WEIGHT) * grounding_ratio, 4)


@dataclass
class GroundingReport:
    valid_sources: int
    total_cases: int = 0
    grounded: list[TestCaseItem] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        return compute_confidence(self.valid_sources, self.total_cases, len(self.grounded))

    @property
    def grounding_rate(self) -> float:
        return len(self.grounded) / self.total_cases if self.total_cases else 0.0

    def as_dict(self) -> dict:
        return {
            "valid_sources": self.valid_sources,
            "total_cases": self.total_cases,
            "grounded_cases": len(self.grounded),
            "grounding_rate": round(self.grounding_rate, 4),
            "confidence": self.confidence,
            "rejected": self.rejected,
        }


def valid_source_count(wi: WorkItemData) -> int:
    return TOTAL_SOURCES - len(wi.missing_or_corrupt_sources())


def verify_test_cases(wi: WorkItemData, cases: list[EvidencedTestCase]) -> GroundingReport:
    """Acepta solo los casos cuya cita aparece en la fuente que dicen citar."""
    texts = source_texts(wi)
    report = GroundingReport(valid_sources=valid_source_count(wi), total_cases=len(cases))
    for case in cases:
        source = GROUNDING_TO_SOURCE[case.grounding_source]
        if quote_in_source(case.evidence_quote, texts.get(source, "")):
            report.grounded.append(TestCaseItem(**case.model_dump(exclude={"evidence_quote"})))
        else:
            report.rejected.append({
                "test_id": case.test_id,
                "grounding_source": case.grounding_source,
                "evidence_quote": case.evidence_quote,
                "reason": f"la cita no aparece en {source}",
            })
    return report


def verify_citation(wi: WorkItemData, source: str, quote: str) -> bool:
    return quote_in_source(quote, source_texts(wi).get(source, ""))
