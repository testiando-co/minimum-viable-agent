"""Acceso a los datos del Work Item en mocks/ (nunca datos quemados en el código).

Nivel 0-1: carga y validación de los JSON de mocks/.
Nivel 2: RAG híbrido sobre todos los archivos de mocks/:
  1. Split: cada fuente del WI (TFS, Meet, Git) se parte en chunks de CHUNK_SIZE
     caracteres con CHUNK_OVERLAP de solapamiento, con metadata (archivo, contexto, fecha).
  2. Recuperación de candidatos con dos retrievers: embeddings locales (FAISS) y
     términos (BM25).
  3. Reranking: puntaje híbrido (embeddings + BM25) ajustado por la fecha (relevancia
     temporal) y se devuelven los TOP_K mejores.
"""

import json
import os
import re
import unicodedata
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import ValidationError
from rank_bm25 import BM25Okapi

from schemas import WorkItemData, WorkItemFile

ROOT_DIR = Path(__file__).resolve().parent.parent
MOCKS_DIR = ROOT_DIR / "mocks"
NOT_FOUND = {"error": "Work Item not found"}
_MOCK_FILE_RE = re.compile(r"^wi_(\d+)_.+\.json$")


# =================================================================================
# Nivel 0-1: carga de Work Items
# =================================================================================
def list_work_item_ids() -> list[str]:
    """Descubre los WI disponibles en mocks/: `wi_4052_happy_path.json` -> `WI-4052`."""
    return sorted(
        {f"WI-{_MOCK_FILE_RE.match(os.path.basename(p)).group(1)}" for p in list_mock_files()},
        key=lambda wi_id: int(wi_id.removeprefix("WI-")),
    )


def list_mock_files() -> list[str]:
    """Rutas de los archivos wi_<id>_*.json de mocks/, ordenadas por nombre."""
    mocks_dir = os.fspath(MOCKS_DIR)
    if not os.path.isdir(mocks_dir):
        return []
    return sorted(
        os.path.join(mocks_dir, name)
        for name in os.listdir(mocks_dir)
        if _MOCK_FILE_RE.match(name) and os.path.isfile(os.path.join(mocks_dir, name))
    )


def _find_mock_file(wi_id: str) -> Optional[Path]:
    """`WI-4052` -> `mocks/wi_4052_*.json`."""
    number = wi_id.strip().upper().removeprefix("WI-")
    if not number.isdigit():
        return None
    matches = sorted(MOCKS_DIR.glob(f"wi_{number}_*.json"))
    return matches[0] if matches else None


def _read_work_item(path: os.PathLike | str) -> Optional[WorkItemData]:
    try:
        return WorkItemFile.model_validate_json(Path(path).read_text(encoding="utf-8")).data
    except (ValidationError, OSError):
        return None


def load_work_item(wi_id: str) -> Optional[WorkItemData]:
    """WI validado contra `WorkItemFile`, o None si no existe, es ilegible o no cumple el esquema."""
    path = _find_mock_file(wi_id)
    return None if path is None else _read_work_item(path)


def fetch_tfs_and_git_data(wi_id: str) -> str:
    """String JSON con `tfs_specification` y `git_commit_diff`, o el error controlado."""
    data = load_work_item(wi_id)
    if data is None:
        return json.dumps(NOT_FOUND)
    return data.model_dump_json(
        include={"wi_id", "title", "tfs_specification", "git_commit_diff"}, indent=2
    )


def load_meet_transcript(wi_id: str) -> Optional[str]:
    """Transcripción completa de la demo con la PO."""
    data = load_work_item(wi_id)
    return None if data is None else data.meet_transcript_po_demo


# =================================================================================
# Nivel 2: RAG híbrido (split + embeddings + BM25 + reranking por fecha)
# =================================================================================
CHUNK_SIZE = 200          # caracteres por chunk
CHUNK_OVERLAP = 20        # caracteres compartidos entre chunks consecutivos
TOP_K = 5                 # fragmentos que devuelve el reranking
CANDIDATES_PER_RETRIEVER = 20
EMBEDDINGS_WEIGHT = 0.5   # peso de embeddings; BM25 pesa 1 - EMBEDDINGS_WEIGHT
RECENCY_WEIGHT = 0.2      # cuánto puede bajar el puntaje un documento antiguo
RECENCY_HALF_LIFE_DAYS = 30

DEFAULT_EMBEDDINGS_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDINGS_CACHE_DIR = ROOT_DIR / ".cache" / "fastembed"

SOURCE_CONTEXT = {
    "TFS": "especificación del ticket en TFS (estructurada, puede estar desactualizada)",
    "MEET_PO_DEMO": "transcripción de la demo con la Product Owner en Google Meet",
    "GIT_DIFF": "resumen del git diff del commit del desarrollador",
}

_STOPWORDS = set(
    "a al con de del el en es la las lo los para por que se su un una y o ya "
    "the of to and in".split()
)


def _tokenize(text: str) -> list[str]:
    """Minúsculas, sin tildes, conservando números como 8.5 y nombres como days_overdue."""
    plain = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    return [t for t in re.findall(r"\w+(?:[.,]\w+)*", plain) if t not in _STOPWORDS]


def source_texts(wi: WorkItemData) -> dict[str, str]:
    """Texto indexable de cada fuente presente en el WI."""
    tfs = wi.tfs_specification.model_dump()
    texts = {"TFS": "\n".join(f"{k}: {v}" for k, v in tfs.items() if v is not None)}
    if wi.meet_transcript_po_demo:
        texts["MEET_PO_DEMO"] = wi.meet_transcript_po_demo
    if wi.git_commit_diff is not None:
        git = wi.git_commit_diff
        texts["GIT_DIFF"] = (
            f"commit {git.commit_hash} ({git.author})\n"
            f"archivos: {', '.join(git.files_modified)}\n{git.diff_summary}"
        )
    return texts


def _file_date(path: os.PathLike | str) -> str:
    return datetime.fromtimestamp(os.path.getmtime(path)).date().isoformat()


def split_work_item_file(
    path: os.PathLike | str,
    *,
    context: Optional[str] = None,
    date_: Optional[str] = None,
) -> list[Document]:
    """Parte un archivo de mocks/ en chunks con metadata.

    Metadata de cada chunk:
      - source_file: nombre del archivo de origen.
      - context: de dónde proviene (por defecto: WI, título y tipo de fuente).
      - date: fecha ISO para la relevancia temporal (por defecto: modificación del archivo).
      - wi_id, source (TFS / MEET_PO_DEMO / GIT_DIFF), corrupt, chunk, start_index.
    `context` y `date_` permiten ingresar la metadata a mano.
    """
    wi = _read_work_item(path)
    if wi is None:
        return []
    doc_date = date.fromisoformat(date_).isoformat() if date_ else _file_date(path)
    corrupt_sources = {problem.split(":")[0] for problem in wi.missing_or_corrupt_sources()}
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )

    documents: list[Document] = []
    for source, text in source_texts(wi).items():
        metadata = {
            "source_file": os.path.basename(path),
            "context": context or f"{wi.wi_id} — {wi.title} · {SOURCE_CONTEXT[source]}",
            "date": doc_date,
            "wi_id": wi.wi_id,
            "source": source,
            "corrupt": source in corrupt_sources,
        }
        for i, chunk in enumerate(splitter.create_documents([text], metadatas=[metadata])):
            chunk.metadata["chunk"] = i
            chunk.metadata["chunk_id"] = f"{metadata['source_file']}#{source}#{i}"
            documents.append(chunk)
    return documents


class HybridIndex:
    """FAISS (embeddings) + BM25 (términos) sobre los mismos chunks, con reranking."""

    def __init__(self, documents: list[Document], embeddings: Embeddings):
        from langchain_community.vectorstores import FAISS

        if not documents:
            raise ValueError("No hay documentos para indexar")
        self.documents = documents
        self._position = {d.metadata["chunk_id"]: i for i, d in enumerate(documents)}
        self.vectorstore = FAISS.from_documents(documents, embeddings)
        self.bm25 = BM25Okapi([_tokenize(d.page_content) or [""] for d in documents])
        self.newest = max(date.fromisoformat(d.metadata["date"]) for d in documents)

    def _recency(self, doc: Document) -> float:
        """1.0 para el documento más reciente; la mitad cada RECENCY_HALF_LIFE_DAYS días."""
        age_days = (self.newest - date.fromisoformat(doc.metadata["date"])).days
        return 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)

    def search(self, query: str, k: int = TOP_K, where: Optional[dict[str, Any]] = None) -> list[Document]:
        """Devuelve hasta k chunks rerankeados; cada uno trae sus puntajes en la metadata."""
        where = where or {}
        allowed = [
            i for i, d in enumerate(self.documents)
            if all(d.metadata.get(key) == value for key, value in where.items())
        ]
        if not allowed:
            return []

        # 1) Candidatos por embeddings (distancia L2 -> similitud en (0, 1]).
        hits = self.vectorstore.similarity_search_with_score(
            query, k=min(CANDIDATES_PER_RETRIEVER, len(allowed)),
            filter=where or None, fetch_k=len(self.documents),
        )
        emb = {self._position[d.metadata["chunk_id"]]: 1 / (1 + float(dist)) for d, dist in hits}

        # 2) Candidatos por términos (BM25).
        bm25_all = [float(s) for s in self.bm25.get_scores(_tokenize(query))]
        bm25_top = sorted(allowed, key=lambda i: bm25_all[i], reverse=True)[:CANDIDATES_PER_RETRIEVER]

        # 3) Reranking: normaliza cada señal contra su máximo y combina.
        candidates = set(emb) | set(bm25_top)
        emb_max = max(emb.values(), default=0) or 1
        bm25_max = max((max(bm25_all[i], 0) for i in candidates), default=0)

        ranked = []
        for i in candidates:
            doc = self.documents[i]
            s_emb = emb.get(i, 0.0) / emb_max
            s_bm25 = max(bm25_all[i], 0) / bm25_max if bm25_max > 0 else 0.0
            s_recency = self._recency(doc)
            hybrid = EMBEDDINGS_WEIGHT * s_emb + (1 - EMBEDDINGS_WEIGHT) * s_bm25
            score = hybrid * ((1 - RECENCY_WEIGHT) + RECENCY_WEIGHT * s_recency)
            ranked.append((score, s_emb, s_bm25, s_recency, doc))
        ranked.sort(key=lambda r: (-r[0], r[4].metadata["chunk_id"]))

        results = []
        for score, s_emb, s_bm25, s_recency, doc in ranked[:k]:
            results.append(Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "score": round(score, 4),
                    "score_embeddings": round(s_emb, 4),
                    "score_bm25": round(s_bm25, 4),
                    "score_recency": round(s_recency, 4),
                },
            ))
        return results


def get_embeddings() -> Embeddings:
    """Embeddings locales con FastEmbed (ONNX, sin API key). Modelo vía MVA_EMBEDDINGS_MODEL."""
    from langchain_community.embeddings import FastEmbedEmbeddings

    return FastEmbedEmbeddings(
        model_name=os.getenv("MVA_EMBEDDINGS_MODEL", DEFAULT_EMBEDDINGS_MODEL),
        cache_dir=os.fspath(EMBEDDINGS_CACHE_DIR),
    )


def build_index(
    embeddings: Optional[Embeddings] = None,
    metadata_overrides: Optional[dict[str, dict[str, str]]] = None,
) -> HybridIndex:
    """Indexa todos los archivos de mocks/.

    `metadata_overrides` permite ingresar la metadata por archivo:
    {"wi_4052_happy_path.json": {"context": "...", "date": "2026-11-20"}}
    """
    overrides = metadata_overrides or {}
    documents = []
    for path in list_mock_files():
        extra = overrides.get(os.path.basename(path), {})
        documents += split_work_item_file(path, context=extra.get("context"), date_=extra.get("date"))
    return HybridIndex(documents, embeddings or get_embeddings())


@lru_cache(maxsize=1)
def get_index() -> HybridIndex:
    """Índice compartido del proceso (se construye una sola vez)."""
    return build_index()


def _fragment(doc: Document) -> dict[str, Any]:
    keys = ("source", "source_file", "context", "date", "chunk",
            "score", "score_embeddings", "score_bm25", "score_recency")
    return {"text": doc.page_content, **{key: doc.metadata[key] for key in keys}}


def search_meet_transcript_rag(wi_id: str, query: str) -> str:
    """Fragmentos relevantes de la transcripción de Meet del WI (JSON), con sus puntajes.

    Si la transcripción falta o está corrupta, lo indica en `warning` y no devuelve fragmentos.
    """
    wi = load_work_item(wi_id)
    if wi is None:
        return json.dumps(NOT_FOUND)
    result: dict[str, Any] = {"wi_id": wi.wi_id, "query": query, "fragments": []}
    meet_problems = [p for p in wi.missing_or_corrupt_sources() if p.startswith("MEET_PO_DEMO")]
    if meet_problems:
        result["warning"] = meet_problems[0]
    else:
        hits = get_index().search(query, where={"wi_id": wi.wi_id, "source": "MEET_PO_DEMO"})
        result["fragments"] = [_fragment(d) for d in hits]
    return json.dumps(result, ensure_ascii=False, indent=2)


def search_work_item_rag(wi_id: str, query: str) -> str:
    """Fragmentos relevantes de todas las fuentes válidas del WI (JSON), con advertencias."""
    wi = load_work_item(wi_id)
    if wi is None:
        return json.dumps(NOT_FOUND)
    hits = get_index().search(query, where={"wi_id": wi.wi_id, "corrupt": False})
    return json.dumps(
        {
            "wi_id": wi.wi_id,
            "query": query,
            "warnings": wi.missing_or_corrupt_sources(),
            "fragments": [_fragment(d) for d in hits],
        },
        ensure_ascii=False,
        indent=2,
    )
