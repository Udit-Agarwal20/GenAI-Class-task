from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import re
import threading
import time
from typing import Any, AsyncGenerator, List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from index_langchain import COLLECTION, DB_PATH, NomicPrefixedEmbeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("proofline-server")

DATA_DIR = Path("data").resolve()
CHUNKS_FILE = Path("langchain_chunks.json")
DEFAULT_MODEL = "gemma4:e2b"
CANDIDATE_MODEL = "gemma4:e4b-it-q4_K_M"
MODEL_NAME = DEFAULT_MODEL
EMBED_MODEL_NAME = "nomic-embed-text"

# Hybrid Retrieval Settings
DENSE_CANDIDATES = 8
BM25_CANDIDATES = 8
FINAL_TOP_K = 5
RRF_K = 60

# Model Parameter Budgets
CONTEXT_WINDOW = 8192
DEFAULT_TEMPERATURE = 0.15
KEEP_ALIVE = "30m"

BUDGET_MAP = {
    "brief": 768,
    "standard": 1024,
    "detailed": 1536,
}

SYSTEM_PROMPT = """You are Proofline AI, an elite document intelligence and technical explanation assistant.
Your answers must be grounded firmly and factually in the provided PDF passages.
Every factual claim must cite its exact evidence using the source markers provided: [S1], [S2], [S3], etc.

CORE PRINCIPLES:
1. Technical Depth & Clarity: Provide well-explained, complete, and insightful answers. Do not default to terse summaries unless brief mode is requested.
2. Structure for Explanations:
   When answering questions that ask to "explain", "how", "why", "compare", or "teach":
   - Direct Answer: Begin with a clear, direct answer to the core question.
   - Core Concepts: Explain the underlying mechanism or concept and why it matters.
   - Step-by-Step Breakdown: Walk through algorithms, traversal orders, or multi-step processes sequentially.
   - Examples & Code: Include concrete code examples (e.g., C++), tree diagrams, or analogies when present in the evidence.
   - Trade-offs & Limitations: Discuss time/space complexities, edge cases, prerequisites, or common mistakes noted in the documents.
   - Key Takeaway: Conclude with a practical summary or rule of thumb.
3. Factuality & Anti-Hallucination:
   - Use only information verified in the retrieved evidence.
   - Do not invent facts, complexity guarantees, or code that is not supported by the passages.
   - If the retrieved passages do not contain enough evidence to answer a question or sub-question, explicitly state: "The provided documents do not contain sufficient evidence to answer [specific detail]."
4. Citations: Cite source labels (e.g., [S1], [S2]) immediately following the factual claim or statement."""

MODE_INSTRUCTIONS = {
    "brief": (
        "DEPTH INSTRUCTION: Brief Mode.\n"
        "Provide a concise, direct answer in 1-2 focused paragraphs or bullet points. "
        "Highlight the primary conclusion and essential facts with citations [S1], [S2]. Avoid unnecessary elaboration."
    ),
    "standard": (
        "DEPTH INSTRUCTION: Standard Mode.\n"
        "Provide a well-structured, thorough answer covering the key concepts, mechanisms, and practical context. "
        "Use clear headings, explain why the concept matters, break down multi-step logic, and cite all claims with [S1], [S2]."
    ),
    "detailed": (
        "DEPTH INSTRUCTION: Detailed Mode.\n"
        "Provide an in-depth, exhaustive technical explanation drawing on all relevant details from the provided evidence. "
        "Organize the response with detailed headings, comprehensive conceptual background, step-by-step algorithmic mechanics, "
        "code implementations or templates if available in the text, time and space complexity analysis, trade-offs, and common pitfalls. "
        "Ground every claim meticulously with citations like [S1], [S2]. If any aspect is not covered in the text, explicitly call it out."
    ),
}

AUDIT_SYSTEM_PROMPT = """You are Proofline AI's evidence auditor.
Independently review the proposed answer against the supplied PDF passages.
Do not use outside knowledge. Do not assume a claim is supported simply because it has a citation.

For each important factual claim, classify it as:
- SUPPORTED: the cited passage directly supports it.
- PARTIALLY SUPPORTED: the passage supports only part of the claim.
- UNSUPPORTED: the evidence does not support the claim.
- MISSING CITATION: the claim has no citation.

Refer to source labels such as [S1] exactly as supplied.
Briefly explain each judgment with a short quote if possible.
This is an automated review, not a guarantee of correctness."""


def reciprocal_rank_fusion(
    dense_docs: List[Document],
    bm25_docs: List[Document],
    top_k: int = 5,
    rrf_k: int = 60,
) -> List[Document]:
    """Combines dense and lexical rankings using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    def get_doc_id(doc: Document) -> str:
        src = str(doc.metadata.get("source", "Unknown"))
        page = str(doc.metadata.get("page", "?"))
        start = str(doc.metadata.get("start_index", ""))
        prefix = doc.page_content.strip()[:60]
        return f"{src}::p{page}::s{start}::{prefix}"

    for rank, doc in enumerate(dense_docs, start=1):
        doc_id = get_doc_id(doc)
        doc_map[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

    for rank, doc in enumerate(bm25_docs, start=1):
        doc_id = get_doc_id(doc)
        doc_map[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

    sorted_doc_ids = sorted(scores.keys(), key=lambda d_id: scores[d_id], reverse=True)
    return [doc_map[d_id] for d_id in sorted_doc_ids[:top_k]]


# Global state holder
class AppState:
    client: Optional[QdrantClient] = None
    vector_store: Optional[QdrantVectorStore] = None
    embeddings: Optional[NomicPrefixedEmbeddings] = None
    bm25_retriever: Optional[BM25Retriever] = None
    models: dict[str, ChatOllama] = {}
    model: Optional[ChatOllama] = None
    lock: threading.Lock = threading.Lock()
    is_warm: bool = False

    def init_bm25(self) -> int:
        if not CHUNKS_FILE.exists():
            logger.warning("No %s file found for BM25 initialization.", CHUNKS_FILE)
            return 0
        try:
            items = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
            docs = [
                Document(
                    page_content=item["text"],
                    metadata={
                        "source": item["source"],
                        "page": item["page"],
                        "start_index": item.get("start_index", 0),
                    },
                )
                for item in items
                if item.get("text", "").strip()
            ]
            if docs:
                with self.lock:
                    self.bm25_retriever = BM25Retriever.from_documents(docs, k=BM25_CANDIDATES)
                logger.info("BM25 retriever initialized with %d document chunks.", len(docs))
                return len(docs)
        except Exception as exc:
            logger.error("Failed to initialize BM25 retriever: %s", exc)
        return 0

    def get_model(self, model_name: str = DEFAULT_MODEL, num_predict: int = 1024, temperature: float = DEFAULT_TEMPERATURE):
        with self.lock:
            if model_name not in self.models:
                logger.info("Instantiating ChatOllama for model=%s (ctx=%d)...", model_name, CONTEXT_WINDOW)
                self.models[model_name] = ChatOllama(
                    model=model_name,
                    temperature=temperature,
                    num_ctx=CONTEXT_WINDOW,
                    num_predict=num_predict,
                    keep_alive=KEEP_ALIVE,
                )
            base_model = self.models[model_name]
        return base_model.bind(options={"num_predict": num_predict, "temperature": temperature})


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Initializing Proofline AI local vector store, BM25 index, and optimized Ollama model...")
    try:
        state.client = QdrantClient(path=DB_PATH)
        state.embeddings = NomicPrefixedEmbeddings()
        state.vector_store = QdrantVectorStore(
            client=state.client,
            collection_name=COLLECTION,
            embedding=state.embeddings,
        )

        # Initialize local BM25 keyword index from existing chunks
        state.init_bm25()

        # Warm up default model
        logger.info("Pre-warming default model (%s) in memory...", DEFAULT_MODEL)
        default_model = state.get_model(DEFAULT_MODEL, num_predict=BUDGET_MAP["standard"], temperature=DEFAULT_TEMPERATURE)
        state.model = state.models[DEFAULT_MODEL]

        t_warm_start = time.perf_counter()
        try:
            _ = state.embeddings.embed_query("warmup")
            _ = state.model.invoke("warmup")
            state.is_warm = True
            logger.info(
                "Local models warmed up successfully in %.2fs.",
                time.perf_counter() - t_warm_start,
            )
        except Exception as warm_exc:
            logger.warning("Warmup notice (non-fatal): %s", warm_exc)

    except Exception as exc:
        logger.error("Error during initialization: %s", exc)
        raise

    yield

    logger.info("Shutting down Proofline AI server...")
    if state.client is not None:
        try:
            state.client.close()
            logger.info("Qdrant client closed.")
        except Exception as exc:
            logger.warning("Error closing Qdrant client: %s", exc)


app = FastAPI(title="Proofline AI Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_documents(documents: List[Document]) -> str:
    sections = []
    for i, doc in enumerate(documents, start=1):
        source = Path(str(doc.metadata.get("source", "Unknown"))).name
        page = doc.metadata.get("page", "?")
        sections.append(
            f"[S{i}]\n"
            f"Source: {source}, page {page}\n"
            f"Passage:\n{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(sections)


def run_audit(question: str, answer: str, context: str) -> str:
    audit_prompt = ChatPromptTemplate.from_messages([
        ("system", AUDIT_SYSTEM_PROMPT),
        (
            "human",
            "Original question:\n{question}\n\nProposed answer:\n{answer}\n\nRetrieved evidence:\n{context}\n\nAudit the answer now.",
        ),
    ])
    audit_chain = audit_prompt | state.model | StrOutputParser()
    return audit_chain.invoke({
        "question": question,
        "answer": answer,
        "context": context,
    })


# Pydantic Schemas
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = Field(default_factory=list)
    depth_mode: Optional[str] = "standard"
    model: Optional[str] = None


class SourceItem(BaseModel):
    label: str
    filename: str
    page: Any
    excerpt: str


class AuditDetails(BaseModel):
    text: str
    status: str
    used_labels: List[str]
    invalid_labels: List[str]


class AuditRequest(BaseModel):
    question: str
    answer: str
    sources: List[SourceItem]


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    audit: Optional[AuditDetails] = None
    timings: Optional[dict] = None


@app.get("/api/health")
def get_health():
    points_count = 0
    qdrant_status = "unavailable"
    if state.client is not None:
        try:
            info = state.client.get_collection(COLLECTION)
            points_count = info.points_count or 0
            qdrant_status = str(info.status)
        except Exception as exc:
            qdrant_status = f"error: {str(exc)}"

    pdf_files = list(DATA_DIR.glob("*.pdf"))
    bm25_count = len(state.bm25_retriever.docs) if (state.bm25_retriever and hasattr(state.bm25_retriever, "docs")) else points_count

    return {
        "status": "ok",
        "service": "Proofline AI",
        "model": DEFAULT_MODEL,
        "available_models": [DEFAULT_MODEL, CANDIDATE_MODEL],
        "depth_modes": ["brief", "standard", "detailed"],
        "hybrid_retrieval": True,
        "bm25_chunks": bm25_count,
        "embedding_model": EMBED_MODEL_NAME,
        "collection": COLLECTION,
        "database_path": DB_PATH,
        "indexed_chunks": points_count,
        "documents_count": len(pdf_files),
        "qdrant_status": qdrant_status,
        "is_warm": state.is_warm,
        "top_k": FINAL_TOP_K,
        "num_ctx": CONTEXT_WINDOW,
        "num_predict_standard": BUDGET_MAP["standard"],
        "num_predict_detailed": BUDGET_MAP["detailed"],
        "num_predict_brief": BUDGET_MAP["brief"],
        "local_only": True,
    }


def format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


@app.get("/api/documents")
def list_documents():
    if not DATA_DIR.exists():
        return {"documents": [], "total": 0}

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    docs_metadata = []

    indexed_sources = set()
    if CHUNKS_FILE.exists():
        try:
            items = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
            for it in items:
                indexed_sources.add(it.get("source"))
        except Exception:
            pass

    for pdf in pdf_files:
        stat = pdf.stat()
        docs_metadata.append({
            "filename": pdf.name,
            "size_bytes": stat.st_size,
            "size_human": format_file_size(stat.st_size),
            "modified_timestamp": stat.st_mtime,
            "is_indexed": (pdf.name in indexed_sources) if indexed_sources else True,
        })

    return {
        "documents": docs_metadata,
        "total": len(docs_metadata),
    }


@app.get("/api/pdfs/{filename}")
def serve_pdf(filename: str):
    clean_name = Path(filename).name
    if not clean_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files can be served")

    target_path = (DATA_DIR / clean_name).resolve()
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="Requested PDF document not found")

    try:
        target_path.relative_to(DATA_DIR)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied: invalid file path")

    return FileResponse(
        path=target_path,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{clean_name}"',
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    raw_filename = file.filename or "uploaded.pdf"
    clean_name = Path(raw_filename).name

    if not clean_name.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF (.pdf) documents are accepted.",
        )

    safe_name = re.sub(r"[^\w\-.]", "_", clean_name)
    if not safe_name.lower().endswith(".pdf"):
        safe_name += ".pdf"

    save_path = DATA_DIR / safe_name

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        save_path.write_bytes(content)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save PDF: {str(exc)}")

    try:
        loader = PyPDFLoader(str(save_path))
        pages = loader.load()
    except Exception as exc:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(
            status_code=422,
            detail=f"Could not parse PDF content: {str(exc)}",
        )

    valid_pages = [p for p in pages if p.page_content.strip()]
    if not valid_pages:
        raise HTTPException(
            status_code=422,
            detail="PDF contains no extractable text or is password-protected/scanned.",
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(valid_pages)

    if not chunks:
        raise HTTPException(status_code=422, detail="No chunks could be generated from PDF.")

    new_documents = []
    chunk_dicts = []
    for chunk in chunks:
        page_num = int(chunk.metadata.get("page", 0)) + 1
        start_idx = chunk.metadata.get("start_index", 0)
        doc = Document(
            page_content=chunk.page_content,
            metadata={
                "source": safe_name,
                "page": page_num,
                "start_index": start_idx,
            },
        )
        new_documents.append(doc)
        chunk_dicts.append({
            "source": safe_name,
            "page": page_num,
            "start_index": start_idx,
            "text": chunk.page_content,
        })

    with state.lock:
        if state.vector_store is None:
            raise HTTPException(status_code=503, detail="Vector store is not initialized.")
        try:
            state.vector_store.add_documents(new_documents)
        except Exception as exc:
            logger.error(f"Failed to add documents to Qdrant: {exc}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate embeddings and index document: {str(exc)}",
            )

    try:
        existing_chunks = []
        if CHUNKS_FILE.exists():
            existing_chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
        for cd in chunk_dicts:
            cd["id"] = len(existing_chunks)
            existing_chunks.append(cd)
        CHUNKS_FILE.write_text(
            json.dumps(existing_chunks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # Refresh the lexical BM25 index after upload (Phase 2 Requirement 6)
        state.init_bm25()
    except Exception as exc:
        logger.warning(f"Could not update langchain_chunks.json: {exc}")

    stat = save_path.stat()
    return {
        "status": "success",
        "message": f"Successfully indexed {safe_name}",
        "filename": safe_name,
        "size_bytes": stat.st_size,
        "size_human": format_file_size(stat.st_size),
        "pages_processed": len(valid_pages),
        "chunks_indexed": len(new_documents),
    }


# ==========================================
# STREAMING CHAT ENDPOINT (Server-Sent Events)
# ==========================================
@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: Request, body: ChatRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    depth_mode = (body.depth_mode or "standard").lower().strip()
    if depth_mode not in BUDGET_MAP:
        depth_mode = "standard"
    budget = BUDGET_MAP[depth_mode]
    req_model = body.model or DEFAULT_MODEL

    t_req_start = time.perf_counter()

    # Step 1: Hybrid Retrieval (Dense Qdrant + Lexical BM25 + Reciprocal Rank Fusion)
    t_ret_start = time.perf_counter()
    with state.lock:
        if state.vector_store is None:
            raise HTTPException(status_code=503, detail="AI engine is not initialized.")
        dense_retriever = state.vector_store.as_retriever(search_kwargs={"k": DENSE_CANDIDATES})
        try:
            dense_docs = dense_retriever.invoke(question)
        except Exception as exc:
            logger.error("Dense retrieval error: %s", exc)
            dense_docs = []

        bm25_docs = []
        if state.bm25_retriever is not None:
            try:
                bm25_docs = state.bm25_retriever.invoke(question)
            except Exception as exc:
                logger.error("BM25 retrieval error: %s", exc)

    # Fuse rankings into top diverse passages (5 chunks)
    documents = reciprocal_rank_fusion(
        dense_docs=dense_docs,
        bm25_docs=bm25_docs,
        top_k=FINAL_TOP_K,
        rrf_k=RRF_K,
    )
    t_ret_end = time.perf_counter()
    retrieval_ms = (t_ret_end - t_ret_start) * 1000

    # Build sources payload
    sources_data = []
    for i, doc in enumerate(documents, start=1):
        source_name = Path(str(doc.metadata.get("source", "Unknown"))).name
        page = doc.metadata.get("page", 1)
        sources_data.append({
            "label": f"S{i}",
            "filename": source_name,
            "page": page,
            "excerpt": doc.page_content.strip(),
        })

    context = format_documents(documents) if documents else "No matching passages found in document library."

    # Build adaptive prompt with trimmed history (last 2 turns max)
    system_text = SYSTEM_PROMPT + "\n\n" + MODE_INSTRUCTIONS[depth_mode]
    messages = [("system", system_text)]
    if body.history:
        for prev in body.history[-2:]:
            if prev.role == "user" and prev.content.strip():
                messages.append(("human", prev.content.strip()))
            elif prev.role == "assistant" and prev.content.strip():
                messages.append(("ai", prev.content.strip()))

    messages.append((
        "human",
        "Question: {question}\n\nRetrieved PDF passages:\n{context}\n\n"
        "Provide a complete, insightful, and well-structured answer strictly grounded in the passages above. "
        "Cite every factual claim with its corresponding source label like [S1], [S2] etc.",
    ))

    prompt = ChatPromptTemplate.from_messages(messages)
    formatted_prompt = prompt.format_prompt(question=question, context=context)

    # Acquire model instance bound with the specific depth budget and temperature
    model_instance = state.get_model(
        model_name=req_model,
        num_predict=budget,
        temperature=DEFAULT_TEMPERATURE,
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        # 1. Immediately emit retrieved sources so UI renders citations within ~100ms
        yield f"event: sources\ndata: {json.dumps({'sources': sources_data})}\n\n"

        if not documents:
            empty_msg = "I could not find any relevant passages in your document library to answer this question. Please ensure relevant documents are indexed."
            yield f"event: token\ndata: {json.dumps({'delta': empty_msg})}\n\n"
            yield f"event: done\ndata: {json.dumps({'used_labels': [], 'invalid_labels': []})}\n\n"
            return

        t_gen_start = time.perf_counter()
        first_token_time = None
        full_text_chunks = []

        try:
            # Stream directly from local model asynchronously
            async for chunk in model_instance.astream(formatted_prompt.to_messages()):
                if await request.is_disconnected():
                    logger.info("Client disconnected from chat stream.")
                    break

                delta = chunk.content
                if delta:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    full_text_chunks.append(delta)
                    yield f"event: token\ndata: {json.dumps({'delta': delta})}\n\n"

        except Exception as exc:
            logger.error("Streaming generation error: %s", exc)
            yield f"event: error\ndata: {json.dumps({'error': f'Generation interrupted: {str(exc)}'})}\n\n"
            return

        t_gen_end = time.perf_counter()
        full_answer = "".join(full_text_chunks)

        ttft_ms = ((first_token_time - t_gen_start) * 1000) if first_token_time else 0
        gen_ms = (t_gen_end - t_gen_start) * 1000
        total_ms = (t_gen_end - t_req_start) * 1000
        token_count = len(full_text_chunks)
        tps = (token_count / (gen_ms / 1000)) if gen_ms > 0 else 0

        # Backend performance metrics logging
        logger.info(
            "[PERF] mode=%s model=%s retrieval=%.1fms ttft=%.1fms gen=%.1fms total=%.1fms chunks=%d tokens=%d speed=%.1ftok/s",
            depth_mode,
            req_model,
            retrieval_ms,
            ttft_ms,
            gen_ms,
            total_ms,
            len(documents),
            token_count,
            tps,
        )

        # Citation validation against the actual retrieved documents provided to the model
        used_numbers = set(re.findall(r"\[S(\d+)\]", full_answer))
        valid_numbers = {str(i) for i in range(1, len(documents) + 1)}
        invalid_numbers = sorted(used_numbers - valid_numbers)
        used_labels = [f"S{n}" for n in sorted(used_numbers)]
        invalid_labels = [f"S{n}" for n in invalid_numbers]

        done_payload = {
            "used_labels": used_labels,
            "invalid_labels": invalid_labels,
            "depth_mode": depth_mode,
            "model": req_model,
            "metrics": {
                "retrieval_ms": round(retrieval_ms, 1),
                "ttft_ms": round(ttft_ms, 1),
                "generation_ms": round(gen_ms, 1),
                "total_ms": round(total_ms, 1),
                "token_count": token_count,
                "tokens_per_sec": round(tps, 1),
            },
        }
        yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ==========================================
# INDEPENDENT EVIDENCE AUDIT ENDPOINT
# ==========================================
@app.post("/api/chat/audit", response_model=AuditDetails)
def audit_endpoint(request: AuditRequest):
    """User-triggered or asynchronous evidence audit. Does NOT block response generation."""
    question = request.question.strip()
    answer = request.answer.strip()
    if not question or not answer:
        raise HTTPException(status_code=400, detail="Question and answer required for audit.")

    # Reconstruct context from provided sources
    context_sections = []
    for s in request.sources:
        context_sections.append(
            f"[{s.label}]\nSource: {s.filename}, page {s.page}\nPassage:\n{s.excerpt}"
        )
    context = "\n\n---\n\n".join(context_sections)

    used_numbers = set(re.findall(r"\[S(\d+)\]", answer))
    valid_numbers = {str(i) for i in range(1, len(request.sources) + 1)}
    invalid_numbers = sorted(used_numbers - valid_numbers)
    used_labels = [f"S{n}" for n in sorted(used_numbers)]
    invalid_labels = [f"S{n}" for n in invalid_numbers]

    t0 = time.perf_counter()
    try:
        audit_text = run_audit(question, answer, context)
        audit_status = "completed"
    except Exception as exc:
        logger.warning(f"Audit generation failed: {exc}")
        audit_text = f"Audit failed: {str(exc)}"
        audit_status = "failed"
    t1 = time.perf_counter()

    logger.info("[PERF] audit_time=%.1fms status=%s", (t1 - t0) * 1000, audit_status)

    return AuditDetails(
        text=audit_text,
        status=audit_status,
        used_labels=used_labels,
        invalid_labels=invalid_labels,
    )


# Non-streaming fallback endpoint
@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    t0 = time.perf_counter()
    with state.lock:
        if state.vector_store is None or state.model is None:
            raise HTTPException(status_code=503, detail="AI engine is not fully initialized.")
        retriever = state.vector_store.as_retriever(search_kwargs={"k": TOP_K})
        try:
            documents = retriever.invoke(question)
        except Exception as exc:
            logger.error(f"Retrieval error: {exc}")
            raise HTTPException(status_code=500, detail="Retrieval failed.")

    t_ret = time.perf_counter()
    if not documents:
        return ChatResponse(
            answer="I could not find relevant passages in your document library to answer this question.",
            sources=[],
            audit=None,
        )

    context = format_documents(documents)
    messages = [("system", SYSTEM_PROMPT)]
    if request.history:
        for prev in request.history[-2:]:
            if prev.role == "user" and prev.content.strip():
                messages.append(("human", prev.content.strip()))
            elif prev.role == "assistant" and prev.content.strip():
                messages.append(("ai", prev.content.strip()))

    messages.append((
        "human",
        "Question: {question}\n\nRetrieved PDF passages:\n{context}\n\nGive a clear answer citing [S1], [S2] etc.",
    ))

    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | state.model | StrOutputParser()

    try:
        answer = chain.invoke({"question": question, "context": context})
    except Exception as exc:
        logger.error(f"Generation error: {exc}")
        raise HTTPException(status_code=500, detail="Generation failed.")

    t_gen = time.perf_counter()

    sources: List[SourceItem] = []
    for i, doc in enumerate(documents, start=1):
        source_name = Path(str(doc.metadata.get("source", "Unknown"))).name
        page = doc.metadata.get("page", 1)
        sources.append(
            SourceItem(
                label=f"S{i}",
                filename=source_name,
                page=page,
                excerpt=doc.page_content.strip(),
            )
        )

    used_numbers = set(re.findall(r"\[S(\d+)\]", answer))
    valid_numbers = {str(i) for i in range(1, len(documents) + 1)}
    invalid_numbers = sorted(used_numbers - valid_numbers)
    used_labels = [f"S{n}" for n in sorted(used_numbers)]
    invalid_labels = [f"S{n}" for n in invalid_numbers]

    timings = {
        "retrieval_ms": round((t_ret - t0) * 1000, 1),
        "generation_ms": round((t_gen - t_ret) * 1000, 1),
        "total_ms": round((t_gen - t0) * 1000, 1),
    }

    return ChatResponse(
        answer=answer,
        sources=sources,
        audit=None,  # Audit is now decoupled and user-triggered on demand!
        timings=timings,
    )


# Serve frontend build if present
DIST_DIR = Path("frontend/dist").resolve()
if DIST_DIR.exists() and (DIST_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
