import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ingestion.pipeline import run_pipeline
from agent.agent import answer

app = FastAPI(
    title="Oculis API",
    description="Multimodal Agentic RAG with hallucination guardrails",
    version="1.0.0"
)

# CORS lets your HTML frontend (running on a different port or
# opened as a local file) talk to this API without being blocked
# by the browser's same-origin security policy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "null",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer:     str
    confidence: float
    flagged:    bool
    warning:    str

class UploadResponse(BaseModel):
    filename:       str
    chunks_stored:  int
    message:        str

class HealthResponse(BaseModel):   
    status:         str
    chunks_in_db:   int
    message:        str


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    # Health check — visit this to confirm the API is running.
    return {"status": "Oculis API is running"}


@app.get("/health", response_model=HealthResponse)
def health():
    # Returns ChromaDB status and current chunk count.
    # The frontend calls this on page load to show the pill in the header.
    #
    try:
        import chromadb
        from config import CHROMA_PATH, COLLECTION_NAME
        db    = chromadb.PersistentClient(path=CHROMA_PATH)
        col   = db.get_or_create_collection(COLLECTION_NAME)
        count = col.count()
        return HealthResponse(
            status="healthy",
            chunks_in_db=count,
            message=f"ChromaDB reachable — {count} chunks stored"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )
    

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    skip_vlm: bool = False
):
    """
    Upload a PDF and ingest it into ChromaDB.

    Why save to a temp file first?
    Set skip_vlm=true in the query string to skip LLaVA captioning:
        POST /upload?skip_vlm=true
    Requires Ollama running locally with LLaVA pulled when skip_vlm=false.
    """
    # Validate file type before doing any work
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    # Write the uploaded bytes to a temp file on disk
    tmp_dir  = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Run the full ingestion pipeline
        chunks_stored = run_pipeline(tmp_path, skip_vlm=skip_vlm)

    except Exception as e:
        error_msg = str(e)
        # Give a helpful hint if Ollama isn't running
        if "ollama" in error_msg.lower() or "connection" in error_msg.lower():
            error_msg = (
                f"LLaVA/Ollama error: {error_msg}. "
                "Make sure Ollama is running ('ollama serve') and "
                "LLaVA is pulled ('ollama pull llava'). "
                "Or upload with ?skip_vlm=true to skip image captioning."
            )
        raise HTTPException(status_code=500, detail= error_msg)

    finally:
        # Always clean up the temp file — even if the pipeline crashed
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return UploadResponse(
        filename=file.filename,
        chunks_stored=chunks_stored,
        message=f"Successfully ingested {chunks_stored} chunks from {file.filename}"
    )


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    # Ask a question against the ingested documents.

    # Flow:
    #     1. Agent searches ChromaDB via rag_search tool
    #     2. Agent optionally uses web_search or calculate tools
    #     3. Agent produces an answer
    #     4. Guardrails check the answer for hallucinations
    #     5. Returns answer + confidence score + flagged status

    # The confidence score tells the frontend whether to show a warning.
    # flagged=True means confidence < 0.6 — likely hallucination.
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        result = answer(question=request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")
    
    # answer() always returns a dict — but validate the shape anyway
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=500,
            detail="Agent returned unexpected output format."
        )


    return AskResponse(
        answer=result.get("answer",     "No answer produced."),
        confidence=result.get("confidence", 0.0),
        flagged=result.get("flagged",    False),
        warning=result.get("warning",    "")
    )
