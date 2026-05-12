from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from agent import build_agent, log_trace
from ingest import run_ingest


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando agente RAG con MCP Context7...")
    app.state.agent = await build_agent()
    print("Agente listo.")
    print("[SERVER] Rutas registradas: GET /, GET /health, POST /chat, POST /ingest")
    yield


app = FastAPI(
    title="LangChain RAG Agent",
    description="Agente RAG especializado en LangChain con MCP Context7 y monitoreo LangSmith.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def ui():
    return FileResponse("static/index.html")


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacia.")
    print(f"\n[SERVER /chat] '{req.question}'")
    result = await app.state.agent.ainvoke({"messages": [("user", req.question)]})
    log_trace(result["messages"])
    raw = result["messages"][-1].content
    answer = raw if isinstance(raw, str) else " ".join(
        c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in raw
    )
    return ChatResponse(answer=answer)


@app.post("/ingest")
def ingest():
    total_chunks = run_ingest()
    return {"status": "ok", "chunks_generados": total_chunks}
