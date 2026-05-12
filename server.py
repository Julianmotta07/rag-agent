from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from agent import build_agent
from ingest import run_ingest


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando agente RAG con MCP Context7...")
    app.state.agent = await build_agent()
    print("Agente listo.")
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
    result = await app.state.agent.ainvoke({"messages": [("user", req.question)]})
    answer = result["messages"][-1].content
    return ChatResponse(answer=answer)


@app.post("/ingest")
def ingest():
    total_chunks = run_ingest()
    return {"status": "ok", "chunks_generados": total_chunks}
