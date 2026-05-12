import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

SYSTEM_PROMPT = """Eres un experto en LangChain y el ecosistema LangChain (LangGraph, LangSmith, LCEL, etc.).

Tienes acceso a estas herramientas:
1. retrieve_langchain_docs: busca en documentacion local. USALA SIEMPRE PRIMERO.
2. resolve-library-id: obtiene el ID de una libreria en Context7. Usala como primer paso de Context7.
3. query-docs: descarga documentacion actualizada de Context7 dado un library ID.

REGLA CRITICA DE FALLBACK:
- Llama a retrieve_langchain_docs primero.
- Evalua si el contenido devuelto responde DIRECTAMENTE la pregunta del usuario.
- Si el contenido es vago, generico, o habla de otro tema: DEBES usar Context7 a continuacion.
- Proceso Context7: primero llama a resolve-library-id con el nombre de la libreria, luego llama a query-docs con el ID obtenido.
- Solo di "No tengo informacion suficiente" si AMBAS fuentes (local y Context7) fallaron.

Instrucciones adicionales:
- Responde siempre en el idioma en que te hagan la pregunta.
- No inventes informacion: basa tus respuestas exclusivamente en lo que los tools devuelven.
- Si la pregunta es sobre LangGraph, LangSmith u otro sub-proyecto, usa "langchain" o el nombre exacto en resolve-library-id.
"""

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="./vectorstore",
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


@tool
def retrieve_langchain_docs(query: str) -> str:
    """Busca en la documentacion local de LangChain. Usar PRIMERO para preguntas sobre LangChain."""
    print(f"  [RAG] Buscando en vectorstore: '{query}'")
    docs = retriever.invoke(query)
    print(f"  [RAG] Chunks encontrados: {len(docs)}")
    for i, doc in enumerate(docs):
        src = doc.metadata.get("source", "desconocido")
        print(f"  [RAG] Chunk {i+1} — fuente: {src} | preview: {doc.page_content[:80].strip()}...")
    if not docs:
        return "No se encontro informacion relevante en la documentacion local."
    return "\n\n".join(doc.page_content for doc in docs)


mcp_client = MultiServerMCPClient(
    {
        "context7": {
            "command": "npx",
            "args": ["-y", "@upstash/context7-mcp"],
            "transport": "stdio",
        }
    }
)


async def build_agent():
    print("[AGENT] Cargando tools MCP (Context7 via npx)...")
    try:
        mcp_tools = await mcp_client.get_tools()
        print(f"[AGENT] Tools MCP cargadas ({len(mcp_tools)}): {[t.name for t in mcp_tools]}")
    except Exception as e:
        print(f"[AGENT] ERROR cargando MCP tools: {e}")
        print("[AGENT] El agente continuara solo con el RAG local.")
        mcp_tools = []

    tools = [retrieve_langchain_docs] + mcp_tools
    print(f"[AGENT] Tools totales disponibles: {[t.name for t in tools]}")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    system_message = SystemMessage(content=SYSTEM_PROMPT)
    return create_react_agent(llm, tools, prompt=system_message)


def _preview(content) -> str:
    if isinstance(content, list):
        text = " ".join(
            c.get("text", str(c)) if isinstance(c, dict) else str(c)
            for c in content
        )
    else:
        text = str(content)
    return text[:120].replace("\n", " ")


def log_trace(messages: list):
    print("\n--- TRAZA DEL AGENTE ---")
    for msg in messages:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  [LLM -> TOOL] {tc['name']}  args: {tc['args']}")
            else:
                print(f"  [LLM -> RESPUESTA FINAL] {_preview(msg.content)}...")
        elif isinstance(msg, ToolMessage):
            print(f"  [TOOL RESULT] ({msg.name}) {_preview(msg.content)}...")
    print("--- FIN TRAZA ---\n")


async def ask_async(question: str) -> str:
    agent = await build_agent()
    print(f"\n[AGENT] Pregunta recibida: '{question}'")
    result = await agent.ainvoke({"messages": [("user", question)]})
    log_trace(result["messages"])
    answer = result["messages"][-1].content
    print(f"[AGENT] Respuesta final: {answer[:200]}")
    return answer


def ask(question: str) -> str:
    return asyncio.run(ask_async(question))


if __name__ == "__main__":
    ask("¿Que es LCEL en LangChain y para que sirve?")
