import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

SYSTEM_PROMPT = """Eres un experto en LangChain y el ecosistema LangChain (LangGraph, LangSmith, LCEL, etc.).

Tienes acceso a dos herramientas:
1. retrieve_langchain_docs: busca en la documentacion local de LangChain. USALA PRIMERO para cualquier pregunta.
2. Herramientas de Context7 (resolve_library_id, get_library_docs): consulta documentacion actualizada en linea. Usalas solo si la documentacion local no responde la pregunta o si se necesita informacion sobre una version especifica.

Instrucciones:
- Responde siempre en el idioma en que te hagan la pregunta.
- Si ningun tool responde la pregunta, di "No tengo informacion suficiente sobre ese tema".
- No inventes informacion: basa tus respuestas exclusivamente en lo que los tools te devuelven.
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
    docs = retriever.invoke(query)
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
    mcp_tools = await mcp_client.get_tools()
    tools = [retrieve_langchain_docs] + mcp_tools
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    system_message = SystemMessage(content=SYSTEM_PROMPT)
    return create_react_agent(llm, tools, prompt=system_message)


async def ask_async(question: str) -> str:
    agent = await build_agent()
    result = await agent.ainvoke({"messages": [("user", question)]})
    answer = result["messages"][-1].content
    print(f"\nPregunta: {question}")
    print(f"Respuesta: {answer}")
    return answer


def ask(question: str) -> str:
    return asyncio.run(ask_async(question))


if __name__ == "__main__":
    ask("¿Que es LCEL en LangChain y para que sirve?")
