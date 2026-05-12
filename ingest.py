import os
import shutil
import time
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

VECTORSTORE_DIR = "./vectorstore"

# Páginas clave de la documentación oficial de LangChain
LANGCHAIN_DOCS_URLS = [
    # Introducción y conceptos core
    "https://python.langchain.com/docs/introduction/",
    "https://python.langchain.com/docs/concepts/",
    # Modelos
    "https://python.langchain.com/docs/concepts/chat_models/",
    "https://python.langchain.com/docs/concepts/messages/",
    "https://python.langchain.com/docs/concepts/embedding_models/",
    # Prompts y outputs
    "https://python.langchain.com/docs/concepts/prompt_templates/",
    "https://python.langchain.com/docs/concepts/output_parsers/",
    # LCEL
    "https://python.langchain.com/docs/concepts/lcel/",
    # RAG
    "https://python.langchain.com/docs/concepts/rag/",
    "https://python.langchain.com/docs/concepts/vectorstores/",
    "https://python.langchain.com/docs/concepts/retrievers/",
    "https://python.langchain.com/docs/concepts/text_splitters/",
    "https://python.langchain.com/docs/concepts/document_loaders/",
    # Tools y Agentes
    "https://python.langchain.com/docs/concepts/tools/",
    "https://python.langchain.com/docs/concepts/agents/",
    "https://python.langchain.com/docs/concepts/tool_calling/",
    # Tutoriales
    "https://python.langchain.com/docs/tutorials/llm_chain/",
    "https://python.langchain.com/docs/tutorials/chatbot/",
    "https://python.langchain.com/docs/tutorials/rag/",
    "https://python.langchain.com/docs/tutorials/agents/",
    "https://python.langchain.com/docs/tutorials/qa_chat_history/",
    # How-to clave
    "https://python.langchain.com/docs/how_to/tool_calling/",
    "https://python.langchain.com/docs/how_to/custom_tools/",
    "https://python.langchain.com/docs/how_to/vectorstores/",
    "https://python.langchain.com/docs/how_to/qa_sources/",
    "https://python.langchain.com/docs/how_to/streaming/",
    "https://python.langchain.com/docs/how_to/structured_output/",
]


def check_url(url: str, timeout: int = 5) -> bool:
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        return r.status_code < 400
    except Exception:
        return False


def load_langchain_docs() -> list:
    print(f"Cargando {len(LANGCHAIN_DOCS_URLS)} paginas de la documentacion de LangChain...")

    # Filtrar URLs accesibles antes de cargar
    accessible = []
    for url in LANGCHAIN_DOCS_URLS:
        if check_url(url):
            accessible.append(url)
        else:
            print(f"  [SKIP] No accesible: {url}")

    if not accessible:
        print("ERROR: Ninguna URL es accesible. Verifica tu conexion a internet.")
        return []

    print(f"URLs accesibles: {len(accessible)}/{len(LANGCHAIN_DOCS_URLS)}")

    loader = WebBaseLoader(
        web_paths=accessible,
        requests_kwargs={"timeout": 15},
    )
    docs = loader.load()
    print(f"Paginas cargadas: {len(docs)}")
    return docs


def load_local_docs() -> list:
    local_path = "data/docs"
    if not os.path.exists(local_path):
        return []
    loader = DirectoryLoader(local_path, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    if docs:
        print(f"Documentos locales cargados: {len(docs)}")
    return docs


def run_ingest() -> int:
    start = time.time()

    if os.path.exists(VECTORSTORE_DIR):
        shutil.rmtree(VECTORSTORE_DIR)
        print("Vectorstore previo eliminado.")

    docs = load_langchain_docs() + load_local_docs()

    if not docs:
        print("No se encontraron documentos. Abortando.")
        return 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"Chunks generados: {len(chunks)}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR,
    )

    elapsed = round(time.time() - start, 1)
    print(f"Vectorstore guardado en {VECTORSTORE_DIR}. Tiempo total: {elapsed}s")
    return len(chunks)


if __name__ == "__main__":
    run_ingest()
