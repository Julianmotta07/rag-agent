import os
import shutil
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import SitemapLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

VECTORSTORE_DIR = "./vectorstore"
SITEMAP_URL = "https://python.langchain.com/sitemap.xml"


def load_langchain_docs() -> list:
    print("Cargando documentacion desde sitemap de python.langchain.com...")
    loader = SitemapLoader(
        web_path=SITEMAP_URL,
        filter_urls=[r"https://python\.langchain\.com/docs/(?!.*api_reference).*"],
        continue_on_failure=True,
        requests_per_second=2,
    )
    docs = loader.load()
    print(f"Paginas cargadas desde sitemap: {len(docs)}")
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

    # Borrar vectorstore previo para garantizar idempotencia
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
    print(f"Vectorstore creado y guardado en {VECTORSTORE_DIR}. Tiempo: {elapsed}s")
    return len(chunks)


if __name__ == "__main__":
    run_ingest()
