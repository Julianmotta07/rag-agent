from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# 1. Cargar documentos desde /data/docs
loader = DirectoryLoader("data/docs", glob="**/*.txt", loader_cls=TextLoader)
docs = loader.load()

# 2. Dividir en chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"Chunks generados: {len(chunks)}")

# 3. Crear embeddings y guardar en ChromaDB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./vectorstore"
)
print("Vectorstore creado y guardado.")