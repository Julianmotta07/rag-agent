from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. Cargar vectorstore existente
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="./vectorstore",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 2. Definir el LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3. Prompt personalizado
prompt = PromptTemplate(
    template="""Usa el siguiente contexto para responder la pregunta.
Si no sabes la respuesta, di "No tengo información suficiente".

Contexto:
{context}

Pregunta: {question}

Respuesta:""",
    input_variables=["context", "question"]
)

# 4. Cadena RAG (sintaxis LCEL moderna)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def ask(question: str):
    result = qa_chain.invoke(question)
    print(f"\nPregunta: {question}")
    print(f"Respuesta: {result}")
    return result

if __name__ == "__main__":
    ask("¿De qué trata el documento?")