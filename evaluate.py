from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from agent import qa_chain, retriever
import os

load_dotenv()

# Configurar LLM y embeddings para RAGAS
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Preguntas de prueba con ground truth
test_cases = [
    {
        "question": "¿Qué es RAG?",
        "ground_truth": "RAG es una técnica que combina recuperación de información con generación de texto, buscando documentos relevantes para generar respuestas."
    },
    {
        "question": "¿Para qué sirve LangSmith?",
        "ground_truth": "LangSmith es una plataforma de monitoreo para aplicaciones LangChain que registra trazas, latencia y tokens usados."
    },
    {
        "question": "¿Qué mide RAGAS?",
        "ground_truth": "RAGAS mide faithfulness, answer relevancy y context precision para evaluar sistemas RAG."
    },
]

# Generar respuestas y contextos
data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

print("Generando respuestas y contextos...")
for case in test_cases:
    print(f"Procesando: {case['question']}")
    answer = qa_chain.invoke(case["question"])
    docs = retriever.invoke(case["question"])
    data["question"].append(case["question"])
    data["answer"].append(answer)
    data["contexts"].append([d.page_content for d in docs])
    data["ground_truth"].append(case["ground_truth"])

# Crear dataset y evaluar
dataset = Dataset.from_dict(data)

print("\nEjecutando evaluación RAGAS...")
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=llm,
    embeddings=embeddings
)

print("\n=== RESULTADOS DE EVALUACIÓN RAGAS ===")
print(result)
print("\n=== RESULTADOS DETALLADOS ===")
print(result.to_pandas())