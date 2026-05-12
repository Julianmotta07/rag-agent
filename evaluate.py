import asyncio
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import ToolMessage
from agent import build_agent

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

test_cases = [
    {
        "question": "¿Que es LCEL (LangChain Expression Language) y para que sirve?",
        "ground_truth": "LCEL es una sintaxis declarativa de LangChain que permite componer cadenas de componentes usando el operador pipe (|). Facilita construir pipelines como retriever | prompt | llm | parser de forma legible y modular.",
    },
    {
        "question": "¿Como funciona un VectorStoreRetriever en LangChain?",
        "ground_truth": "Un VectorStoreRetriever recupera documentos relevantes desde una base de datos vectorial buscando los k vectores mas similares a la consulta usando busqueda por similitud semantica (cosine similarity o similares).",
    },
    {
        "question": "¿Como se define un tool personalizado en LangChain usando el decorador @tool?",
        "ground_truth": "Se usa el decorador @tool sobre una funcion Python. El nombre de la funcion se convierte en el nombre del tool y el docstring se usa como descripcion para que el LLM sepa cuando invocarlo.",
    },
]


def extract_rag_context(messages: list) -> list[str]:
    return [
        msg.content
        for msg in messages
        if isinstance(msg, ToolMessage) and msg.name == "retrieve_langchain_docs"
    ]


async def run_evaluation():
    agent = await build_agent()
    data: dict[str, list] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    print("Generando respuestas y contextos...")
    for case in test_cases:
        print(f"  Procesando: {case['question']}")
        result = await agent.ainvoke({"messages": [("user", case["question"])]})
        answer = result["messages"][-1].content
        contexts = extract_rag_context(result["messages"])

        # Si el agente no uso el RAG local (uso Context7 u otro tool), contexto sera vacio.
        # RAGAS puede manejar contextos vacios pero los scores seran mas bajos.
        if not contexts:
            contexts = ["(el agente no uso documentacion local para esta pregunta)"]

        data["question"].append(case["question"])
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        data["ground_truth"].append(case["ground_truth"])

    dataset = Dataset.from_dict(data)

    print("\nEjecutando evaluacion RAGAS...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    print("\n=== RESULTADOS DE EVALUACION RAGAS ===")
    print(result)
    print("\n=== RESULTADOS DETALLADOS ===")
    print(result.to_pandas())


if __name__ == "__main__":
    asyncio.run(run_evaluation())
