# 🤖 RAG Agent — LangChain + LangSmith + RAGAS

Agente de preguntas y respuestas basado en **Retrieval-Augmented Generation (RAG)** que indexa documentos locales en una base de datos vectorial (ChromaDB), responde preguntas con GPT-4o-mini, se monitorea en tiempo real con LangSmith y se evalúa con métricas de calidad usando RAGAS.

---

## 📐 Arquitectura general

```
data/docs/        ← Documentos fuente (.txt)
     ↓
ingest.py         ← Carga, divide en chunks y genera embeddings
     ↓
vectorstore/      ← ChromaDB guarda los vectores localmente
     ↓
agent.py          ← RAG chain: recupera contexto + genera respuesta con GPT-4o-mini
     ↓
evaluate.py       ← RAGAS evalúa la calidad del sistema con métricas
     ↑
LangSmith         ← Monitorea cada traza automáticamente
```

---

## 📁 Estructura del proyecto

```
rag_agent/
├── agent.py            # Cadena RAG principal
├── ingest.py           # Ingesta de documentos al vectorstore
├── evaluate.py         # Evaluación con RAGAS
├── requirements.txt    # Dependencias
├── data/
    └── docs/           # Documentos .txt
```

---

## ⚙️ Requisitos previos

- Python 3.10 o superior
- Cuenta en [OpenAI](https://platform.openai.com/) con API key activa
- Cuenta en [LangSmith](https://smith.langchain.com/) con API key (gratuita)

---

## 🚀 Instalación paso a paso

### 1. Clonar el repositorio

```bash
git clone https://github.com/Julianmotta07/rag-agent.git
cd rag-agent
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Crear el archivo `.env`

En la raíz del proyecto crea un archivo llamado exactamente `.env` y pega esto adentro con tus propias claves:

```env
OPENAI_API_KEY=clave-de-openai
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=clave-de-langsmith
LANGCHAIN_PROJECT=rag-agent-demo
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### 5. Indexar los documentos (crear el vectorstore)

```bash
python ingest.py
```

Salida esperada:
```
Chunks generados: 8
Vectorstore creado y guardado.
```

Esto crea la carpeta `vectorstore/` con la base de datos ChromaDB local. Solo necesitas correrlo una vez, o cada vez que cambies/agregues documentos en `data/docs/`.

### 6. Hacer preguntas al agente

```bash
python agent.py
```

### 7. Evaluar la calidad del sistema

```bash
python evaluate.py
```

Imprime un DataFrame con las métricas RAGAS para cada pregunta de prueba.

---

## 🔍 Descripción detallada de cada módulo

---

### `ingest.py` — Ingesta y vectorización

Este script prepara la base de conocimiento del agente. Se corre **una vez** (o cuando se actualicen los documentos).

**Flujo interno:**

1. `DirectoryLoader` recorre `data/docs/` buscando archivos `.txt`
2. `RecursiveCharacterTextSplitter` divide cada documento en chunks de **500 caracteres** con **50 de overlap** (el overlap evita que el contexto se corte en mitad de una idea)
3. `OpenAIEmbeddings` convierte cada chunk en un vector numérico usando el modelo `text-embedding-3-small`
4. `Chroma.from_documents` guarda todos los vectores en `./vectorstore/` en disco

**Para agregar tus propios documentos:** simplemente copia archivos `.txt` en `data/docs/` y vuelve a correr `python ingest.py`. Si quieres regenerar desde cero, borra la carpeta `vectorstore/` primero.

---

### `agent.py` — Cadena RAG

Implementa la cadena de recuperación y generación usando **LCEL** (LangChain Expression Language), que encadena componentes con el operador `|`.

**Flujo de una pregunta:**

```
pregunta
  → retriever busca los 4 chunks más similares en ChromaDB
  → format_docs los une en un solo bloque de texto (contexto)
  → PromptTemplate arma el prompt con contexto + pregunta
  → ChatOpenAI (GPT-4o-mini) genera la respuesta
  → StrOutputParser extrae el texto limpio
```

**Parámetros clave:**

| Parámetro | Valor | Por qué |
|---|---|---|
| `k=4` en el retriever | Recupera 4 chunks | Balance entre contexto suficiente y no exceder el prompt |
| `temperature=0` | Respuestas determinísticas | Reproducibilidad en evaluación |
| Modelo embeddings | `text-embedding-3-small` | Eficiente y económico para RAG |
| Modelo LLM | `gpt-4o-mini` | Rápido y de bajo costo |

El prompt instruye al modelo a responder solo con el contexto dado y decir "No tengo información suficiente" si no puede responder, evitando alucinaciones.

**Función principal:**

```python
from agent import ask
resultado = ask("¿Qué es ChromaDB?")
```

También puedes importar `qa_chain` y `retriever` directamente si los necesitas por separado (como hace `evaluate.py`).

---

### `evaluate.py` — Evaluación con RAGAS

Evalúa la calidad del sistema RAG de forma automatizada usando **RAGAS**, un framework diseñado específicamente para esto.

**Cómo funciona:**

1. Define 3 preguntas de prueba, cada una con una respuesta esperada (`ground_truth`)
2. Para cada pregunta: invoca el agente para obtener la respuesta, y el retriever para obtener los chunks recuperados
3. Arma un `Dataset` de HuggingFace con los campos `question`, `answer`, `contexts`, `ground_truth`
4. Llama a `evaluate()` con las 4 métricas instanciadas

**Las 4 métricas:**

| Métrica | Qué mide | Rango |
|---|---|---|
| `Faithfulness` | ¿La respuesta se basa en el contexto recuperado o alucina? | 0 a 1 (1 = sin alucinaciones) |
| `AnswerRelevancy` | ¿La respuesta es pertinente a la pregunta? | 0 a 1 (1 = muy relevante) |
| `ContextPrecision` | ¿Los chunks recuperados son útiles para responder? | 0 a 1 (1 = contexto muy preciso) |
| `ContextRecall` | ¿El contexto cubre lo que dice el ground truth? | 0 a 1 (1 = cobertura total) |

**Salida esperada:** una tabla con el score de cada métrica por pregunta, más el promedio general.

> Las métricas se calculan haciendo llamadas adicionales al LLM, así que consumen tokens de OpenAI.

---

### `data/docs/` — Documentos fuente

Carpeta donde viven los documentos que el agente usará como fuente de conocimiento. El archivo `demo.txt` incluido contiene información sobre LangChain, RAG, ChromaDB, RAGAS y LangSmith.

Puedes reemplazarlo o agregar cualquier `.txt` con el contenido que quieras que el agente conozca.

---

## 📊 Monitoreo con LangSmith

Con `LANGCHAIN_TRACING_V2=true` en el `.env`, **cada llamada al agente queda registrada automáticamente** en tu cuenta de LangSmith, sin modificar ninguna línea de código.

En [smith.langchain.com](https://smith.langchain.com) → proyecto `rag-agent-demo` verás:

- Traza completa de cada invocación (cada nodo de la cadena)
- Latencia de cada paso: retrieval, construcción del prompt, llamada al LLM
- Tokens consumidos y costo estimado
- Input y output de cada componente
- Historial de todas las ejecuciones

Cada persona que corra el proyecto con **su propia** `LANGCHAIN_API_KEY` verá las trazas en **su propia cuenta** de LangSmith. Las trazas no se mezclan entre usuarios.

---

## 🧩 Dependencias

| Librería | Uso |
|---|---|
| `langchain` | Framework principal de orquestación |
| `langchain-openai` | Embeddings y LLM de OpenAI |
| `langchain-community` | DirectoryLoader para cargar documentos |
| `langchain-chroma` | Integración LangChain con ChromaDB |
| `chromadb` | Base de datos vectorial local |
| `langsmith` | SDK de monitoreo (el tracing se activa con la env var) |
| `ragas` | Evaluación del sistema RAG con métricas |
| `datasets` | Formato de dataset compatible con RAGAS |
| `python-dotenv` | Carga de variables desde `.env` |
| `tiktoken` | Conteo de tokens (requerido por langchain-openai) |

---