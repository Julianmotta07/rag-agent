# RAG Agent — LangChain Expert con MCP Context7 + FastAPI + LangSmith + RAGAS

Agente especializado en LangChain que combina un vectorstore local (ChromaDB con docs oficiales de python.langchain.com) con herramientas MCP en tiempo real (Context7), expuesto como API REST con FastAPI y monitoreado con LangSmith. La calidad del sistema se evalúa con RAGAS.

---

## Arquitectura

```
FastAPI (server.py)
       |
       v
LangGraph ReAct Agent (agent.py)
       |
       +--- retrieve_langchain_docs  <-- ChromaDB local (docs de python.langchain.com)
       |
       +--- Context7 MCP tools       <-- Documentación actualizada en línea
              (resolve-library-id,
               get-library-docs)
       |
       v
  GPT-4o-mini sintetiza la respuesta
       |
       v
  LangSmith registra cada traza
```

El agente usa `retrieve_langchain_docs` primero (barato, rápido). Solo cae en Context7 si la doc local no responde o si se pregunta por una versión específica.

---

## Estructura del proyecto

```
rag-agent/
├── agent.py        # Agente LangGraph ReAct con tool RAG + MCP
├── ingest.py       # Ingesta desde sitemap de python.langchain.com
├── evaluate.py     # Evaluación con RAGAS
├── server.py       # API REST con FastAPI
├── requirements.txt
├── data/
│   └── docs/       # Documentos .txt complementarios (opcional)
```

---

## Requisitos previos

- Python 3.10+
- **Node.js con npx** — necesario para lanzar el servidor MCP Context7 (`npx -y @upstash/context7-mcp`)
- Cuenta en [OpenAI](https://platform.openai.com/) con API key y crédito
- Cuenta en [LangSmith](https://smith.langchain.com/) con API key (gratuita)

---

## Instalación

### 1. Clonar e instalar dependencias

```bash
git clone https://github.com/Julianmotta07/rag-agent.git
cd rag-agent
pip install -r requirements.txt
```

### 2. Crear el archivo `.env`

```env
OPENAI_API_KEY=sk-proj-...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=rag-agent-demo
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

Context7 no requiere API key.

### 3. Indexar la documentación de LangChain

```bash
python ingest.py
```

Descarga y vectoriza la documentación oficial de python.langchain.com. Puede tardar varios minutos dependiendo de la velocidad de red. Solo necesitas correrlo una vez, o cuando quieras actualizar la base de conocimiento.

Salida esperada:
```
Cargando documentacion desde sitemap de python.langchain.com...
Paginas cargadas desde sitemap: 350+
Chunks generados: 2000+
Vectorstore creado y guardado. Tiempo: Xs
```

### 4. Levantar el servidor

```bash
uvicorn server:app --reload --port 8000
```

El servidor carga el agente y las tools MCP al arrancar (espera ~10s la primera vez por `npx`).

---

## Uso de la API

### Hacer una pregunta

```bash
# PowerShell
Invoke-RestMethod -Method POST `
  -Uri http://localhost:8000/chat `
  -ContentType "application/json" `
  -Body '{"question": "Que es LCEL en LangChain?"}'

# curl (bash)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Que es LCEL en LangChain?"}'
```

Respuesta:
```json
{"answer": "LCEL (LangChain Expression Language) es..."}
```

### Re-indexar documentos

```bash
curl -X POST http://localhost:8000/ingest
```

### Health check

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### Documentación interactiva

Con el servidor corriendo, abre [http://localhost:8000/docs](http://localhost:8000/docs) para la UI de Swagger.

---

## Uso como script (sin servidor)

```bash
python agent.py
```

---

## Evaluar la calidad con RAGAS

```bash
python evaluate.py
```

Evalúa 3 preguntas sobre LangChain con 4 métricas:

| Métrica | Qué mide |
|---|---|
| Faithfulness | ¿La respuesta se basa en el contexto o alucina? |
| AnswerRelevancy | ¿La respuesta es pertinente a la pregunta? |
| ContextPrecision | ¿Los chunks recuperados son útiles? |
| ContextRecall | ¿El contexto cubre el ground truth? |

---

## Monitoreo con LangSmith

Con `LANGCHAIN_TRACING_V2=true`, cada invocación queda registrada en [smith.langchain.com](https://smith.langchain.com) → proyecto `rag-agent-demo`.

Verás el árbol completo del agente ReAct:
- Nodo `agent`: decisión del LLM sobre qué tool usar
- Nodo `tools`: invocación de `retrieve_langchain_docs` o tools de Context7
- Latencia, tokens y costo por paso

---

## Cómo funciona Context7 (MCP)

Context7 es un servidor MCP que expone documentación actualizada de librerías populares. El agente tiene acceso a dos tools de Context7:

- `resolve-library-id`: busca el ID de una librería por nombre
- `get-library-docs`: descarga la documentación de esa librería

El agente cae en Context7 cuando la pregunta requiere información que no está en el vectorstore local (por ejemplo, features de versiones recientes de LangGraph).

---

## Dependencias clave

| Librería | Rol |
|---|---|
| `langchain` | Framework principal |
| `langchain-openai` | LLM y embeddings de OpenAI |
| `langgraph` | Agente ReAct con tool-calling |
| `langchain-mcp-adapters` | Integración MCP → tools LangChain |
| `langchain-chroma` | Vectorstore local |
| `langsmith` | Monitoreo de trazas |
| `fastapi` + `uvicorn` | Servidor HTTP |
| `ragas` | Evaluación de calidad del RAG |
| `beautifulsoup4` + `lxml` | Parseo del sitemap y HTML |
