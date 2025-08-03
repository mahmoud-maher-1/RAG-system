

# RAG-System

A Java-based **Retrieval-Augmented Generation (RAG)** system powered by [LangChain4j](https://github.com/langchain4j/langchain4j). This system combines local LLMs via **Ollama**, **Milvus** as a vector database, and **MiniLM** for embeddings — allowing you to retrieve contextual documents and generate smart answers.

---

## Features

- Load and embed custom documents
- Store and retrieve vectors using Milvus
- Generate context-aware answers using Ollama-powered LLMs
- Built in Java with LangChain4j

---

## Project Structure

```

RAG-system/
├── pom.xml
└── src/main/java/org/ragsys/
├── Main.java
├── DocumentLoader.java
└── RagPipeline.java

````

---

## Requirements

- Java 21
- Maven 3.6+
- Docker (for Milvus and Ollama)
- Ollama (running a local model like `mistral`)
- Milvus (vector DB running locally)

---

##  Setup

### 1. Clone the project

```bash
git clone https://github.com/mahmoud-maher-1/RAG-system.git
cd RAG-system
````

### 2. Start services

**Ollama:**

```bash
ollama run mistral
```

**Milvus:**

```bash
docker run -p 19530:19530 -p 9091:9091 milvusdb/milvus:v2.3.9
```

### 3. Build

```bash
mvn clean install
```

### 4. Run

```bash
mvn exec:java -Dexec.mainClass="org.ragsys.Main"
```

---

## How It Works

| File                  | Description                                                        |
| --------------------- | ------------------------------------------------------------------ |
| `Main.java`           | Entry point — initializes and runs the pipeline                    |
| `DocumentLoader.java` | Reads and splits documents into chunks                             |
| `RagPipeline.java`    | Handles embedding, storing to Milvus, retrieval, and LLM responses |

---

## Input & Output

* **Input**: Plaintext documents
* **Query**: User asks a question
* **Output**: LLM-generated answer with document context

---
