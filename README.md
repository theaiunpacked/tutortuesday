# RAG System with Claude API
### Build a Retrieval-Augmented Generation system from scratch — in under 30 minutes

> **Tutor Tuesday** · [The AI Unpacked](https://www.youtube.com/@TheAIUnpackedWeekly) · by [Raj Jain](https://linkedin.com/in/rajjain)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green?style=flat-square)](https://langchain.com)
[![Claude API](https://img.shields.io/badge/Claude-Sonnet%204-teal?style=flat-square)](https://anthropic.com)
[![FAISS](https://img.shields.io/badge/FAISS-Open%20Source-orange?style=flat-square)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## What This Is

This repo contains the complete code walkthrough from the **Tutor Tuesday** episode on [The AI Unpacked](https://www.youtube.com/@TheAIUnpackedWeekly).

You will build a working RAG system that:
- Ingests any PDF document
- Converts it into a searchable vector index using FAISS
- Answers questions grounded strictly in your document using Claude
- Returns source citations so every answer is auditable

No fine-tuning. No cloud vector DB subscription. No PhD required.

---

## Stack

| Component | Tool | Why |
|---|---|---|
| Language | Python 3.9+ | Universal, beginner-friendly |
| Orchestration | LangChain | Connects all the pieces cleanly |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Free, runs locally, no API cost |
| Vector Store | FAISS | Open source, fast, runs on your machine |
| LLM | Claude API (`claude-sonnet-4-20250514`) | Best-in-class instruction following |
| PDF Loader | PyPDF | Lightweight, no dependencies |

**Everything except the Claude API is 100% open source.** Your documents are embedded locally — they never leave your machine until the final query hits Claude.

---

## Prerequisites

- Python 3.9 or higher
- An [Anthropic API key](https://console.anthropic.com/) (free tier available)
- A PDF document to query against

---

## Quickstart

**1. Clone the repo**

```bash
git clone https://github.com/yourusername/rag-claude.git
cd rag-claude
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Set your API key**

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

Or create a `.env` file:

```
ANTHROPIC_API_KEY=your-api-key-here
```

**4. Add your PDF**

Place any PDF in the project root. Rename it or update `PDF_PATH` in `config.py`.

**5. Run**

```bash
python rag_claude.py
```

Then type any question about your document at the prompt.

---

## Project Structure

```
rag-claude/
│
├── rag_claude.py          # Main script — full pipeline in one file
├── config.py              # Configuration (PDF path, model, chunk size)
├── requirements.txt       # All dependencies
├── sample_policy.pdf      # Sample HR policy PDF to test with
├── .env.example           # Environment variable template
└── README.md
```

---

## How It Works

```
Your Question
      │
      ▼
  Embeddings Model  (HuggingFace MiniLM — runs locally)
      │
      ▼
  FAISS Vector Search  (finds the 4 most relevant chunks)
      │
      ▼
  Prompt Builder  (injects chunks + your question into Claude's context)
      │
      ▼
  Claude API  (reads chunks, generates grounded answer)
      │
      ▼
  Answer + Source Citations
```

**Why RAG instead of fine-tuning?**

| | RAG | Fine-tuning |
|---|---|---|
| Cost | Low | High |
| Setup time | Hours | Days to weeks |
| Update documents | Instant re-index | Full retraining |
| Auditability | Citations included | Black box |
| Data stays private | Yes (with local embeddings) | Depends on provider |

---

## The Code

### `config.py`

```python
PDF_PATH = "sample_policy.pdf"
INDEX_PATH = "rag_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K_RESULTS = 4
```

### `rag_claude.py`

```python
# RAG System using Claude API
# The AI Unpacked — Tutor Tuesday
# github.com/yourusername/rag-claude

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from config import *

# --- STEP 1: Load & Chunk ---
print("Loading document...")
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
chunks = splitter.split_documents(docs)
print(f"  {len(chunks)} chunks created")

# --- STEP 2: Embed & Store ---
print("Building vector index...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(INDEX_PATH)
print("  Vector store saved")

# --- STEP 3: Build the Chain ---
llm = ChatAnthropic(
    model=CLAUDE_MODEL,
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    max_tokens=1024
)

retriever = FAISS.load_local(
    INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True
).as_retriever(search_kwargs={"k": TOP_K_RESULTS})

prompt = PromptTemplate(
    template="""You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have that information in the document."

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# --- STEP 4: Ask ---
print("\nRAG system ready. Type 'quit' to exit.\n")
while True:
    question = input("Ask a question: ").strip()
    if question.lower() in ["quit", "exit", "q"]:
        break
    if not question:
        continue

    result = qa_chain({"query": question})
    print(f"\nAnswer: {result['result']}")
    print("\nSources:")
    for doc in result['source_documents']:
        print(f"  → Page {doc.metadata.get('page', '?')}: {doc.page_content[:120].strip()}...")
    print()
```

---

## Security Vulnerabilities — Know Before You Deploy

This tutorial system is intentionally simple. Before using in production, be aware of these risks:

**1. Prompt Injection**
A malicious user can craft inputs designed to override your system prompt. Always sanitise user input before it hits the chain. Consider a separate input validation layer.

**2. Data Leakage via Source Documents**
Returning raw source chunks to users may expose content they aren't authorised to see. In multi-user systems, implement role-based retrieval filtering — filter chunks by user permissions before they reach the LLM.

**3. Index Poisoning**
If your ingestion pipeline is open, an attacker can inject malicious documents into the vector store. Validate and hash all source files. Only ingest from trusted, authorised sources.

**4. Insecure API Key Handling**
Never hardcode your Anthropic API key. Always use environment variables or a secrets manager (AWS Secrets Manager, HashiCorp Vault). A leaked key means someone else runs LLM calls on your account.

**5. No Rate Limiting**
This system has no query throttle. Add rate limiting at the API gateway layer before any production deployment to prevent cost amplification attacks.

> For a production-grade implementation covering all of the above, watch the upcoming Thursday Live episode on [The AI Unpacked](https://www.youtube.com/@TheAIUnpackedWeekly).

---

## Real-World Use Cases

| Industry | Application |
|---|---|
| Legal | Case file and precedent retrieval — built for Full Faith Law Firm |
| EdTech | Student Q&A grounded in course material — powering Knoewit |
| HR & Compliance | Employee handbook chatbot, policy retrieval |
| Finance | Internal report Q&A, regulatory document search |
| Healthcare | Clinical guideline retrieval for clinical staff |
| Customer Support | Product documentation chatbot |

---

## What's Next — Part 2

Next Tutor Tuesday: **Add memory and chat history to your RAG system** — so it remembers what you asked earlier in the conversation.

Subscribe to [The AI Unpacked](https://www.youtube.com/@TheAIUnpackedWeekly) so you don't miss it.

---

## Requirements

```
anthropic
langchain
langchain-anthropic
langchain-community
faiss-cpu
pypdf
sentence-transformers
python-dotenv
```

---

## License

MIT — use it, fork it, build on it. Attribution appreciated.

---

## Connect

**Raj Jain** — AI Consultant & Cloud Architect

- YouTube: [@TheAIUnpackedWeekly](https://www.youtube.com/@TheAIUnpackedWeekly)
- X: [@the_ai_unpacked](https://x.com/the_ai_unpacked)
- LinkedIn: [linkedin.com/in/rajjain](https://linkedin.com/in/rajjain)
- Email: connect.real.raj.jain@gmail.com

---

*If this helped you — star the repo. It tells me who built along.*
