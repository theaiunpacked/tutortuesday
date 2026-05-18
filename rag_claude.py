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
