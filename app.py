from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -------------------------
# Load LLM (Qwen)
# -------------------------
llm = pipeline(
    "text-generation",
    model="Qwen/Qwen3-0.6B",
    device=-1  # CPU
)

# -------------------------
# Load Embedding Model
# -------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# Load Documents
# -------------------------
with open("data.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f if line.strip()]

# -------------------------
# Create FAISS Index
# -------------------------
doc_embeddings = embedder.encode(documents)
dimension = doc_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(title="RAG API")

# -------------------------
# Request Model
# -------------------------
class QueryRequest(BaseModel):
    question: str
    max_tokens: int = None  # Optional, will be dynamic if not provided

# -------------------------
# Root Endpoint
# -------------------------
@app.get("/")
def root():
    return {"status": "RAG API running"}

# -------------------------
# Ask Question Endpoint
# -------------------------
@app.post("/ask")
def ask_question(req: QueryRequest):
    # -------------------------
    # Dynamically calculate max_tokens
    # -------------------------
    if req.max_tokens is None:
        base_tokens = 30
        extra_tokens_per_word = 5
        context_length = sum(len(doc.split()) for doc in documents)
        dynamic_max_tokens = base_tokens + len(req.question.split()) * extra_tokens_per_word + int(context_length * 0.1)
        # Limit tokens to avoid too long generation
        dynamic_max_tokens = min(max(dynamic_max_tokens, 30), 150)
    else:
        dynamic_max_tokens = req.max_tokens

    # -------------------------
    # Retrieve top relevant documents
    # -------------------------
    question_embedding = embedder.encode([req.question])
    _, indices = index.search(np.array(question_embedding), k=3)
    retrieved_docs = [documents[i] for i in indices[0]]
    context = "\n".join(retrieved_docs)

    # -------------------------
    # Build prompt (instruction-locked)
    # -------------------------
    prompt = f"""
You are a highly reliable and precise question-answering assistant.
Answer ONLY using the provided context.
Give a SHORT and DIRECT answer.
Do NOT explain, do NOT guess if the answer is not in the context.
Do NOT create options.

Context:
{context}

Question:
{req.question}

Final Answer:
"""

    # -------------------------
    # Generate answer
    # -------------------------
    output = llm(
        prompt,
        max_new_tokens=dynamic_max_tokens,
        do_sample=False,
        temperature=0
    )

    # Extract only the answer after "Final Answer:"
    generated_text = output[0]["generated_text"]
    answer = generated_text.split("Final Answer:")[-1].strip()

    # If empty, say "Answer not found"
    if not answer:
        answer = "Answer not found in the provided context."

    return {
        "question": req.question,
        "max_tokens_used": dynamic_max_tokens,
        "retrieved_context": retrieved_docs,
        "answer": answer
    }
