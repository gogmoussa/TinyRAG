import faiss
import pickle
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import datetime

# (Keep all previous imports and code)

def log_interaction(question, answer, filename="chat_log.txt"):
    timestamp = datetime.datetime.now().isoformat()
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}]\nQ: {question}\nA: {answer}\n\n")


# Load components
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("camping_index.faiss")
chunks, sources = pickle.load(open("camping_meta.pkl", "rb"))

llm = Llama(model_path="local_model.gguf", n_ctx=2048, n_threads=4)

# Ask a question
question = input("Ask a wilderness camping question: ")

# Embed the question and retrieve top matches
query_vector = embedding_model.encode([question])
_, I = index.search(query_vector, k=3)  # Top 3 results

# Gather relevant chunks
context = "\n---\n".join([chunks[i] for i in I[0]])

# Build prompt
prompt = f"""
You are a Canadian wilderness camping assistant.
Use the context below to answer the user's question.

Context:
{context}

Question:
{question}

Answer:"""

# Get answer
response = llm(prompt, max_tokens=256)

answer_text = response["choices"][0]["text"].strip()
print("\n💬 Answer:\n" + answer_text)
log_interaction(question, answer_text)