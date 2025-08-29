import os
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load and chunk text files
def load_chunks(folder_path, chunk_size=500):
    chunks, sources = [], []
    for file in Path(folder_path).glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                chunks.append(chunk)
                sources.append(file.name)
    return chunks, sources

# Embed and build index
def build_faiss_index(chunks):
    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

chunks, sources = load_chunks("canada_wilderness_camping")
index, embeddings = build_faiss_index(chunks)

# Save index and metadata
faiss.write_index(index, "camping_index.faiss")
with open("camping_meta.pkl", "wb") as f:
    pickle.dump((chunks, sources), f)

print("✅ Index built and saved.")
