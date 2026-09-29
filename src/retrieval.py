
import faiss
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


# =====================================
# 1. PROJECT PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent
STORE_DIR = BASE_DIR / "vectorstore"


# =====================================
# 2. LOAD EMBEDDING MODEL
# =====================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# =====================================
# 3. LOAD SAVED FAISS INDEX
# =====================================

index = faiss.read_index(
    str(STORE_DIR / "index.faiss")
)


# =====================================
# 4. LOAD SAVED CHUNKS
# =====================================

with open(
    STORE_DIR / "chunks.json",
    "r",
    encoding="utf-8"
) as file:
    chunks = json.load(file)


# =====================================
# 5. CHECK VECTOR-METADATA MAPPING
# =====================================

if index.ntotal != len(chunks):
    raise ValueError(
        "FAISS index and chunk metadata have different sizes."
    )


# =====================================
# 6. RETRIEVAL FUNCTION
# =====================================

def retrieve(question, k=5):

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    # Convert question to an embedding
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )

    # FAISS requires float32
    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    # Normalize query vector
    faiss.normalize_L2(query_embedding)

    # Avoid requesting more results than available
    k = min(k, index.ntotal)

    # Search FAISS
    scores, indices = index.search(
        query_embedding,
        k
    )

    # Collect retrieved chunks
    results = []

    for index_id, score in zip(
        indices[0],
        scores[0]
    ):

        chunk = chunks[int(index_id)]

        results.append({
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"],
            "source": chunk["source"],
            "text": chunk["text"],
            "score": float(score)
        })

    return results


# =====================================
# 7. TEST RETRIEVAL
# =====================================

if __name__ == "__main__":

    question = input("Enter your question: ")

    results = retrieve(question, k=5)

    for rank, result in enumerate(results, start=1):

        print("\n==========================")
        print("RANK:", rank)
        print("==========================")

        print("Chunk ID:", result["chunk_id"])
        print("Page:", result["page"])
        print("Source:", result["source"])
        print("Similarity:", round(result["score"], 4))

        print("\nTEXT:")
        print(result["text"])