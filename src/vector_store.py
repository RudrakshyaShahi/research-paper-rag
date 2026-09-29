import faiss
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

from pdf_processor import extract_text_from_pdf
from chunker import create_chunks


# =====================================
# PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = BASE_DIR / "data" / "papers" / "cnhandout.pdf"

STORE_DIR = BASE_DIR / "vectorstore"

# Create vectorstore folder if it doesn't exist
STORE_DIR.mkdir(parents=True, exist_ok=True)


# =====================================
# BUILD VECTOR STORE
# =====================================

def build_vector_store():

    print("Starting vector store creation...")

    # 1. Extract PDF text
    pages = extract_text_from_pdf(str(PDF_PATH))

    print("PDF extracted.")

    # 2. Create chunks
    chunks = create_chunks(
        pages,
        PDF_PATH.name
    )

    print("Chunks created:", len(chunks))

    if len(chunks) == 0:
        raise ValueError("No chunks were created.")

    # 3. Load embedding model
    print("Loading embedding model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    # 4. Get chunk text
    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # 5. Create embeddings
    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )

    # 6. Convert to float32
    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    print("Embedding shape:", embeddings.shape)

    # 7. Normalize embeddings
    faiss.normalize_L2(embeddings)

    # 8. Create FAISS index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    # 9. Add embeddings
    index.add(embeddings)

    print("Vectors added to FAISS:", index.ntotal)

    # =====================================
    # SAVE FAISS INDEX
    # =====================================

    index_path = STORE_DIR / "index.faiss"

    faiss.write_index(
        index,
        str(index_path)
    )

    print("FAISS index saved to:")
    print(index_path)

    # =====================================
    # SAVE CHUNKS + METADATA
    # =====================================

    chunks_path = STORE_DIR / "chunks.json"

    with open(
        chunks_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("Chunks saved to:")
    print(chunks_path)

    # =====================================
    # FINAL CHECK
    # =====================================

    print("\n==============================")
    print("VECTOR STORE CREATED")
    print("==============================")

    print("Total chunks:", len(chunks))

    print("Vectors:", index.ntotal)

    print("Dimensions:", dimension)

    print("Index exists:", index_path.exists())

    print("Metadata exists:", chunks_path.exists())


# =====================================
# RUN PROGRAM
# =====================================

if __name__ == "__main__":
    build_vector_store()