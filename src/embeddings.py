from sentence_transformers import SentenceTransformer

from pdf_processor import extract_text_from_pdf
from chunker import create_chunks


# =====================================
# 1. LOAD THE EMBEDDING MODEL
# =====================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# =====================================
# 2. PDF PATH
# =====================================

pdf_path = "data/papers/cnhandout.pdf"


# =====================================
# 3. EXTRACT TEXT FROM PDF
# =====================================

pages = extract_text_from_pdf(pdf_path)


# =====================================
# 4. CREATE CHUNKS
# =====================================

chunks = create_chunks(pages, pdf_path)


# =====================================
# 5. GET TEXT FROM EACH CHUNK
# =====================================

texts = [chunk["text"] for chunk in chunks]


# =====================================
# 6. CREATE EMBEDDINGS
# =====================================

embeddings = model.encode(texts)


# =====================================
# 7. TEST THE EMBEDDINGS
# =====================================

print("\n==============================")
print("EMBEDDING RESULTS")
print("==============================")

print("Number of chunks:", len(chunks))

print("Embedding shape:", embeddings.shape)

print("First embedding:")

print(embeddings[0])


# =====================================
# 8. CHECK FIRST CHUNK + EMBEDDING
# =====================================

print("\n==============================")
print("FIRST CHUNK")
print("==============================")

print("Chunk ID:", chunks[0]["chunk_id"])
print("Page:", chunks[0]["page"])
print("Source:", chunks[0]["source"])

print("\nText:")
print(chunks[0]["text"])

print("\nEmbedding dimensions:", len(embeddings[0]))