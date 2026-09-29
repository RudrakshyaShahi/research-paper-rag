import numpy as np

from sentence_transformers import SentenceTransformer

from pdf_processor import extract_text_from_pdf
from chunker import create_chunks


def cosine_similarity(vector_a, vector_b):

    dot_product = np.dot(vector_a, vector_b)

    magnitude_a = np.linalg.norm(vector_a)

    magnitude_b = np.linalg.norm(vector_b)

    similarity = dot_product / (magnitude_a * magnitude_b)

    return similarity


# =====================================
# LOAD MODEL
# =====================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# =====================================
# PDF
# =====================================

pdf_path = "data/papers/cnhandout.pdf"


# =====================================
# EXTRACT + CHUNK
# =====================================

pages = extract_text_from_pdf(pdf_path)

chunks = create_chunks(pages, pdf_path)


# =====================================
# CREATE DOCUMENT EMBEDDINGS
# =====================================

texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(texts)


# =====================================
# USER QUESTION
# =====================================

question = "What is TCP?"


# =====================================
# CREATE QUESTION EMBEDDING
# =====================================

question_embedding = model.encode(question)


# =====================================
# CALCULATE SIMILARITY
# =====================================

scores = []

for i, embedding in enumerate(embeddings):

    score = cosine_similarity(
        question_embedding,
        embedding
    )

    scores.append((i, score))


# =====================================
# SORT BY SIMILARITY
# =====================================

scores.sort(
    key=lambda x: x[1],
    reverse=True
)


# =====================================
# DISPLAY TOP 5
# =====================================

print("\n==============================")
print("TOP 5 RELEVANT CHUNKS")
print("==============================")

for index, score in scores[:5]:

    print("\n------------------------------")

    print("Chunk ID:", chunks[index]["chunk_id"])

    print("Page:", chunks[index]["page"])

    print("Similarity:", score)

    print("Text:")

    print(chunks[index]["text"])