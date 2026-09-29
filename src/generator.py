import os

from google import genai

from retrieval import retrieve
from reranker import rerank
from citations import create_citation_context, format_citations


# ==========================================
# GEMINI CLIENT
# ==========================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is not set.\n"
        'Set it in PowerShell using:\n'
        '$env:GEMINI_API_KEY="YOUR_API_KEY"'
    )

client = genai.Client(api_key=api_key)


# ==========================================
# GENERATE ANSWER
# ==========================================

def generate_answer(question, context):

    prompt = f"""
You are a question-answering assistant for a
document-based Retrieval-Augmented Generation (RAG) system.

Your job is to answer the user's question using ONLY
the information provided in the context.

IMPORTANT RULES:

1. Use ONLY the provided context.
2. Do NOT use outside knowledge.
3. Do NOT make up information.
4. If the answer cannot be found in the context, say:
   "The answer is not available in the provided document."
5. Keep the answer clear and concise.
6. Whenever you use information from a source,
   add its citation number at the end of the relevant sentence.
7. Use citations exactly in this format:
   [1], [2], [3], etc.
8. Only use citation numbers that actually exist in the context.
9. Do not create fake citations.
10. If multiple sources support the same statement,
    you may use multiple citations, for example [1][3].

CONTEXT:
====================

{context}

====================

QUESTION:

{question}

====================

ANSWER:
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text


# ==========================================
# MAIN RAG PIPELINE
# ==========================================

if __name__ == "__main__":

    # --------------------------------------
    # 1. Get question
    # --------------------------------------

    question = input("Enter your question: ")

    if not question.strip():

        print("Question cannot be empty.")

        exit()


    # --------------------------------------
    # 2. Retrieve chunks
    # --------------------------------------

    print("\nRetrieving relevant chunks...")

    results = retrieve(
        question,
        k=10
    )


    # --------------------------------------
    # 3. Rerank chunks
    # --------------------------------------

    print("Reranking results...")

    reranked_results = rerank(
        question,
        results
    )


    # --------------------------------------
    # 4. Select top 5
    # --------------------------------------

    top_results = reranked_results[:5]


    # --------------------------------------
    # 5. Create numbered citation context
    # --------------------------------------

    context = create_citation_context(
        top_results
    )


    # --------------------------------------
    # 6. Generate answer
    # --------------------------------------

    print("Generating answer with Gemini...")

    answer = generate_answer(
        question,
        context
    )


    # --------------------------------------
    # 7. Display answer
    # --------------------------------------

    print("\n==============================")
    print("ANSWER")
    print("==============================")

    print(answer)


    # --------------------------------------
    # 8. Display citations
    # --------------------------------------

    print("\n==============================")
    print("SOURCES")
    print("==============================")

    print(
        format_citations(top_results)
    )