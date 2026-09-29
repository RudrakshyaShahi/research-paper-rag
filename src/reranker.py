from sentence_transformers import CrossEncoder

from retrieval import retrieve


# =====================================
# 1. LOAD CROSS-ENCODER
# =====================================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# =====================================
# 2. RERANK FUNCTION
# =====================================

def rerank(question, results):

    # Create question + chunk pairs

    pairs = [
        [question, result["text"]]
        for result in results
    ]

    # Get cross-encoder scores

    scores = reranker.predict(pairs)

    # Add scores to results

    for result, score in zip(results, scores):

        result["rerank_score"] = float(score)

    # Sort by reranking score

    results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return results


# =====================================
# 3. TEST
# =====================================

if __name__ == "__main__":

    question = input(
        "Enter your question: "
    )

    # Retrieve more candidates first
    results = retrieve(
        question,
        k=10
    )

    # Rerank them
    reranked_results = rerank(
        question,
        results
    )

    # Display top 5
    print("\n==============================")
    print("RERANKED RESULTS")
    print("==============================")

    for rank, result in enumerate(
        reranked_results[:5],
        start=1
    ):

        print("\n------------------------------")

        print("Rank:", rank)

        print(
            "Chunk ID:",
            result["chunk_id"]
        )

        print(
            "Page:",
            result["page"]
        )

        print(
            "Source:",
            result["source"]
        )

        print(
            "FAISS score:",
            round(result["score"], 4)
        )

        print(
            "Rerank score:",
            round(
                result["rerank_score"],
                4
            )
        )

        print("\nTEXT:")

        print(result["text"])