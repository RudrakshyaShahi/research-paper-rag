def create_citation_context(results):
    """
    Creates numbered context for the LLM.

    Each retrieved chunk gets a citation number
    such as [1], [2], [3].
    """

    context_parts = []

    for i, result in enumerate(results, start=1):

        context_parts.append(
            f"""
[{i}]
Source: {result['source']}
Page: {result['page']}

Text:
{result['text']}
"""
        )

    return "\n\n".join(context_parts)


def format_citations(results):
    """
    Creates a human-readable list of sources
    for displaying after the answer.
    """

    citations = []

    for i, result in enumerate(results, start=1):

        citation = (
            f"[{i}] "
            f"{result['source']} — "
            f"Page {result['page']}"
        )

        citations.append(citation)

    return "\n".join(citations)