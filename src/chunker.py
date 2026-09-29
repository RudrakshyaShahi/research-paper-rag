from nltk.tokenize import sent_tokenize

from pdf_processor import extract_text_from_pdf
from text_cleaner import clean_text


def create_chunks(pages, source, chunk_size=5, overlap=2):

    all_chunks = []

    chunk_id = 0

    for page in pages:

        page_number = page["page"]

        # Clean the text from the page
        text = clean_text(page["text"])

        # Skip empty pages
        if not text:
            continue

        # Split page into sentences
        sentences = sent_tokenize(text)

        start = 0

        while start < len(sentences):

            end = start + chunk_size

            # Select sentences for this chunk
            chunk_sentences = sentences[start:end]

            # Join sentences into one piece of text
            chunk_text = " ".join(chunk_sentences)

            # Store chunk + metadata
            all_chunks.append({
                "chunk_id": chunk_id,
                "page": page_number,
                "source": source,
                "text": chunk_text
            })

            chunk_id += 1

            # Move forward while keeping overlap
            start += chunk_size - overlap

    return all_chunks


# =====================================
# TEST THE CHUNKING + METADATA
# =====================================

pdf_path = "data/papers/cnhandout.pdf"

# Extract pages from PDF
pages = extract_text_from_pdf(pdf_path)

# Create chunks
chunks = create_chunks(pages, pdf_path)


# Print first 5 chunks
for chunk in chunks[:5]:

    print("\n==============================")

    print("CHUNK ID:", chunk["chunk_id"])

    print("PAGE:", chunk["page"])

    print("SOURCE:", chunk["source"])

    print("TEXT:")

    print(chunk["text"])