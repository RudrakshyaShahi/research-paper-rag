import fitz


def extract_text_from_pdf(pdf_path):

    doc = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc):

        text = page.get_text()

        pages.append({
            "page": page_number + 1,
            "text": text
        })

    return pages


pdf_path = "data/papers/cnhandout.pdf"

pages = extract_text_from_pdf(pdf_path)

for page in pages:

    print("PAGE:", page["page"])
    print(page["text"])
    print("-" * 50)