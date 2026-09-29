import re


def clean_text(text):

    # Normalize Windows line endings
    text = text.replace("\r\n", "\n")

    # Fix words broken by a line break
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    # Replace remaining newlines with spaces
    text = re.sub(r'\n+', ' ', text)

    # Remove excessive spaces
    text = re.sub(r'[ \t]+', ' ', text)

    # Remove spaces at the beginning/end
    text = text.strip()

    return text
