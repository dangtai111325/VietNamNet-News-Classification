"""Exact text transformations used by the canonical notebook."""
import re


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    from pyvi import ViTokenizer
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = ViTokenizer.tokenize(text)
    return re.sub(r"\s+", " ", text).strip()


def prepare_text(title: str, content: str) -> str:
    return clean_text(str(title) + " " + str(content))
