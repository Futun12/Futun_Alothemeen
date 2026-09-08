"""Lab 1 starter: versioned bilingual preprocessing for Bayan."""
import re
import unicodedata

PREPROC_VERSION = "1.2.0"


def normalize(text: str) -> str:
    """Return deterministic Bayan normalisation while preserving task signal."""
    # TODO(Lab 1): implement the course normalisation contract.
    text = unicodedata.normalize("NFKC", text) 
    text = text.replace("\u0640", "") #tatweel 
    text = re.sub(r"<br\s*/?>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def mask_pii(text: str) -> str:
    """Mask supported phone numbers and Saudi national-ID-shaped values."""
    # TODO(Lab 1): replace supported PII with <PHONE> / <NATIONAL_ID>.
    text = re.sub(r"(?<!\d)(?:\+966|966|0)5\d{8}(?!\d)","<PHONE>",text,)
    text = re.sub(r"(?<!\d)[12]\d{9}(?!\d)", "<NATIONAL_ID>",text,)
    return text


def preprocess(text: str) -> str:
    """Apply the shared train/eval/serve preprocessing contract."""
    # TODO(Lab 1): compose masking and normalisation in the intended order.
    text = normalize(text)
    text = mask_pii(text)
    return text
