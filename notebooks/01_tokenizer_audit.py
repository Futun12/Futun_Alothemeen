"""Lab 1 starter: audit four tokenizer candidates on Bayan AR/EN text."""
from pathlib import Path
import pandas as pd
import numpy as np
from transformers import AutoTokenizer


CANDIDATES = {
    "bert-base-multilingual-cased": "mBERT",
    "xlm-roberta-base": "XLM-R",
    "CAMeL-Lab/bert-base-arabic-camelbert-mix": "CAMeLBERT",
    "distilbert-base-uncased": "DistilBERT",
}

DATA = Path("data/raw/bayan_feedback.csv")

def fertility(tokenizer, texts) -> float:
    # total subword pieces / whitespace words
    total_pieces = 0
    total_words = 0

    for text in texts:
        words = text.split()
        pieces = tokenizer.tokenize(text)

        total_words += len(words)
        total_pieces += len(pieces)

    if total_words == 0:
        return 0.0

    return total_pieces / total_words
def unk_rate(tokenizer, texts) -> float:
    total_tokens = 0
    total_unk = 0

    for text in texts:
        tokens = tokenizer.tokenize(text)

        total_tokens += len(tokens)
        total_unk += tokens.count(tokenizer.unk_token)

    if total_tokens == 0:
        return 0.0

    return total_unk / total_tokens

def main():
    df = pd.read_csv(DATA)

    # Find the text column used in this dataset
    if "raw_text" in df.columns:
        text_column = "raw_text"
    elif "text" in df.columns:
        text_column = "text"
    elif "feedback_text" in df.columns:
        text_column = "feedback_text"
    else:
        raise ValueError(
            f"Could not find the feedback text column. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Separate Arabic and English feedback
    ar_texts = (
        df[df["lang"].str.lower() == "ar"][text_column]
        .dropna()
        .astype(str)
        .tolist()
    )

    en_texts = (
        df[df["lang"].str.lower() == "en"][text_column]
        .dropna()
        .astype(str)
        .tolist()
    )

    results = []

    # Audit each tokenizer
    for model_name, short_name in CANDIDATES.items():
        print(f"\nLoading {short_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Fertility
        ar_fertility = fertility(tokenizer, ar_texts)
        en_fertility = fertility(tokenizer, en_texts)
        ar_unk_rate = unk_rate(tokenizer, ar_texts)
        # Sequence lengths
        ar_lengths = [
            len(tokenizer.encode(text, add_special_tokens=True))
            for text in ar_texts
        ]

        en_lengths = [
            len(tokenizer.encode(text, add_special_tokens=True))
            for text in en_texts
        ]

        # p95 sequence length
        ar_p95 = np.percentile(ar_lengths, 95)
        en_p95 = np.percentile(en_lengths, 95)

        results.append(
            {
                "Tokenizer": short_name,
                "AR fertility": round(ar_fertility, 3),
                "EN fertility": round(en_fertility, 3),
                "AR p95 length": round(ar_p95, 1),
                "EN p95 length": round(en_p95, 1),
                "AR UNK rate": round(ar_unk_rate, 4),
        }
        )

    # Print final comparison table
    results_df = pd.DataFrame(results)

    print("\nTokenizer Audit Results")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
