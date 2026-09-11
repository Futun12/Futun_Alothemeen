"""Lab 5: versioned FAISS index build."""

import json
from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

from bayan.preprocessing.core import PREPROC_VERSION, preprocess


DATA_PATH = "data/search/bayan_cases.csv"

MODEL_NAME = "intfloat/multilingual-e5-base"


def build_index(
    prefix: str,
    limit: int | None = None,
):
    # ---------------------------------------------------------
    # 1. Load the historical case corpus
    # ---------------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    if limit is not None:
        df = df.head(limit).copy()

    if len(df) == 0:
        raise ValueError("No cases available for indexing.")

    # ---------------------------------------------------------
    # 2. Apply the same preprocessing contract
    # ---------------------------------------------------------
    texts = [
        preprocess(str(text))
        for text in df["case_text"]
    ]

    # ---------------------------------------------------------
    # 3. Load bilingual embedding model
    # ---------------------------------------------------------
    model = SentenceTransformer(MODEL_NAME)

    # ---------------------------------------------------------
    # 4. Encode corpus
    # ---------------------------------------------------------
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    embeddings = embeddings.astype("float32")

    # ---------------------------------------------------------
    # 5. L2-normalise vectors
    # ---------------------------------------------------------
    faiss.normalize_L2(embeddings)

    # ---------------------------------------------------------
    # 6. Build FAISS index
    # ---------------------------------------------------------
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)

    index.add(embeddings)

    # ---------------------------------------------------------
    # 7. Prepare output paths
    # ---------------------------------------------------------
    prefix_path = Path(prefix)

    prefix_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_path = Path(
        f"{prefix}_index.faiss"
    )

    metadata_path = Path(
        f"{prefix}_metadata.jsonl"
    )

    manifest_path = Path(
        f"{prefix}_manifest.json"
    )

    # ---------------------------------------------------------
    # 8. Persist FAISS index
    # ---------------------------------------------------------
    faiss.write_index(
        index,
        str(index_path),
    )

    # ---------------------------------------------------------
    # 9. Persist metadata
    # ---------------------------------------------------------
    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for _, row in df.iterrows():
            record = row.to_dict()

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

    # ---------------------------------------------------------
    # 10. Persist manifest
    # ---------------------------------------------------------
    manifest = {
        "model": MODEL_NAME,
        "preproc_version": PREPROC_VERSION,
        "n_vectors": int(index.ntotal),
        "dim": int(dim),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "manifest_path": str(manifest_path),
        "n_vectors": int(index.ntotal),
        "dim": int(dim),
    }