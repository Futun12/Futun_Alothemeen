"""Lab 5: two-stage bilingual case search."""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from bayan.preprocessing.core import PREPROC_VERSION, preprocess


RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class CaseSearch:
    def __init__(self, prefix: str):
        self.index_path = Path(f"{prefix}_index.faiss")
        self.metadata_path = Path(f"{prefix}_metadata.jsonl")
        self.manifest_path = Path(f"{prefix}_manifest.json")

        self.manifest = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )

        if self.manifest["preproc_version"] != PREPROC_VERSION:
            raise ValueError("Preprocessing version mismatch.")

        self.index = faiss.read_index(str(self.index_path))

        self.metadata = []

        with self.metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.metadata.append(json.loads(line))

        self.encoder = SentenceTransformer(
            self.manifest["model"]
        )

        self.reranker = CrossEncoder(
            RERANKER_MODEL
        )

    def search(
        self,
        query: str,
        k: int = 5,
        candidates: int = 50,
        min_score: float = 0.25,
    ):
        clean_query = preprocess(query)

        if not clean_query:
            return []

        model_query = "query: " + clean_query

        query_vector = self.encoder.encode(
            [model_query],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        faiss.normalize_L2(query_vector)

        # Over-fetch because the corpus contains many duplicate texts.
        fetch_size = min(
            max(candidates * 20, candidates),
            self.index.ntotal,
        )

        bi_scores, indices = self.index.search(
            query_vector,
            fetch_size,
        )

        retrieved = []
        seen_texts = set()

        for score, index_id in zip(
            bi_scores[0],
            indices[0],
        ):
            if index_id < 0:
                continue

            case = dict(self.metadata[index_id])

            text_key = preprocess(
                str(case["case_text"])
            )

            # Do not allow identical cases to occupy
            # multiple candidate positions.
            if text_key in seen_texts:
                continue

            seen_texts.add(text_key)

            case["bi_score"] = float(score)
            retrieved.append(case)

            if len(retrieved) >= candidates:
                break

        if not retrieved:
            return []

        pairs = [
            [
                clean_query,
                preprocess(str(case["case_text"])),
            ]
            for case in retrieved
        ]

        rerank_scores = self.reranker.predict(
            pairs
        )

        for case, score in zip(
            retrieved,
            rerank_scores,
        ):
            case["score"] = float(score)

        retrieved.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        if min_score == float("-inf"):
            return retrieved[:k]

        return [
            case
            for case in retrieved
            if case["score"] >= min_score
        ][:k]