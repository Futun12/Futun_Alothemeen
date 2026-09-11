"""Lab 5: labelled-query retrieval evaluation."""

import json
import time

import faiss
import numpy as np

from bayan.preprocessing.core import preprocess
from bayan.search.service import CaseSearch


PREFIX = "artifacts/search/case_index_v1"
QUERY_FILE = "data/search/bayan_queries.jsonl"

K = 10
CANDIDATES = 50


def load_queries():
    with open(QUERY_FILE, encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def recall_at_10(results, relevant):
    relevant = set(relevant)

    if not relevant:
        return 0.0

    return len(
        set(results[:10]) & relevant
    ) / len(relevant)


def mrr_at_10(results, relevant):
    relevant = set(relevant)

    for rank, case_id in enumerate(
        results[:10],
        start=1,
    ):
        if case_id in relevant:
            return 1.0 / rank

    return 0.0


# ---------------------------------------------------------
# Correct bi-encoder retrieval path
# ---------------------------------------------------------

def bi_search(searcher, query):
    clean = preprocess(query)

    if "e5" in searcher.manifest["model"].lower():
        clean = "query: " + clean

    vector = searcher.encoder.encode(
        [clean],
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    # Correct L2 normalisation
    faiss.normalize_L2(vector)

    scores, indices = searcher.index.search(
        vector,
        min(1000, searcher.index.ntotal),
    )

    results = []
    seen_texts = set()

    for index_id in indices[0]:
        if index_id < 0:
            continue

        case = searcher.metadata[index_id]

        text_key = preprocess(
            str(case["case_text"])
        )

        if text_key in seen_texts:
            continue

        seen_texts.add(text_key)

        results.append(
            case["case_id"]
        )

        if len(results) == K:
            break

    return results


# ---------------------------------------------------------
# Lab 5 Step 4
# Build intentionally broken unnormalised FAISS index
# ---------------------------------------------------------

def build_unnormalised_index(searcher):
    texts = []

    for case in searcher.metadata:
        clean = preprocess(
            str(case["case_text"])
        )

        if "e5" in searcher.manifest["model"].lower():
            clean = "passage: " + clean

        texts.append(clean)

    vectors = searcher.encoder.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    # INTENTIONAL BUG:
    # Do NOT run:
    # faiss.normalize_L2(vectors)

    bug_index = faiss.IndexFlatIP(
        vectors.shape[1]
    )

    bug_index.add(vectors)

    return bug_index


# ---------------------------------------------------------
# Lab 5 Step 4
# Search broken index without L2 normalisation
# ---------------------------------------------------------

def bi_search_without_l2(
    searcher,
    bug_index,
    query,
):
    clean = preprocess(query)

    if "e5" in searcher.manifest["model"].lower():
        clean = "query: " + clean

    vector = searcher.encoder.encode(
        [clean],
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    # INTENTIONAL BUG:
    # Do NOT run:
    # faiss.normalize_L2(vector)

    _, indices = bug_index.search(
        vector,
        K,
    )

    return [
        searcher.metadata[index_id]["case_id"]
        for index_id in indices[0]
        if index_id >= 0
    ]


def main():
    queries = load_queries()

    searcher = CaseSearch(PREFIX)

    answerable = [
        q
        for q in queries
        if not q["no_answer"]
    ]

    no_answer = [
        q
        for q in queries
        if q["no_answer"]
    ]

    before_recall = []
    before_mrr = []

    after_recall = []
    after_mrr = []

    ar_scores = []
    en_scores = []

    bi_latency = []
    rerank_latency = []

    # -----------------------------------------------------
    # Step 3 evaluation
    # -----------------------------------------------------

    for q in answerable:
        relevant = q["relevant_case_ids"]

        # Bi-encoder retrieval
        start = time.perf_counter()

        before = bi_search(
            searcher,
            q["query"],
        )

        bi_latency.append(
            time.perf_counter() - start
        )

        before_recall.append(
            recall_at_10(
                before,
                relevant,
            )
        )

        before_mrr.append(
            mrr_at_10(
                before,
                relevant,
            )
        )

        # Two-stage retrieval
        start = time.perf_counter()

        results = searcher.search(
            q["query"],
            k=K,
            candidates=CANDIDATES,
            min_score=float("-inf"),
        )

        rerank_latency.append(
            time.perf_counter() - start
        )

        after = [
            result["case_id"]
            for result in results
        ]

        recall = recall_at_10(
            after,
            relevant,
        )

        mrr = mrr_at_10(
            after,
            relevant,
        )

        after_recall.append(recall)
        after_mrr.append(mrr)

        if q["lang"] == "ar":
            ar_scores.append(
                (recall, mrr)
            )
        else:
            en_scores.append(
                (recall, mrr)
            )

    # -----------------------------------------------------
    # Language slices
    # -----------------------------------------------------

    ar_recall = np.mean(
        [x[0] for x in ar_scores]
    )

    en_recall = np.mean(
        [x[0] for x in en_scores]
    )

    ar_mrr = np.mean(
        [x[1] for x in ar_scores]
    )

    en_mrr = np.mean(
        [x[1] for x in en_scores]
    )

    # -----------------------------------------------------
    # Print normal retrieval results
    # -----------------------------------------------------

    print("\nWithout reranking")

    print(
        f"Recall@10: "
        f"{np.mean(before_recall):.4f}"
    )

    print(
        f"MRR@10: "
        f"{np.mean(before_mrr):.4f}"
    )

    print("\nWith reranking")

    print(
        f"Recall@10: "
        f"{np.mean(after_recall):.4f}"
    )

    print(
        f"MRR@10: "
        f"{np.mean(after_mrr):.4f}"
    )

    # -----------------------------------------------------
    # Cross-lingual gap
    # -----------------------------------------------------

    print("\nCross-lingual slice gap")

    print(
        f"Recall gap: "
        f"{abs(ar_recall - en_recall):.4f}"
    )

    print(
        f"MRR gap: "
        f"{abs(ar_mrr - en_mrr):.4f}"
    )

    # -----------------------------------------------------
    # No-answer threshold
    # -----------------------------------------------------

    print(
        "\nNo-answer threshold behaviour"
    )

    thresholds = [
        -5.0,
        -2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
        5.0,
    ]

    best_threshold = None
    best_correct = -1

    for threshold in thresholds:
        correct = 0

        for q in no_answer:
            results = searcher.search(
                q["query"],
                k=K,
                candidates=CANDIDATES,
                min_score=threshold,
            )

            if not results:
                correct += 1

        print(
            f"min_score={threshold}: "
            f"{correct}/{len(no_answer)}"
        )

        if correct > best_correct:
            best_correct = correct
            best_threshold = threshold

    print(
        f"\nBest min_score: "
        f"{best_threshold}"
    )

    print(
        f"No-answer correctness: "
        f"{best_correct}/{len(no_answer)}"
    )

    # -----------------------------------------------------
    # Latency
    # -----------------------------------------------------

    print("\nStage latency")

    print(
        f"Bi-encoder: "
        f"{np.mean(bi_latency) * 1000:.2f} "
        f"ms/query"
    )

    print(
        f"Two-stage: "
        f"{np.mean(rerank_latency) * 1000:.2f} "
        f"ms/query"
    )

    # =====================================================
    # LAB 5 STEP 4
    # Planted unnormalised-vector bug
    # =====================================================

    print(
        "\nBuilding intentionally "
        "unnormalised index..."
    )

    bug_index = build_unnormalised_index(
        searcher
    )

    bug_recall = []
    bug_mrr = []

    for q in answerable:
        results = bi_search_without_l2(
            searcher,
            bug_index,
            q["query"],
        )

        bug_recall.append(
            recall_at_10(
                results,
                q["relevant_case_ids"],
            )
        )

        bug_mrr.append(
            mrr_at_10(
                results,
                q["relevant_case_ids"],
            )
        )

    # -----------------------------------------------------
    # Step 4 comparison
    # -----------------------------------------------------

    print(
        "\nStep 4 - "
        "Planted unnormalised-vector bug"
    )

    print(
        "\nCorrect L2-normalised path"
    )

    print(
        f"Recall@10: "
        f"{np.mean(before_recall):.4f}"
    )

    print(
        f"MRR@10: "
        f"{np.mean(before_mrr):.4f}"
    )

    print(
        "\nUnnormalised index/query path"
    )

    print(
        f"Recall@10: "
        f"{np.mean(bug_recall):.4f}"
    )

    print(
        f"MRR@10: "
        f"{np.mean(bug_mrr):.4f}"
    )


if __name__ == "__main__":
    main()