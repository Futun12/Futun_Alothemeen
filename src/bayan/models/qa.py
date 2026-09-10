"""Lab 3B: extractive QA post-processing."""


def best_span(
    start_logits,
    end_logits,
    offsets,
    *,
    null_score,
    null_threshold,
    max_answer_len=30,
    top_k=20,
):
    start_indices = sorted(
        range(len(start_logits)),
        key=lambda i: start_logits[i],
        reverse=True,
    )[:top_k]

    end_indices = sorted(
        range(len(end_logits)),
        key=lambda i: end_logits[i],
        reverse=True,
    )[:top_k]

    best_score = float("-inf")
    best_start = None
    best_end = None

    for start_index in start_indices:
        for end_index in end_indices:

            # Reject inverted spans.
            if end_index < start_index:
                continue

            # Reject spans that are too long.
            if end_index - start_index + 1 > max_answer_len:
                continue

            start_offset = offsets[start_index]
            end_offset = offsets[end_index]

            # Reject special tokens.
            if start_offset is None or end_offset is None:
                continue

            # Reject invalid offsets.
            if start_offset == (0, 0) or end_offset == (0, 0):
                continue

            start_char = start_offset[0]
            end_char = end_offset[1]

            if end_char <= start_char:
                continue

            score = (
                float(start_logits[start_index])
                + float(end_logits[end_index])
            )

            if score > best_score:
                best_score = score
                best_start = start_char
                best_end = end_char

    # No valid span found.
    if best_start is None:
        return {
            "answer": None,
            "start": None,
            "end": None,
            "score": None,
        }

    # Honest no-answer path.
    if null_score - best_score >= null_threshold:
        return {
            "answer": None,
            "start": None,
            "end": None,
            "score": best_score,
        }

    return {
        "answer": (best_start, best_end),
        "start": best_start,
        "end": best_end,
        "score": best_score,
    }