"""Lab 3B: run the 12-question QA smoke set."""

import json
import re
import string

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from bayan.models.qa import best_span


SMOKE_PATH = "data/eval/qa_smoke_set.json"
QA_DATA_PATH = "data/models/bayan_qa.json"

CHECKPOINT = "deepset/xlm-roberta-base-squad2"

NULL_THRESHOLD = 0.0


def load_questions(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    examples = []

    for item in data["data"]:
        for paragraph in item["paragraphs"]:
            context = paragraph["context"]

            for qa in paragraph["qas"]:
                examples.append(
                    {
                        "id": qa["id"],
                        "question": qa["question"],
                        "context": context,
                        "answers": qa["answers"],
                        "is_impossible": qa["is_impossible"],
                    }
                )

    return examples


def normalize_answer(text):
    """
    Normalize predicted and expected answers for smoke-set comparison.

    This ignores:
    - capitalization
    - punctuation
    - extra whitespace
    - English articles: a, an, the
    """

    if text is None:
        return None

    text = text.lower()

    text = "".join(
        char for char in text
        if char not in string.punctuation
    )

    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text,
    )

    text = " ".join(text.split())

    return text


def main():

    # --------------------------------------------------
    # 1. Load supplied datasets
    # --------------------------------------------------

    smoke_examples = load_questions(SMOKE_PATH)
    qa_examples = load_questions(QA_DATA_PATH)

    # The supplied smoke file currently contains
    # answerable examples only.
    #
    # Use 9 answerable questions from the smoke file
    # and 3 supplied unanswerable QA examples so that
    # the lab contract is 9 answerable + 3 null.

    answerable = [
        example
        for example in smoke_examples
        if not example["is_impossible"]
    ][:9]

    unanswerable = [
        example
        for example in qa_examples
        if example["is_impossible"]
    ][:3]

    examples = answerable + unanswerable

    print("QA smoke set")
    print("Answerable:", len(answerable))
    print("Unanswerable:", len(unanswerable))

    # --------------------------------------------------
    # 2. Load QA checkpoint
    # --------------------------------------------------

    print("\nLoading QA checkpoint:")
    print(CHECKPOINT)

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        CHECKPOINT
    )

    model.eval()

    # --------------------------------------------------
    # 3. Counters
    # --------------------------------------------------

    answerable_correct = 0
    null_correct = 0

    # --------------------------------------------------
    # 4. Run QA examples
    # --------------------------------------------------

    for example in examples:

        question = example["question"]
        context = example["context"]

        encoded = tokenizer(
            question,
            context,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=384,
        )

        # Save offsets before passing input to model.
        offsets = encoded.pop(
            "offset_mapping"
        )[0].tolist()

        sequence_ids = encoded.sequence_ids(0)

        # --------------------------------------------------
        # Only context tokens can form an answer.
        # --------------------------------------------------

        context_offsets = []

        for index, offset in enumerate(offsets):

            if sequence_ids[index] == 1:
                context_offsets.append(
                    tuple(offset)
                )
            else:
                context_offsets.append(None)

        # --------------------------------------------------
        # Run QA model
        # --------------------------------------------------

        with torch.no_grad():
            outputs = model(**encoded)

        start_logits = (
            outputs.start_logits[0]
            .detach()
            .cpu()
            .numpy()
        )

        end_logits = (
            outputs.end_logits[0]
            .detach()
            .cpu()
            .numpy()
        )

        # --------------------------------------------------
        # Null/no-answer score
        # --------------------------------------------------

        cls_index = 0

        null_score = (
            float(start_logits[cls_index])
            + float(end_logits[cls_index])
        )

        # --------------------------------------------------
        # Find best valid answer span
        # --------------------------------------------------

        result = best_span(
            start_logits,
            end_logits,
            context_offsets,
            null_score=null_score,
            null_threshold=NULL_THRESHOLD,
            max_answer_len=30,
            top_k=20,
        )

        # --------------------------------------------------
        # 5. Unanswerable example
        # --------------------------------------------------

        if example["is_impossible"]:

            predicted_answer = result.get(
                "answer"
            )

            correct = (
                predicted_answer is None
            )

            if correct:
                null_correct += 1

            print(
                example["id"],
                "| expected = None",
                "| predicted =",
                predicted_answer,
                "|",
                "PASS" if correct else "FAIL",
            )

            continue

        # --------------------------------------------------
        # 6. Answerable example
        # --------------------------------------------------

        answer_span = result.get("answer")

        if answer_span is None:

            predicted_text = None

        else:

            start_char, end_char = answer_span

            predicted_text = context[
                start_char:end_char
            ].strip()

        expected_answers = [
            answer["text"].strip()
            for answer in example["answers"]
        ]

        # --------------------------------------------------
        # Normalize before comparison.
        #
        # Example:
        #
        # expected:
        # Bayan portal
        #
        # prediction:
        # the Bayan portal
        #
        # Both normalize to:
        # bayan portal
        # --------------------------------------------------

        normalized_prediction = normalize_answer(
            predicted_text
        )

        normalized_expected = [
            normalize_answer(answer)
            for answer in expected_answers
        ]

        correct = (
            normalized_prediction is not None
            and normalized_prediction
            in normalized_expected
        )

        if correct:
            answerable_correct += 1

        print(
            example["id"],
            "| expected =",
            expected_answers,
            "| predicted =",
            predicted_text,
            "|",
            "PASS" if correct else "FAIL",
        )

    # --------------------------------------------------
    # 7. Final results
    # --------------------------------------------------

    print("\nQA Smoke Results")

    print(
        f"Answerable: "
        f"{answerable_correct}/"
        f"{len(answerable)} "
        f"correct spans"
    )

    print(
        f"Unanswerable: "
        f"{null_correct}/"
        f"{len(unanswerable)} "
        f"answer=None"
    )

    # --------------------------------------------------
    # 8. Lab target
    # --------------------------------------------------

    if (
        answerable_correct == len(answerable)
        and null_correct == len(unanswerable)
    ):

        print(
            "\nPASS: QA smoke target achieved."
        )

    else:

        print(
            "\nFAIL: QA smoke target not achieved."
        )


if __name__ == "__main__":
    main()