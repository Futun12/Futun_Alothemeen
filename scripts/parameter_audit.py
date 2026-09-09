"""Lab 2 starter: parameter accounting for mBERT and CAMeLBERT."""


from collections import defaultdict

from transformers import AutoModel


CHECKPOINTS = [
    "bert-base-multilingual-cased",
    "CAMeL-Lab/bert-base-arabic-camelbert-mix",
]


def get_bucket(parameter_name):
    """Return the parameter category based on its name."""

    name = parameter_name.lower()

    if "embedding" in name:
        return "embeddings"

    if "attention" in name:
        return "attention"

    if "intermediate" in name:
        return "ffn"

    if "output.dense" in name:
        return "ffn"

    if "layernorm" in name or "layer_norm" in name:
        return "norms"

    if "pooler" in name:
        return "pooler"

    return "other"


def audit(checkpoint):
    """Load a checkpoint and count its parameters by category."""

    print(f"\nLoading: {checkpoint}")

    model = AutoModel.from_pretrained(checkpoint)

    buckets = defaultdict(int)

    total_parameters = 0

    for name, parameter in model.named_parameters():
        parameter_count = parameter.numel()

        total_parameters += parameter_count

        bucket = get_bucket(name)

        buckets[bucket] += parameter_count

    print(f"\nCheckpoint: {checkpoint}")
    print(f"Total parameters: {total_parameters:,}")

    print("\nParameter breakdown:")

    categories = [
        "embeddings",
        "attention",
        "ffn",
        "norms",
        "pooler",
        "other",
    ]

    for category in categories:
        count = buckets[category]

        percentage = (
            count / total_parameters * 100
            if total_parameters > 0
            else 0
        )

        print(
            f"{category:12s} "
            f"{count:15,d} "
            f"{percentage:6.2f}%"
        )

    return {
        "checkpoint": checkpoint,
        "total": total_parameters,
        "buckets": dict(buckets),
    }


def main():
    results = []

    for checkpoint in CHECKPOINTS:
        result = audit(checkpoint)
        results.append(result)

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    categories = [
        "embeddings",
        "attention",
        "ffn",
        "norms",
        "pooler",
        "other",
    ]

    for result in results:
        print(f"\n{result['checkpoint']}")

        total = result["total"]

        for category in categories:
            count = result["buckets"].get(category, 0)

            percentage = count / total * 100

            print(
                f"{category:12s}: "
                f"{count:15,d} "
                f"({percentage:6.2f}%)"
            )


if __name__ == "__main__":
    main()
















