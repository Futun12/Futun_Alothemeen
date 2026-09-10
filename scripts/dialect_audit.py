"""Lab 4: audit dialect mix in the Arabic Bayan feedback slice."""

import pandas as pd


DATA_PATH = "data/raw/bayan_feedback.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    # Keep Arabic feedback only.
    arabic_df = df[df["lang"] == "ar"].copy()

    print("Arabic feedback rows:")
    print(len(arabic_df))

    # Count dialect/region values.
    distribution = (
        arabic_df["dialect_region"]
        .fillna("unknown")
        .value_counts()
        .rename_axis("dialect_region")
        .reset_index(name="count")
    )

    # Add percentage.
    distribution["percentage"] = (
        distribution["count"]
        / len(arabic_df)
        * 100
    ).round(2)

    print("\nDialect / region distribution:")
    print(distribution.to_string(index=False))

    print("\nImplication:")
    print(
        "Evaluating only on Modern Standard Arabic (MSA) would not fully "
        "represent Bayan's Arabic data because dialectal language can differ "
        "in vocabulary, spelling, and expression across regions."
    )


if __name__ == "__main__":
    main()