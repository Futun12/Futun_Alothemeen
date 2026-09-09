"""Lab 3A: TF-IDF + LinearSVC baseline."""

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DATA_PATH = "data/raw/bayan_feedback.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    train_df = df[df["split"] == "train"]
    validation_df = df[df["split"] == "validation"]

    X_train = train_df["text"]
    y_train = train_df["topic"]

    X_validation = validation_df["text"]
    y_validation = validation_df["topic"]

    print("Train size:")
    print(len(train_df))

    print("\nValidation size:")
    print(len(validation_df))

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LinearSVC(),
            ),
        ]
    )

    print("\nTraining TF-IDF + LinearSVC baseline...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)

    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro",
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_validation,
            predictions,
            digits=4,
        )
    )

    print("\nBaseline macro-F1:")
    print(f"{macro_f1:.4f}")


if __name__ == "__main__":
    main()