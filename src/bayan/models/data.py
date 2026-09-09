"""Lab 3A: dataset construction and grouped split integrity."""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def build_topic_dataset(*args, **kwargs):
    df = pd.read_csv("data/raw/bayan_feedback.csv")

    # 70% train, 30% remaining
    first_split = GroupShuffleSplit(
        n_splits=1,
        train_size=0.70,
        random_state=42,
    )

    train_idx, remaining_idx = next(
        first_split.split(
            df,
            groups=df["citizen_group_id"],
        )
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    remaining_df = df.iloc[remaining_idx].reset_index(drop=True)

    # Split remaining 30% into 20% validation and 10% test
    second_split = GroupShuffleSplit(
        n_splits=1,
        train_size=2 / 3,
        random_state=42,
    )

    validation_idx, test_idx = next(
        second_split.split(
            remaining_df,
            groups=remaining_df["citizen_group_id"],
        )
    )

    validation_df = remaining_df.iloc[validation_idx].reset_index(drop=True)
    test_df = remaining_df.iloc[test_idx].reset_index(drop=True)

    return {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }