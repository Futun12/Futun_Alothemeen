"""Lab 3A: fine-tune the Bayan topic classifier."""

import argparse
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from bayan.models.data import build_topic_dataset


CHECKPOINT = "xlm-roberta-base"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="artifacts/topic_classifier",
        help="Where to save the trained classifier artefact.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load grouped dataset.
    ds = build_topic_dataset()

    train_df = ds["train"]
    validation_df = ds["validation"]
    test_df = ds["test"]

    # Create label mapping.
    labels = sorted(train_df["topic"].unique())

    label2id = {
        label: i
        for i, label in enumerate(labels)
    }

    id2label = {
        i: label
        for label, i in label2id.items()
    }

    print("Labels:")
    print(labels)

    print("\nDataset sizes:")
    print("Train:", len(train_df))
    print("Validation:", len(validation_df))
    print("Test:", len(test_df))

    # Convert topic names to numeric labels.
    def prepare_dataframe(df):
        df = df[["text", "topic"]].copy()
        df["label"] = df["topic"].map(label2id)
        return df[["text", "label"]]

    train_dataset = Dataset.from_pandas(
        prepare_dataframe(train_df),
        preserve_index=False,
    )

    validation_dataset = Dataset.from_pandas(
        prepare_dataframe(validation_df),
        preserve_index=False,
    )

    test_dataset = Dataset.from_pandas(
        prepare_dataframe(test_df),
        preserve_index=False,
    )

    # Lab 1 checkpoint decision.
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
        )

    train_dataset = train_dataset.map(
        tokenize,
        batched=True,
    )

    validation_dataset = validation_dataset.map(
        tokenize,
        batched=True,
    )

    test_dataset = test_dataset.map(
        tokenize,
        batched=True,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    def compute_metrics(eval_pred):
        logits, labels_true = eval_pred
        predictions = np.argmax(logits, axis=-1)

        return {
            "accuracy": accuracy_score(
                labels_true,
                predictions,
            ),
            "macro_f1": f1_score(
                labels_true,
                predictions,
                average="macro",
            ),
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")

    trainer.train()

    print("\nValidation results:")
    validation_results = trainer.evaluate(
        validation_dataset
    )

    print(validation_results)

    print("\nFrozen test results:")
    test_results = trainer.evaluate(
        test_dataset
    )

    print(test_results)

    # Save re-runnable artefact.
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("\nSaved classifier to:")
    print(output_dir)


if __name__ == "__main__":
    main()