"""Lab 4: compare Arabic-centric checkpoints by all/Gulf/MSA slices."""

import time

import numpy as np
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from bayan.models.data import build_topic_dataset


MODELS = {
    "multilingual incumbent": "xlm-roberta-base",
    "CAMeLBERT-mix": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "CAMeLBERT-DA": "CAMeL-Lab/bert-base-arabic-camelbert-da",
}


def compute_fertility(texts, tokenizer):
    total_words = 0
    total_subwords = 0

    for text in texts:
        words = str(text).split()

        for word in words:
            tokens = tokenizer.tokenize(word)

            total_words += 1
            total_subwords += len(tokens)

    if total_words == 0:
        return 0.0

    return total_subwords / total_words


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
    )

    return {
        "macro_f1": macro_f1,
    }


def prepare_dataset(
    dataframe,
    tokenizer,
    label2id,
):
    dataset = Dataset.from_dict(
        {
            "text": dataframe["text"].tolist(),
            "label": [
                label2id[label]
                for label in dataframe["topic"]
            ],
        }
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
        )

    return dataset.map(
        tokenize,
        batched=True,
    )


def evaluate_slice(
    trainer,
    dataset,
):
    if len(dataset) == 0:
        return 0.0

    result = trainer.predict(dataset)

    predictions = np.argmax(
        result.predictions,
        axis=-1,
    )

    return f1_score(
        result.label_ids,
        predictions,
        average="macro",
    )


def run_model(
    model_name,
    checkpoint,
    train_df,
    validation_df,
    test_df,
    label_names,
    label2id,
    id2label,
):
    print("\n" + "=" * 60)
    print(model_name)
    print(checkpoint)
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint
    )

    train_dataset = prepare_dataset(
        train_df,
        tokenizer,
        label2id,
    )

    validation_dataset = prepare_dataset(
        validation_df,
        tokenizer,
        label2id,
    )

    test_dataset = prepare_dataset(
        test_df,
        tokenizer,
        label2id,
    )

    gulf_df = test_df[
        test_df["dialect_region"] == "Gulf"
    ]

    msa_df = test_df[
        test_df["dialect_region"] == "MSA"
    ]

    gulf_dataset = prepare_dataset(
        gulf_df,
        tokenizer,
        label2id,
    )

    msa_dataset = prepare_dataset(
        msa_df,
        tokenizer,
        label2id,
    )

    fertility = compute_fertility(
        test_df["text"].tolist(),
        tokenizer,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=len(label_names),
        label2id=label2id,
        id2label=id2label,
    )

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    training_args = TrainingArguments(
        output_dir=f"artifacts/arabic_bakeoff/{safe_name}",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    start_time = time.time()

    trainer.train()

    train_time = time.time() - start_time

    all_f1 = evaluate_slice(
        trainer,
        test_dataset,
    )

    gulf_f1 = evaluate_slice(
        trainer,
        gulf_dataset,
    )

    msa_f1 = evaluate_slice(
        trainer,
        msa_dataset,
    )

    print("\nResults")
    print(f"All macro-F1:  {all_f1:.4f}")
    print(f"Gulf macro-F1: {gulf_f1:.4f}")
    print(f"MSA macro-F1:  {msa_f1:.4f}")
    print(f"AR fertility:  {fertility:.4f}")
    print(f"Train time:    {train_time:.2f} seconds")

    return {
        "name": model_name,
        "checkpoint": checkpoint,
        "all": all_f1,
        "gulf": gulf_f1,
        "msa": msa_f1,
        "fertility": fertility,
    }


def main():
    datasets = build_topic_dataset()

    train_df = datasets["train"]
    validation_df = datasets["validation"]
    test_df = datasets["test"]

    # Arabic slice only.
    train_df = train_df[
        train_df["lang"] == "ar"
    ].reset_index(drop=True)

    validation_df = validation_df[
        validation_df["lang"] == "ar"
    ].reset_index(drop=True)

    test_df = test_df[
        test_df["lang"] == "ar"
    ].reset_index(drop=True)

    print("Arabic dataset sizes:")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(validation_df)}")
    print(f"Test: {len(test_df)}")

    print("\nTest dialect distribution:")
    print(
        test_df["dialect_region"]
        .value_counts()
    )

    label_names = sorted(
        train_df["topic"].unique()
    )

    label2id = {
        label: index
        for index, label in enumerate(label_names)
    }

    id2label = {
        index: label
        for label, index in label2id.items()
    }

    print("\nLabels:")
    print(label_names)

    results = []

    for model_name, checkpoint in MODELS.items():
        result = run_model(
            model_name,
            checkpoint,
            train_df,
            validation_df,
            test_df,
            label_names,
            label2id,
            id2label,
        )

        results.append(result)

    print("\n")
    print("=" * 75)
    print("LAB 4 — ARABIC MODEL BAKE-OFF")
    print("=" * 75)

    print(
        f"{'Model':25}"
        f"{'All':>10}"
        f"{'Gulf':>10}"
        f"{'MSA':>10}"
        f"{'Fertility':>12}"
    )

    for result in results:
        print(
            f"{result['name']:25}"
            f"{result['all']:>10.4f}"
            f"{result['gulf']:>10.4f}"
            f"{result['msa']:>10.4f}"
            f"{result['fertility']:>12.4f}"
        )

    incumbent = results[0]

    dialect_model = next(
        result
        for result in results
        if result["name"] == "CAMeLBERT-DA"
    )

    gulf_delta = (
        dialect_model["gulf"]
        - incumbent["gulf"]
    )

    print("\nCAMeLBERT-DA Gulf improvement vs incumbent:")
    print(
        f"{gulf_delta:+.4f} "
        f"({gulf_delta * 100:+.2f} points)"
    )


if __name__ == "__main__":
    main()