#!/usr/bin/env python3
"""Train a model on the provided dataset."""

__author__ = "Dave Hall <me@davehall.com.au>"
__copyright__ = "Copyright 2024- 2025, Skwashd Services Pty Ltd https://gata.works"
__license__ = "MIT"


import datetime
import json
import os
import shutil
import tarfile
from collections.abc import Callable
from dataclasses import dataclass, field

import datasets
import numpy as np
import pandas as pd
import torch
import transformers.utils.logging
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

datasets.disable_progress_bar()
transformers.utils.logging.set_verbosity_warning()
transformers.utils.logging.disable_progress_bar()


@dataclass
class SageMakerEnvironmentArguments:
    """Arguments pertaining to SageMaker training environment."""

    output_data_dir: str = field(
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data/"),
        metadata={"help": "SageMaker output data directory"},
    )
    model_dir: str = field(
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model/"),
        metadata={"help": "SageMaker model output directory"},
    )
    n_gpus: int = field(
        default=int(os.environ.get("SM_NUM_GPUS", "0")),
        metadata={"help": "Number of GPUs"},
    )
    training_dir: str = field(
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train/"),
        metadata={"help": "Training data directory"},
    )
    test_dir: str = field(
        default=os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test/"),
        metadata={"help": "Test data directory"},
    )


@dataclass
class ModelArguments:
    """Arguments pertaining to model configuration."""

    download_model: bool = field(
        default=False, metadata={"help": "Whether to download the model"}
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """Custom training arguments that inherit from TrainingArguments with default output_dir."""

    output_dir: str = field(
        default=os.environ.get("SM_MODEL_DIR", "output"),
        metadata={
            "help": "The output directory where model predictions and checkpoints will be written."
        },
    )

    checkpoint_dir: str = field(
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data/")
    )

    disable_tqdm: bool = field(default=True)
    eval_strategy: str = field(default="epoch")
    load_best_model_at_end: bool = field(default=True)
    logging_steps: float = field(default=100)
    logging_strategy: str = field(default="steps")
    metric_for_best_model: str = field(default="f1")
    num_train_epochs: int = field(default=4)
    save_strategy: str = field(default="epoch")

    # Disable some features that are not needed for CPU training. Override with hyperparameters.json
    dataloader_num_workers: int = field(default=0)
    fp16: bool = field(default=False)  # Will be updated based on GPU availability
    optim: str = field(default="adamw_torch")  # Use PyTorch's AdamW implementation
    gradient_accumulation_steps: int = field(
        default=4
    )  # Accumulate gradients to simulate larger batch


def compute_metrics(pred: EvalPrediction) -> dict:
    """
    Compute metrics for prediction.

    Args:
    ----
        pred: The prediction object containing:
            predictions: np.ndarray of shape (n_samples, n_labels)
            label_ids: np.ndarray of shape (n_samples,)

    Returns:
    -------
        The evaluation metrics.

    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)  # ty: ignore[possibly-missing-attribute]

    accuracy = accuracy_score(labels, preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="weighted",
        zero_division=0,
    )

    per_class_precision, per_class_recall, per_class_f1, _ = (
        precision_recall_fscore_support(
            labels,
            preds,
            average=None,
            zero_division=0,
        )
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    for i in range(len(per_class_precision)):
        metrics.update(
            {
                f"precision_class_{i}": per_class_precision[i],
                f"recall_class_{i}": per_class_recall[i],
                f"f1_class_{i}": per_class_f1[i],
            }
        )

    return metrics


def empty_path(path: str, retain: list) -> None:
    """
    Delete files in a directory except those specifically excluded.

    Args:
    ----
        path: The path to the directory containing the files to delete.
        retain: A list of files to exclude from deletion.

    """
    if not os.path.exists(path):
        print(f"Warning: Path {path} does not exist. Skipping cleanup.")
        return

    for entry in os.listdir(path):
        if entry in retain:
            continue

        target = os.path.join(path, entry)
        try:
            if os.path.isfile(target):
                os.remove(target)
            elif os.path.isdir(target):
                shutil.rmtree(target)
        except OSError as e:
            print(f"Warning: Failed to remove {target}: {e}")


def get_optimal_batch_size(gpu_memory_gb: float) -> int:
    """
    Estimate optimal batch size based on available GPU memory.

    Args:
    ----
        gpu_memory_gb: Available GPU memory in gigabytes.

    Returns:
    -------
        Estimated optimal batch size for training.

    """
    return max(1, int((gpu_memory_gb * 0.8 - 2) / 0.5))  # Simplified heuristic


def parse_sagemaker_args() -> tuple[
    ModelArguments, CustomTrainingArguments, SageMakerEnvironmentArguments
]:
    """Parse arguments from SageMaker hyperparameters.json and command line."""
    parser = HfArgumentParser(
        [ModelArguments, CustomTrainingArguments, SageMakerEnvironmentArguments]  # type: ignore[arg-type] HfArgumentParser uses NewType nominal typing which doesn't like our concrete dataclass types
    )

    # First, try to find hyperparameters.json
    hyperparameters_path = "/opt/ml/input/config/hyperparameters.json"
    if os.path.exists(hyperparameters_path):
        with open(hyperparameters_path) as f:
            hyperparameters = json.load(f)

        # Clean the hyperparameters - SageMaker adds quotes to all values
        cleaned_hp = {}
        for k, v in hyperparameters.items():
            val = v
            # Remove any surrounding quotes
            if isinstance(val, str):
                val = val.strip("\"'")
                # Convert to appropriate type if needed
                if val.lower() == "true":
                    val = True
                elif val.lower() == "false":
                    val = False
                elif val.replace(".", "", 1).isdigit():
                    val = float(val) if "." in v else int(val)
            cleaned_hp[k] = val

        model_args, training_args, sagemaker_args = parser.parse_dict(
            cleaned_hp, allow_extra_keys=True
        )
    else:
        model_args, training_args, sagemaker_args = parser.parse_args_into_dataclasses()

    return model_args, training_args, sagemaker_args  # type: ignore[return-value] returns concrete dataclass types


def print_classification_report(
    labels: list, predictions: list, id2label: dict
) -> None:
    """
    Print classification report.

    This helps us understand the performance of the model on each class.

    Args:
    ----
        labels: List of true labels.
        predictions: List of predicted labels.
        id2label: Mapping of label IDs to label names.

    """
    cm = confusion_matrix(labels, predictions, labels=list(id2label.keys()))
    values = np.array(list(id2label.values()))
    pretty = pd.DataFrame(cm, columns=values, index=values)
    print("Confusion Matrix:")
    print(pretty)

    print("\nClassification Report:")
    report = classification_report(labels, predictions, zero_division=0)
    print(report)


def tokenize_in_batches(
    dataset: datasets.Dataset, tokenizer: Callable, batch_size: int = 1000
) -> datasets.Dataset:
    """
    Tokenize dataset in batches to reduce memory pressure.

    Args:
    ----
        dataset: The dataset to tokenize.
        tokenizer: The tokenizer to use.
        batch_size: The number of examples to tokenize at once.

    Returns:
    -------
        The tokenized dataset.

    """
    return dataset.map(
        lambda batch: tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=512
        ),
        batched=True,
        batch_size=batch_size,
    )


def validate_dataset(dataset: datasets.Dataset) -> None:
    """
    Validate dataset has required fields and is not empty.

    Args:
    ----
        dataset: The dataset to validate.

    Raises:
    ------
        ValueError: If the dataset is empty or missing required fields.

    """
    if dataset.num_rows == 0:
        raise ValueError("Dataset is empty")  # noqa TRY003 We don't need a custom exception class here

    if "text" not in dataset.features or "label" not in dataset.features:
        raise ValueError("Dataset missing required fields: text and/or label")  # noqa TRY003 We don't need a custom exception class here

    # Check for empty texts
    empty_texts = sum(1 for item in dataset if not item["text"])
    if empty_texts > 0:
        print(f"Warning: Dataset contains {empty_texts} empty text fields")


def main() -> None:
    """Train the model."""
    model_args, training_args, sagemaker_args = parse_sagemaker_args()

    train_dataset = datasets.load_dataset(
        "json", data_files=os.path.join(sagemaker_args.training_dir, "data.json")
    )["train"]
    test_dataset = datasets.load_dataset(
        "json", data_files=os.path.join(sagemaker_args.test_dir, "data.json")
    )["train"]

    # Validate datasets
    validate_dataset(train_dataset)
    validate_dataset(test_dataset)

    train_labels = pd.DataFrame(set(train_dataset["label"]))
    test_labels = pd.DataFrame(set(test_dataset["label"]))
    all_labels = pd.concat([train_labels, test_labels], ignore_index=True)[0].tolist()
    unique_labels = np.unique(all_labels)

    label2id = {int(v): i for i, v in enumerate(unique_labels)}
    id2label = {i: int(v) for i, v in enumerate(unique_labels)}

    # We do this inline to avoid saving the label2id mapping to disk.
    def map_labels(data: dict) -> dict:
        data["label"] = label2id[data["label"]]
        return data

    # If we're mapping the labels inline, we might as use the same pattern for other map calls.
    def tokenize(data: dict) -> dict:
        return tokenizer(data["text"], truncation=True, padding=True)

    train_dataset = train_dataset.map(map_labels)
    test_dataset = test_dataset.map(map_labels)

    model_name = "bert-base-uncased"
    device = (
        "cuda" if torch.cuda.is_available() and sagemaker_args.n_gpus > 0 else "cpu"
    )

    # Set up training arguments based on hardware availability
    if torch.cuda.is_available() and sagemaker_args.n_gpus > 0:
        # Enable mixed precision for GPU training
        training_args.fp16 = True

        # Set dynamic batch size based on GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        training_args.per_device_train_batch_size = get_optimal_batch_size(gpu_memory)
        print(
            f"Using GPU with {gpu_memory:.2f}GB memory, batch size set to {training_args.per_device_train_batch_size}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=not model_args.download_model
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        id2label=id2label,
        label2id=label2id,
        local_files_only=not model_args.download_model,
    ).to(torch.device(device))

    # Use memory-efficient tokenization
    tokenized_train_dataset = tokenize_in_batches(train_dataset, tokenizer)
    tokenized_test_dataset = tokenize_in_batches(test_dataset, tokenizer)

    training_args.logging_dir = os.path.join(sagemaker_args.output_data_dir, "logs")
    training_args.use_cpu = sagemaker_args.n_gpus == 0
    training_args.output_dir = sagemaker_args.output_data_dir

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=None,
    )

    if os.path.isdir(training_args.checkpoint_dir) and (
        checkpoint_path := get_last_checkpoint(training_args.checkpoint_dir)
    ):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        try:
            trainer.train(resume_from_checkpoint=checkpoint_path)
        except ValueError as e:
            print(f"Error resuming from checkpoint: {e}, starting from scratch")
            trainer.train()
    else:
        print("Training from scratch...")
        trainer.train()

    trainer.save_model(sagemaker_args.model_dir)
    tokenizer.save_pretrained(sagemaker_args.model_dir, legacy_format=True)

    # Add model metadata and versioning
    now = datetime.datetime.now(tz=datetime.UTC)
    model_version = now.strftime("%Y%m%d_%H%M%S")
    model_metadata = {
        "version": model_version,
        "training_args": {
            k: str(v) for k, v in vars(training_args).items() if not k.startswith("_")
        },
        "label_mapping": {str(k): v for k, v in id2label.items()},
        "model_name": model_name,
        "training_date": now.isoformat(),
    }

    # Save metadata
    metadata_path = os.path.join(sagemaker_args.model_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(model_metadata, f, indent=2)

    tarball_name = os.path.join(sagemaker_args.model_dir, "model.tar.gz")
    with tarfile.open(tarball_name, "w:gz") as tar:
        for filename in [
            "config.json",
            "model.safetensors",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "training_args.bin",
            "vocab.txt",
            "metadata.json",
        ]:
            filepath = os.path.join(sagemaker_args.model_dir, filename)
            if os.path.exists(filepath):
                tar.add(filepath, arcname=filename)
            else:
                print(f"Warning: File {filename} not found, skipping from tarball")

    print(f"Model tarball written to {tarball_name}")

    # Remove the unnecessary files to save on S3 storage costs
    empty_path(sagemaker_args.model_dir, ["model.tar.gz"])

    eval_result = trainer.evaluate()
    print("***** Eval results *****")
    for key, value in sorted(eval_result.items()):
        print(f"{key} = {value}")

    predictions = trainer.predict(tokenized_test_dataset).predictions.argmax(axis=-1)  # type: ignore[arg-type, union-attr] Trainer expects torch Dataset but accepts datasets.Dataset, predictions is always ndarray with argmax method
    true_labels = tokenized_test_dataset["label"]

    print_classification_report(true_labels, predictions, id2label)


if __name__ == "__main__":
    main()
