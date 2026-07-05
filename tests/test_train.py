"""Tests for the training utilities."""

__author__ = "Dave Hall <me@davehall.com.au>"
__copyright__ = "Copyright 2024 - 2026, Skwashd Services Pty Ltd https://gata.works"
__license__ = "MIT"

import json
import pathlib
import sys

import datasets
import numpy as np
import pytest
from transformers import EvalPrediction

import train


def test_clean_hyperparameters() -> None:
    """Test quote stripping and type conversion."""
    raw = {
        "num_train_epochs": '"2"',
        "learning_rate": "'3.5'",
        "download_model": "true",
        "fp16": '"false"',
        "model_name": "bert-base-uncased",
        "n_gpus": 1,
    }

    cleaned = train.clean_hyperparameters(raw)

    assert cleaned == {
        "num_train_epochs": 2,
        "learning_rate": 3.5,
        "download_model": True,
        "fp16": False,
        "model_name": "bert-base-uncased",
        "n_gpus": 1,
    }
    assert isinstance(cleaned["num_train_epochs"], int)
    assert isinstance(cleaned["learning_rate"], float)


def test_compute_metrics() -> None:
    """Test the metrics include overall and per class scores."""
    pred = EvalPrediction(
        predictions=np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]),
        label_ids=np.array([0, 1, 0]),
    )

    metrics = train.compute_metrics(pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    for i in range(2):
        assert metrics[f"precision_class_{i}"] == 1.0
        assert metrics[f"recall_class_{i}"] == 1.0
        assert metrics[f"f1_class_{i}"] == 1.0


def test_empty_path(tmp_path: pathlib.Path) -> None:
    """Test everything except the retained files is removed."""
    (tmp_path / "model.tar.gz").write_text("keep")
    (tmp_path / "config.json").write_text("remove")
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    (checkpoint / "optimizer.pt").write_text("remove")

    train.empty_path(str(tmp_path), ["model.tar.gz"])

    assert [entry.name for entry in tmp_path.iterdir()] == ["model.tar.gz"]


def test_empty_path_missing(capsys: pytest.CaptureFixture) -> None:
    """Test a missing directory is skipped with a warning."""
    train.empty_path("/tmp/gata-finetune-does-not-exist", [])

    assert "does not exist" in capsys.readouterr().out


def test_get_optimal_batch_size() -> None:
    """Test the batch size heuristic."""
    assert train.get_optimal_batch_size(24.0) == 34
    assert train.get_optimal_batch_size(0.0) == 1


def test_parse_sagemaker_args_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Test the defaults used when hyperparameters.json is absent."""
    monkeypatch.setattr(
        train, "HYPERPARAMETERS_PATH", str(tmp_path / "hyperparameters.json")
    )
    monkeypatch.setattr(sys, "argv", ["train.py"])

    model_args, training_args, _ = train.parse_sagemaker_args()

    assert model_args.download_model is False
    assert training_args.num_train_epochs == 4
    assert training_args.metric_for_best_model == "f1"


def test_parse_sagemaker_args_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Test values from hyperparameters.json override the defaults."""
    hyperparameters_path = tmp_path / "hyperparameters.json"
    hyperparameters_path.write_text(
        json.dumps({"num_train_epochs": '"2"', "download_model": "true"})
    )
    monkeypatch.setattr(train, "HYPERPARAMETERS_PATH", str(hyperparameters_path))

    model_args, training_args, _ = train.parse_sagemaker_args()

    assert model_args.download_model is True
    assert training_args.num_train_epochs == 2


def test_tokenize_in_batches() -> None:
    """Test tokenised columns are added to the dataset."""

    def fake_tokenizer(texts: list[str], **_kwargs: object) -> dict:
        return {
            "input_ids": [[101, 102]] * len(texts),
            "attention_mask": [[1, 1]] * len(texts),
        }

    dataset = datasets.Dataset.from_dict(
        {"text": ["ticket one", "ticket two", "ticket three"], "label": [0, 1, 0]}
    )

    tokenized = train.tokenize_in_batches(dataset, fake_tokenizer, batch_size=2)

    assert "input_ids" in tokenized.features
    assert "attention_mask" in tokenized.features
    assert tokenized.num_rows == 3


def test_validate_dataset_empty() -> None:
    """Test an empty dataset is rejected."""
    dataset = datasets.Dataset.from_dict({"text": [], "label": []})

    with pytest.raises(ValueError, match="empty"):
        train.validate_dataset(dataset)


def test_validate_dataset_missing_fields() -> None:
    """Test a dataset without a label column is rejected."""
    dataset = datasets.Dataset.from_dict({"text": ["a ticket"]})

    with pytest.raises(ValueError, match="required fields"):
        train.validate_dataset(dataset)


def test_validate_dataset_warns_on_empty_text(capsys: pytest.CaptureFixture) -> None:
    """Test empty text fields trigger a warning without failing."""
    dataset = datasets.Dataset.from_dict({"text": ["a ticket", ""], "label": [0, 1]})

    train.validate_dataset(dataset)

    assert "1 empty text fields" in capsys.readouterr().out
