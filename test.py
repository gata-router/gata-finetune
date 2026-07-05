#!/usr/bin/env python
"""
End-to-end smoke test for train.py.

Trains the real bert-base-uncased pipeline on a tiny balanced synthetic dataset
and asserts the model trains, serialises to a tarball, and reloads for
inference. This catches breakage from code or dependency changes (e.g. stricter
config validation in newer transformers releases) that import-only checks miss.

Run it locally with:

    uv run ./test.py

The first local run downloads bert-base-uncased into the Hugging Face cache; it
is reused afterwards. In CI this runs inside the built Docker image, where the
model is already baked in and HF_HUB_OFFLINE=1 keeps it off the network -- so the
test never adds a second Hub download (the source of unauthenticated rate limits).
"""

import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent

# Files train.py is expected to bundle into model.tar.gz for downstream serving.
REQUIRED_ARTIFACTS = {
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "metadata.json",
}

# Synthetic, schema-valid sample tickets keyed by a few realistic large integer
# labels. The pipeline is what is under test (not model quality), so a tiny fixed
# corpus is sufficient and keeps the test self-contained -- the real data/
# directory is gitignored and absent in CI.
SAMPLE_TICKETS = {
    16066574298652: "the company website shows a broken image on the homepage",
    16066628349340: "my work laptop keeps freezing and crashing several times a day",
    16066628724508: "the design application crashes on startup every single time",
}
TRAIN_PER_LABEL = 10
TEST_PER_LABEL = 3


def build_subset(dest: Path) -> None:
    """Write a small, balanced synthetic train/test split for the pipeline."""
    splits = {
        "train": (0, TRAIN_PER_LABEL),
        "test": (TRAIN_PER_LABEL, TRAIN_PER_LABEL + TEST_PER_LABEL),
    }
    for name, (start, end) in splits.items():
        split_dir = dest / name
        split_dir.mkdir(parents=True, exist_ok=True)
        with open(split_dir / "data.json", "w") as handle:
            for label, base in SAMPLE_TICKETS.items():
                for i in range(start, end):
                    text = f"{base} this is report number {i} please look into it"
                    handle.write(json.dumps({"text": text, "label": label}) + "\n")


def run_training(work: Path) -> None:
    """Invoke train.py as a subprocess on the subset, on CPU, for one epoch."""
    cmd = [
        sys.executable,
        str(REPO / "train.py"),
        "--download_model",
        "True",
        "--num_train_epochs",
        "1",
        "--training_dir",
        str(work / "train"),
        "--test_dir",
        str(work / "test"),
        "--model_dir",
        str(work / "model"),
        "--output_data_dir",
        str(work / "output"),
        "--checkpoint_dir",
        str(work / "output"),
        "--n_gpus",
        "0",
        "--per_device_train_batch_size",
        "4",
        "--per_device_eval_batch_size",
        "4",
        "--gradient_accumulation_steps",
        "1",
        "--logging_steps",
        "5",
        "--report_to",
        "none",
    ]
    # Inherit the environment so CI's HF_HUB_OFFLINE=1 (set on the container)
    # reaches train.py and forces use of the model baked into the image.
    env = {
        **os.environ,
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    }
    subprocess.run(cmd, check=True, env=env)


def verify_artifacts(work: Path) -> None:
    """Assert the tarball exists and contains every required artifact."""
    tarball = work / "model" / "model.tar.gz"
    assert tarball.exists(), "train.py did not produce model.tar.gz"

    with tarfile.open(tarball) as tar:
        names = set(tar.getnames())
    missing = REQUIRED_ARTIFACTS - names
    assert not missing, f"model.tar.gz is missing required artifacts: {sorted(missing)}"


def verify_reload(work: Path) -> None:
    """Reload the model from the tarball and run a single inference."""
    import torch
    import transformers.utils.logging
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    transformers.utils.logging.set_verbosity_error()

    extract_dir = work / "reload"
    extract_dir.mkdir()
    with tarfile.open(work / "model" / "model.tar.gz") as tar:
        tar.extractall(extract_dir, filter="data")

    tokenizer = AutoTokenizer.from_pretrained(extract_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        extract_dir, local_files_only=True
    )
    model.eval()

    encoded = tokenizer(  # ty: ignore[call-non-callable] transformers 5 types from_pretrained as optional, but it raises rather than returning None
        "password reset request for user account",
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64,
    )
    with torch.no_grad():
        logits = model(**encoded).logits
    prediction = int(logits.argmax(-1))

    assert prediction in model.config.id2label, (
        f"prediction {prediction} not in id2label {model.config.id2label}"
    )
    # transformers/huggingface_hub require these to be JSON-style label maps.
    assert all(isinstance(key, str) for key in model.config.label2id), (
        "label2id keys must be strings"
    )


def main() -> None:
    """Run the end-to-end smoke test in a throwaway directory."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        build_subset(work)
        run_training(work)
        verify_artifacts(work)
        verify_reload(work)
    print("e2e smoke test passed")


if __name__ == "__main__":
    main()
