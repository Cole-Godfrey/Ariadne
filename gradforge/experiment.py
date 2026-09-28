"""Train the MLP on MNIST and save a reproducible experiment record."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .mnist import load_mnist
from .nn import MLP
from .optim import Adam
from .plot import plot_history
from .tensor import Tensor, cross_entropy


def evaluate(model: MLP, images: np.ndarray, labels: np.ndarray, batch_size: int = 1024) -> tuple[float, float]:
    total_loss = 0.0
    correct = 0
    for start in range(0, len(labels), batch_size):
        end = start + batch_size
        logits = model(Tensor(images[start:end]))
        targets = labels[start:end]
        total_loss += float(cross_entropy(logits, targets).data) * len(targets)
        correct += int(np.count_nonzero(logits.data.argmax(axis=1) == targets))
    return total_loss / len(labels), correct / len(labels)


def train(
    data_dir: Path,
    output_dir: Path,
    seed: int = 42,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    validation_size: int = 5_000,
) -> dict[str, object]:
    if epochs < 1 or batch_size < 1 or not 0 < validation_size < 60_000:
        raise ValueError("epochs and batch size must be positive; validation size must be within (0, 60000)")
    if learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    images, labels, test_images, test_labels = load_mnist(data_dir)
    split_seed, model_seed, shuffle_seed = np.random.SeedSequence(seed).spawn(3)
    split_rng = np.random.default_rng(split_seed)
    model_rng = np.random.default_rng(model_seed)
    shuffle_rng = np.random.default_rng(shuffle_seed)
    permutation = split_rng.permutation(len(labels))
    validation_indices = permutation[:validation_size]
    training_indices = permutation[validation_size:]
    train_images, train_labels = images[training_indices], labels[training_indices]
    val_images, val_labels = images[validation_indices], labels[validation_indices]

    architecture = (784, 256, 128, 10)
    model = MLP(architecture, model_rng)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    best_score = (-1.0, float("-inf"))
    best_weights: list[np.ndarray] = []
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        order = shuffle_rng.permutation(len(train_labels))
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            logits = model(Tensor(train_images[indices]))
            loss = cross_entropy(logits, train_labels[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss, train_accuracy = evaluate(model, train_images, train_labels)
        val_loss, val_accuracy = evaluate(model, val_images, val_labels)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{epochs}: train loss {train_loss:.4f}, acc {train_accuracy:.4%}; "
            f"val loss {val_loss:.4f}, acc {val_accuracy:.4%}",
            flush=True,
        )
        score = (val_accuracy, -val_loss)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_weights = [parameter.data.copy() for parameter in model.parameters()]

    for parameter, weight in zip(model.parameters(), best_weights):
        parameter.data[...] = weight
    test_loss, test_accuracy = evaluate(model, test_images, test_labels)
    summary = {
        "seed": seed,
        "architecture": list(architecture),
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "train_examples": len(train_labels),
        "validation_examples": len(val_labels),
        "test_examples": len(test_labels),
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(history[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    plot_history(history, output_dir / "curves.svg")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        output_dir / "best_model.npz",
        **{f"parameter_{index}": parameter.data for index, parameter in enumerate(model.parameters())},
    )
    print(f"best epoch {best_epoch}; test loss {test_loss:.4f}, test accuracy {test_accuracy:.4%}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--validation-size", type=int, default=5_000)
    arguments = parser.parse_args()
    train(**vars(arguments))


if __name__ == "__main__":
    main()
