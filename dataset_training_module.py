from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

MPL_CONFIG_DIR = Path(__file__).resolve().parent / ".matplotlib"
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


EMOTION_LABELS = [
    "Neutral",
    "Happy",
    "Sad",
    "Angry",
    "Fear",
    "Surprise",
    "Disgust",
]


@dataclass
class TrainingConfig:
    samples: int = 1400
    visual_dim: int = 24
    audio_dim: int = 18
    classes: int = 7
    test_ratio: float = 0.2
    learning_rate: float = 0.08
    epochs: int = 60
    batch_size: int = 64
    weight_decay: float = 1e-4
    seed: int = 42
    noise_scale: float = 1.45
    vocab_size: int = 512


class SoftmaxClassifier:
    def __init__(self, input_dim: int, num_classes: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, 0.05, size=(input_dim, num_classes))
        self.bias = np.zeros((1, num_classes), dtype=np.float64)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        logits = features @ self.weights + self.bias
        return self._softmax(logits)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(features), axis=1)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        cfg: TrainingConfig,
    ) -> dict[str, list[float]]:
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        rng = np.random.default_rng(cfg.seed)
        num_samples = x_train.shape[0]
        num_classes = self.bias.shape[1]
        one_hot = np.eye(num_classes, dtype=np.float64)

        for epoch in range(cfg.epochs):
            indices = rng.permutation(num_samples)
            x_epoch = x_train[indices]
            y_epoch = y_train[indices]

            for start in range(0, num_samples, cfg.batch_size):
                end = start + cfg.batch_size
                batch_x = x_epoch[start:end]
                batch_y = y_epoch[start:end]

                probs = self.predict_proba(batch_x)
                target = one_hot[batch_y]
                error = probs - target

                grad_w = (batch_x.T @ error) / len(batch_x)
                grad_b = error.mean(axis=0, keepdims=True)

                grad_w += cfg.weight_decay * self.weights
                self.weights -= cfg.learning_rate * grad_w
                self.bias -= cfg.learning_rate * grad_b

            train_metrics = evaluate_model(self, x_train, y_train)
            val_metrics = evaluate_model(self, x_val, y_val)

            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])

            print(
                f"Epoch {epoch + 1:02d}/{cfg.epochs} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )

        return history


def load_csv_dataset(csv_path: Path, classes: int) -> tuple[np.ndarray, np.ndarray]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    if "label" not in rows[0]:
        raise ValueError("CSV dataset must include a 'label' column.")

    feature_names = [name for name in rows[0].keys() if name != "label"]
    features = np.array(
        [[float(row[name]) for name in feature_names] for row in rows],
        dtype=np.float64,
    )
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)

    if labels.min() < 0 or labels.max() >= classes:
        raise ValueError(f"Labels must be in the range [0, {classes - 1}]")

    return features, labels


def load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def generate_synthetic_multimodal_dataset(cfg: TrainingConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    visual_centers = rng.normal(0.0, 1.15, size=(cfg.classes, cfg.visual_dim))
    audio_centers = rng.normal(0.0, 1.05, size=(cfg.classes, cfg.audio_dim))

    features: list[np.ndarray] = []
    labels: list[int] = []
    samples_per_class = cfg.samples // cfg.classes

    for class_index in range(cfg.classes):
        for _ in range(samples_per_class):
            visual = visual_centers[class_index] + rng.normal(0.0, cfg.noise_scale, cfg.visual_dim)
            audio = audio_centers[class_index] + rng.normal(0.0, cfg.noise_scale, cfg.audio_dim)
            features.append(np.concatenate([visual, audio]))
            labels.append(class_index)

    x = np.vstack(features).astype(np.float64)
    y = np.array(labels, dtype=np.int64)

    indices = rng.permutation(len(y))
    return x[indices], y[indices]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def row_to_text_fields(row: dict[str, str]) -> str:
    parts = [
        row.get("utterance", ""),
        row.get("speaker", ""),
        row.get("sentiment", ""),
    ]
    return " ".join(part for part in parts if part)


def build_hashed_text_features(rows: list[dict[str, str]], vocab_size: int) -> np.ndarray:
    features = np.zeros((len(rows), vocab_size + 4), dtype=np.float64)
    sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}

    for row_index, row in enumerate(rows):
        tokens = tokenize(row_to_text_fields(row))
        for token in tokens:
            bucket = hash(token) % vocab_size
            features[row_index, bucket] += 1.0

        token_count = len(tokenize(row.get("utterance", "")))
        speaker_len = len((row.get("speaker", "") or "").strip())
        sentiment = (row.get("sentiment", "") or "").strip().lower()

        features[row_index, vocab_size] = float(token_count)
        features[row_index, vocab_size + 1] = float(speaker_len)
        features[row_index, vocab_size + 2] = 1.0 if "?" in row.get("utterance", "") else 0.0
        features[row_index, vocab_size + 3] = float(sentiment_map.get(sentiment, 0))

    return features


def load_meld_dataset(cfg: TrainingConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    processed_root = Path(__file__).resolve().parent / "datasets" / "MELD" / "processed"
    train_rows = load_csv_rows(processed_root / "meld_train_normalized.csv")
    dev_rows = load_csv_rows(processed_root / "meld_dev_normalized.csv")
    test_rows = load_csv_rows(processed_root / "meld_test_normalized.csv")

    x_train = build_hashed_text_features(train_rows, cfg.vocab_size)
    x_dev = build_hashed_text_features(dev_rows, cfg.vocab_size)
    x_test = build_hashed_text_features(test_rows, cfg.vocab_size)

    y_train = np.array([int(row["label"]) for row in train_rows], dtype=np.int64)
    y_dev = np.array([int(row["label"]) for row in dev_rows], dtype=np.int64)
    y_test = np.array([int(row["label"]) for row in test_rows], dtype=np.int64)

    return x_train, y_train, x_dev, y_dev, x_test, y_test


def split_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    test_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    boundary = int(len(labels) * (1.0 - test_ratio))
    x_train = features[:boundary]
    y_train = labels[:boundary]
    x_test = features[boundary:]
    y_test = labels[boundary:]
    return x_train, x_test, y_train, y_test


def standardize_features(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def compute_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probabilities[np.arange(len(labels)), labels], 1e-9, 1.0)
    return float(-np.log(clipped).mean())


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float((predictions == labels).mean())


def confusion_matrix(predictions: np.ndarray, labels: np.ndarray, classes: int) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for true_label, pred_label in zip(labels, predictions):
        matrix[true_label, pred_label] += 1
    return matrix


def evaluate_model(
    model: SoftmaxClassifier,
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    probabilities = model.predict_proba(features)
    predictions = probabilities.argmax(axis=1)
    return {
        "loss": compute_loss(probabilities, labels),
        "accuracy": compute_accuracy(predictions, labels),
        "predictions": predictions,
        "probabilities": probabilities,
    }


def save_learning_curves(history: dict[str, list[float]], output_dir: Path) -> Path:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2.2)
    axes[0].plot(epochs, history["val_loss"], label="Validation Loss", linewidth=2.2)
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="Train Accuracy", linewidth=2.2)
    axes[1].plot(epochs, history["val_accuracy"], label="Validation Accuracy", linewidth=2.2)
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure_path = output_dir / "learning_curves.png"
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def save_confusion_matrix(matrix: np.ndarray, output_dir: Path, labels: list[str]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Test Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")

    figure_path = output_dir / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def save_class_distribution(labels: np.ndarray, output_dir: Path) -> Path:
    counts = np.bincount(labels, minlength=len(EMOTION_LABELS))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(EMOTION_LABELS, counts, color="#4C78A8")
    ax.set_title("Dataset Class Distribution")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Samples")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)

    figure_path = output_dir / "class_distribution.png"
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def ensure_output_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / "outputs" / f"training_run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_pipeline(dataset_path: Path | None, cfg: TrainingConfig, output_root: Path) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_root)

    if dataset_path is None:
        features, labels = generate_synthetic_multimodal_dataset(cfg)
        dataset_name = "synthetic_multimodal"
    else:
        features, labels = load_csv_dataset(dataset_path, cfg.classes)
        dataset_name = dataset_path.stem

    x_train, x_test, y_train, y_test = split_dataset(features, labels, cfg.test_ratio)
    x_train, x_test, mean, std = standardize_features(x_train, x_test)

    model = SoftmaxClassifier(input_dim=x_train.shape[1], num_classes=cfg.classes, seed=cfg.seed)
    history = model.fit(x_train, y_train, x_test, y_test, cfg)

    train_metrics = evaluate_model(model, x_train, y_train)
    test_metrics = evaluate_model(model, x_test, y_test)
    matrix = confusion_matrix(test_metrics["predictions"], y_test, cfg.classes)

    learning_curves_path = save_learning_curves(history, output_dir)
    confusion_matrix_path = save_confusion_matrix(matrix, output_dir, EMOTION_LABELS[: cfg.classes])
    class_distribution_path = save_class_distribution(labels, output_dir)

    metrics = {
        "dataset": dataset_name,
        "generated_at": datetime.now().isoformat(),
        "config": asdict(cfg),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "feature_dim": int(x_train.shape[1]),
        "train_accuracy": round(float(train_metrics["accuracy"]), 4),
        "test_accuracy": round(float(test_metrics["accuracy"]), 4),
        "train_loss": round(float(train_metrics["loss"]), 4),
        "test_loss": round(float(test_metrics["loss"]), 4),
        "confusion_matrix": matrix.tolist(),
        "artifacts": {
            "learning_curves": str(learning_curves_path),
            "confusion_matrix": str(confusion_matrix_path),
            "class_distribution": str(class_distribution_path),
        },
        "normalization": {
            "mean_shape": list(mean.shape),
            "std_shape": list(std.shape),
        },
    }

    metrics_path = output_dir / "metrics.json"
    model_path = output_dir / "softmax_model.npz"
    np.savez(model_path, weights=model.weights, bias=model.bias)
    metrics["artifacts"]["model_weights"] = str(model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nTraining complete.")
    print(f"Dataset: {dataset_name}")
    print(f"Train accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Test accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"Metrics file:   {metrics_path}")
    print(f"Graphs saved in {output_dir}")

    return metrics


def run_meld_pipeline(cfg: TrainingConfig, output_root: Path) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_root)
    x_train, y_train, x_dev, y_dev, x_test, y_test = load_meld_dataset(cfg)
    x_train, x_dev, mean, std = standardize_features(x_train, x_dev)
    x_test = (x_test - mean) / std

    model = SoftmaxClassifier(input_dim=x_train.shape[1], num_classes=cfg.classes, seed=cfg.seed)
    history = model.fit(x_train, y_train, x_dev, y_dev, cfg)

    train_metrics = evaluate_model(model, x_train, y_train)
    dev_metrics = evaluate_model(model, x_dev, y_dev)
    test_metrics = evaluate_model(model, x_test, y_test)
    matrix = confusion_matrix(test_metrics["predictions"], y_test, cfg.classes)

    learning_curves_path = save_learning_curves(history, output_dir)
    confusion_matrix_path = save_confusion_matrix(matrix, output_dir, EMOTION_LABELS[: cfg.classes])
    class_distribution_path = save_class_distribution(
        np.concatenate([y_train, y_dev, y_test]),
        output_dir,
    )

    metrics = {
        "dataset": "meld",
        "generated_at": datetime.now().isoformat(),
        "config": asdict(cfg),
        "train_samples": int(len(y_train)),
        "dev_samples": int(len(y_dev)),
        "test_samples": int(len(y_test)),
        "feature_dim": int(x_train.shape[1]),
        "train_accuracy": round(float(train_metrics["accuracy"]), 4),
        "dev_accuracy": round(float(dev_metrics["accuracy"]), 4),
        "test_accuracy": round(float(test_metrics["accuracy"]), 4),
        "train_loss": round(float(train_metrics["loss"]), 4),
        "dev_loss": round(float(dev_metrics["loss"]), 4),
        "test_loss": round(float(test_metrics["loss"]), 4),
        "confusion_matrix": matrix.tolist(),
        "artifacts": {
            "learning_curves": str(learning_curves_path),
            "confusion_matrix": str(confusion_matrix_path),
            "class_distribution": str(class_distribution_path),
        },
        "normalization": {
            "mean_shape": list(mean.shape),
            "std_shape": list(std.shape),
        },
    }

    metrics_path = output_dir / "metrics.json"
    model_path = output_dir / "softmax_model.npz"
    np.savez(model_path, weights=model.weights, bias=model.bias)
    metrics["artifacts"]["model_weights"] = str(model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nTraining complete.")
    print("Dataset: MELD")
    print(f"Train accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Dev accuracy:   {metrics['dev_accuracy']:.4f}")
    print(f"Test accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"Metrics file:   {metrics_path}")
    print(f"Graphs saved in {output_dir}")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and test a lightweight multimodal emotion classifier and save matplotlib graphs."
    )
    parser.add_argument("--dataset-path", type=Path, default=None, help="Optional CSV path with numeric features and a 'label' column.")
    parser.add_argument("--dataset", choices=["synthetic", "meld"], default="synthetic", help="Dataset source to use.")
    parser.add_argument("--samples", type=int, default=1400, help="Number of synthetic samples to generate when no CSV is provided.")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.08, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainingConfig(
        samples=args.samples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    if args.dataset == "meld":
        run_meld_pipeline(cfg, Path(__file__).resolve().parent)
    else:
        run_pipeline(args.dataset_path, cfg, Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
