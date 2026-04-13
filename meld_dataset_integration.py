from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


EMOTION_TO_ID = {
    "neutral": 0,
    "joy": 1,
    "sadness": 2,
    "anger": 3,
    "fear": 4,
    "surprise": 5,
    "disgust": 6,
}


@dataclass(frozen=True)
class MeldSplit:
    name: str
    source_file: str


SPLITS = [
    MeldSplit("train", "train_sent_emo.csv"),
    MeldSplit("dev", "dev_sent_emo.csv"),
    MeldSplit("test", "test_sent_emo.csv"),
]


def dataset_root() -> Path:
    return Path(__file__).resolve().parent / "datasets" / "MELD"


def source_dir() -> Path:
    return dataset_root() / "data" / "MELD"


def processed_dir() -> Path:
    path = dataset_root() / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_text(text: str) -> str:
    return " ".join((text or "").replace("\n", " ").split())


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def convert_row(row: dict[str, str], split_name: str) -> dict[str, str | int]:
    emotion = (row.get("Emotion") or "").strip().lower()
    if emotion not in EMOTION_TO_ID:
        raise ValueError(f"Unsupported emotion label '{emotion}' in split '{split_name}'")

    return {
        "split": split_name,
        "label": EMOTION_TO_ID[emotion],
        "label_name": emotion,
        "utterance": normalize_text(row.get("Utterance", "")),
        "speaker": (row.get("Speaker") or "").strip(),
        "sentiment": (row.get("Sentiment") or "").strip().lower(),
        "dialogue_id": row.get("Dialogue_ID", "").strip(),
        "utterance_id": row.get("Utterance_ID", "").strip(),
        "season": row.get("Season", "").strip(),
        "episode": row.get("Episode", "").strip(),
        "start_time": row.get("StartTime", "").strip(),
        "end_time": row.get("EndTime", "").strip(),
    }


def write_processed_split(split_name: str, rows: list[dict[str, str | int]]) -> Path:
    output_path = processed_dir() / f"meld_{split_name}_normalized.csv"
    fieldnames = [
        "split",
        "label",
        "label_name",
        "utterance",
        "speaker",
        "sentiment",
        "dialogue_id",
        "utterance_id",
        "season",
        "episode",
        "start_time",
        "end_time",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def summarize_rows(rows: list[dict[str, str | int]]) -> dict[str, object]:
    emotion_counts = Counter(str(row["label_name"]) for row in rows)
    speakers = {str(row["speaker"]) for row in rows if str(row["speaker"]).strip()}
    dialogues = {str(row["dialogue_id"]) for row in rows if str(row["dialogue_id"]).strip()}

    return {
        "rows": len(rows),
        "dialogues": len(dialogues),
        "speakers": len(speakers),
        "emotion_counts": dict(sorted(emotion_counts.items())),
    }


def integrate_meld() -> dict[str, object]:
    root = dataset_root()
    source = source_dir()
    if not source.exists():
        raise FileNotFoundError(f"MELD source directory not found: {source}")

    summary: dict[str, object] = {
        "dataset": "MELD",
        "root": str(root),
        "source_dir": str(source),
        "processed_dir": str(processed_dir()),
        "splits": {},
        "label_mapping": EMOTION_TO_ID,
    }

    for split in SPLITS:
        split_source = source / split.source_file
        rows = [convert_row(row, split.name) for row in load_rows(split_source)]
        output_path = write_processed_split(split.name, rows)
        split_summary = summarize_rows(rows)
        split_summary["source_file"] = str(split_source)
        split_summary["processed_file"] = str(output_path)
        summary["splits"][split.name] = split_summary

    summary_path = processed_dir() / "meld_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_file"] = str(summary_path)
    return summary


def main() -> None:
    summary = integrate_meld()
    print("MELD integration complete.")
    print(f"Processed files: {summary['processed_dir']}")
    for split_name, split_summary in summary["splits"].items():
        print(
            f"{split_name}: rows={split_summary['rows']} "
            f"dialogues={split_summary['dialogues']} "
            f"speakers={split_summary['speakers']}"
        )
    print(f"Summary file: {summary['summary_file']}")


if __name__ == "__main__":
    main()
