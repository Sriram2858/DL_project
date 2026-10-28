"""Download and validate the official CLINC150 data_full.json file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
REQUIRED_SPLITS = ("train", "val", "test", "oos_train", "oos_val", "oos_test")


def validate_dataset(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = [name for name in REQUIRED_SPLITS if name not in data]
    if missing:
        raise ValueError(f"Missing required CLINC150 splits: {', '.join(missing)}")
    for split in REQUIRED_SPLITS:
        rows = data[split]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Split {split!r} must be a non-empty list")
        for row in rows:
            if not isinstance(row, list) or len(row) != 2 or not all(isinstance(x, str) for x in row):
                raise ValueError(f"Unexpected record in {split!r}: {row!r}")
    train_labels = {row[1] for row in data["train"]}
    if "oos" in train_labels:
        raise ValueError("The in-domain train split unexpectedly contains the OOS label")
    if len(train_labels) != 150:
        raise ValueError(f"Expected 150 in-domain intents, found {len(train_labels)}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/clinc150/data_full.json"))
    parser.add_argument("--url", default=URL)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        request = Request(args.url, headers={"User-Agent": "VoxAssist dataset downloader"})
        with urlopen(request, timeout=60) as response:
            args.output.write_bytes(response.read())
    data = validate_dataset(args.output)
    print(f"Validated {args.output}")
    for split in REQUIRED_SPLITS:
        print(f"{split:9s}: {len(data[split])}")


if __name__ == "__main__":
    main()
