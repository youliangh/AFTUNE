"""Fetch and freeze the exact MentalChat16K inputs used by the experiment.

This is an explicit network step. The experiment runner itself only reads a
previously frozen local JSONL and verifies its manifest hash.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from urllib.request import urlopen

from .config import DEFAULT_DATASET_PATH, DEFAULT_DATASET_REVISION


DATASET = "ShenLab/MentalChat16K"
FILES = (
    ("Synthetic_Data_10K.csv", "52273a1d682986013dd6ad077be8a79075bb1877a022cbb33d170527b7fdadea"),
    ("Interview_Data_6K.csv", "7a2192299807ab2aa202a51dcc5b8b4cb35d1ee6b6eb38e9573be0aa51dc4992"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_checked(destination: Path, filename: str, expected: str) -> None:
    target = destination / filename
    if target.exists():
        actual = _sha256(target)
        if actual != expected:
            raise ValueError(f"Existing {target} has SHA256 {actual}, expected {expected}")
        return
    url = (f"https://huggingface.co/datasets/{DATASET}/resolve/"
           f"{DEFAULT_DATASET_REVISION}/{filename}?download=true")
    temporary = destination / f".{filename}.{os.getpid()}.part"
    try:
        with urlopen(url, timeout=120) as source, temporary.open("wb") as sink:
            shutil.copyfileobj(source, sink, length=4 * 1024 * 1024)
        actual = _sha256(temporary)
        if actual != expected:
            raise ValueError(f"Downloaded {filename} has SHA256 {actual}, expected {expected}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def fetch_dataset(destination: str | Path | None = None) -> Path:
    """Return a local, checksum-verified JSONL with one ``input`` per line."""
    root = Path(destination).expanduser().resolve() if destination else Path(DEFAULT_DATASET_PATH).parent
    root.mkdir(parents=True, exist_ok=True)
    for filename, expected in FILES:
        _fetch_checked(root, filename, expected)

    output = root / "inputs.jsonl"
    manifest_path = root / "manifest.json"
    if output.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("revision") == DEFAULT_DATASET_REVISION and
                manifest.get("source_sha256") == dict(FILES) and
                manifest.get("derived_sha256") == _sha256(output) and
                manifest.get("rows") == sum(1 for _ in output.open(encoding="utf-8"))):
            return output
        raise ValueError("Existing input JSONL differs from its frozen manifest")

    temporary = root / f".inputs.jsonl.{os.getpid()}.part"
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as sink:
            for filename, _expected in FILES:
                with (root / filename).open("r", encoding="utf-8-sig", newline="") as source:
                    for source_row, row in enumerate(csv.DictReader(source)):
                        value = row.get("input")
                        if not isinstance(value, str) or not value.strip():
                            continue
                        sink.write(json.dumps({
                            "input": value,
                            "source_file": filename,
                            "source_row": source_row,
                        }, ensure_ascii=False) + "\n")
                        count += 1
        derived_hash = _sha256(temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    manifest = {
        "dataset": DATASET,
        "revision": DEFAULT_DATASET_REVISION,
        "source_sha256": dict(FILES),
        "derived_file": "inputs.jsonl",
        "derived_sha256": derived_hash,
        "rows": count,
        "derivation": "CSV input field, nonempty only, source order Synthetic then Interview; UTF-8 JSONL",
    }
    temporary_manifest = root / f".manifest.json.{os.getpid()}.part"
    try:
        temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=None)
    args = parser.parse_args(argv)
    output = fetch_dataset(args.destination)
    print(f"{output}: SHA256={_sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
