from __future__ import annotations

import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

TRACKED_LIBRARIES = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "scikit-learn",
    "numpy",
    "pandas",
    "nltk",
]


def load_yaml_config(config_path: str = "configs/base.yaml") -> dict:

    """
    Load a YAML config file
    """

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    

def load_processed_split(
        path: str | Path,
        required_columns: list[str] | None = None,
) -> pd.DataFrame:
    
    """
    Load one processed CSV split and validate a minimal schema:
    - cast text to str
    - cast label to int if present
    - add example_id if missing
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed split not found: {path}")
    
    df = pd.read_csv(path)

    required_columns = required_columns or ["text", "label"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {missing}")
    
    out = df.copy()

    if "text" in out.columns:
        out["text"] = out["text"].astype(str)

    if "label" in out.columns:
        out["label"] = out["label"].astype(int)

    if "example_id" not in out.columns:
        out.insert(0, "example_id", range(len(out)))

    return out


def word_count(texts: pd.Series) -> pd.Series:

    """
    The single definition of text length: number of whitespace-separated words.
    """

    return texts.astype(str).str.split().str.len().astype(int)


def get_commit_hash() -> str:

    """
    Hash of the current git commit (HEAD).
    """

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def get_library_versions() -> dict[str, str]:

    """
    Python, platform and versions of the main libraries.
    """

    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for library in TRACKED_LIBRARIES:
        versions[library] = version(library)
    return versions


def save_json(data: dict[str, Any], output_path: str | Path) -> None:

    """
    Save a dictionary to JSON.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
