from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


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


def add_text_length_features(
        df: pd.DataFrame,
        text_column: str = "text",
) -> pd.DataFrame:
    
    """
    Add text length features using the given text column.
    """

    out = df.copy()
    text_series = out[text_column].astype(str)

    out["text_length_chars"] = text_series.str.len()
    out["text_length_words"] = text_series.str.split().str.len()

    return out
