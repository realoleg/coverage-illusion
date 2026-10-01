from __future__ import annotations

import pandas as pd


def apply_condition(df: pd.DataFrame, condition: str) -> pd.Series:

    """
    Texts of df under the given condition, aligned with df rows.
    Only 'clean' exists so far; the degradations are added in step 3.
    """

    if condition == "clean":
        return df["text"].astype(str)

    raise ValueError(f"Unknown condition: {condition}")
