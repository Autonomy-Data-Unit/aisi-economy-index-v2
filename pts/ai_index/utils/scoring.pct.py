# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # OnetScoreSet
#
# Standard output format for O\*NET occupation-level score nodes.
# All score nodes produce an `OnetScoreSet`, which is a validated
# DataFrame with `onet_code` + float score columns in [0, 1].

# %%
#|default_exp utils.scoring

# %%
#|export
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class OnetScoreSet:
    """Standard output from any O*NET score node.

    Each score node computes one or more float columns per O*NET occupation
    code. All values must be in [0, 1] range.
    """
    name: str
    scores: pd.DataFrame

    def validate(self):
        """Raise ValueError if the score set is malformed."""
        if "onet_code" not in self.scores.columns:
            raise ValueError(f"OnetScoreSet '{self.name}': missing 'onet_code' column")
        if self.scores["onet_code"].duplicated().any():
            n_dups = self.scores["onet_code"].duplicated().sum()
            raise ValueError(f"OnetScoreSet '{self.name}': {n_dups} duplicate onet_code values")
        score_cols = [c for c in self.scores.columns if c != "onet_code"]
        if not score_cols:
            raise ValueError(f"OnetScoreSet '{self.name}': no score columns")
        for col in score_cols:
            if not pd.api.types.is_float_dtype(self.scores[col]):
                raise ValueError(f"OnetScoreSet '{self.name}': column '{col}' is not float")
            if self.scores[col].isna().any():
                n_na = self.scores[col].isna().sum()
                raise ValueError(f"OnetScoreSet '{self.name}': column '{col}' has {n_na} NaN values")
            vmin = self.scores[col].min()
            vmax = self.scores[col].max()
            if vmin < -0.001 or vmax > 1.001:
                raise ValueError(
                    f"OnetScoreSet '{self.name}': column '{col}' has values outside [0, 1] "
                    f"(min={vmin:.4f}, max={vmax:.4f})"
                )

    def save(self, output_dir: Path):
        """Write to the standard location: {output_dir}/scores.csv"""
        self.validate()
        self.scores.to_csv(output_dir / "scores.csv", index=False)

    @staticmethod
    def load(path: Path) -> pd.DataFrame:
        """Load a scores.csv file."""
        return pd.read_csv(path)
