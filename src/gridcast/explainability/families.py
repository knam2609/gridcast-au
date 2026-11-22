"""Feature families: the feature catalogue's groups of a variant (m4_plan.md §2.6).

Correlated features share SHAP credit (several temperature terms, the demand lags), so
families are the main unit of every summary. For variant C they are exactly the six of
the plan: lead time; calendar at the target; calendar 24 h before; demand at the issue
time; demand at the same time yesterday and last week; weather at the target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gridcast.features import catalogue


@dataclass(frozen=True, slots=True)
class Families:
    keys: tuple[str, ...]  # catalogue group keys, in catalogue order
    titles: tuple[str, ...]
    columns: tuple[str, ...]  # the variant's feature columns, in order
    index: np.ndarray  # (n_columns,) int: family position of each column

    def sums(self, contributions: np.ndarray) -> np.ndarray:
        """``(n, n_families)`` sums of ``(n, n_columns)`` contributions."""
        if contributions.ndim != 2 or contributions.shape[1] != len(self.columns):
            raise ValueError("contributions must be (rows, the variant's feature columns)")
        out = np.zeros((len(contributions), len(self.keys)), dtype=np.float64)
        for k in range(len(self.keys)):
            out[:, k] = contributions[:, self.index == k].sum(axis=1)
        return out

    def of(self, column: str) -> str:
        return self.keys[int(self.index[self.columns.index(column)])]


def families(variant_id: str) -> Families:
    """The families of ``variant_id``: one per catalogue group, every column in exactly one."""
    spec = catalogue.variant(variant_id)
    keys, titles, index = [], [], []
    for k, key in enumerate(spec.groups):
        group = catalogue.group(key)
        keys.append(key)
        titles.append(group.title)
        index += [k] * len(group.features)
    columns = tuple(catalogue.feature_names(variant_id))
    if len(index) != len(columns):
        raise AssertionError("catalogue groups and feature names disagree")
    return Families(tuple(keys), tuple(titles), columns, np.asarray(index, dtype=np.int64))
