from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ._types import (
    AbundanceTable,
    EmptyAbundanceError,
    InsufficientReadsError,
    MINIMUM_READS_PER_SAMPLE,
    SPECIES_RESOLVED_GENERA,
)


def filter_low_read_samples(
    counts: NDArray[np.int64],
    minimum_reads: int = MINIMUM_READS_PER_SAMPLE,
) -> NDArray[np.bool_]:
    if counts.ndim != 2:
        raise InsufficientReadsError("counts must be a (samples, taxa) matrix")
    if counts.size == 0:
        raise EmptyAbundanceError("count matrix is empty")
    return counts.sum(axis=1) >= minimum_reads


def counts_to_relative_abundance(counts: NDArray[np.int64]) -> NDArray[np.float64]:
    totals = counts.sum(axis=1, keepdims=True).astype(np.float64)
    if np.any(totals == 0.0):
        raise EmptyAbundanceError("at least one sample has zero reads")
    return counts.astype(np.float64) / totals


def rollup_to_genus(
    abundances: NDArray[np.float64],
    species_taxa: Sequence[str],
    genus_lookup: Mapping[str, str],
) -> AbundanceTable:
    if abundances.shape[1] != len(species_taxa):
        raise EmptyAbundanceError("abundance columns and species labels disagree")
    rolled: dict[str, NDArray[np.float64]] = {}
    n_samples = abundances.shape[0]
    for column_index, species in enumerate(species_taxa):
        genus = genus_lookup[species]
        if genus in SPECIES_RESOLVED_GENERA:
            label = species
        else:
            label = genus
        accumulator = rolled.setdefault(label, np.zeros(n_samples, dtype=np.float64))
        accumulator += abundances[:, column_index]
    ordered = sorted(rolled.keys())
    out = np.stack([rolled[name] for name in ordered], axis=1)
    return AbundanceTable(abundances=out, taxa=tuple(ordered))


def taxon_relative_abundance(table: AbundanceTable, name: str) -> NDArray[np.float64]:
    if name not in table.taxa:
        return np.zeros(table.abundances.shape[0], dtype=np.float64)
    column_index = table.taxa.index(name)
    return table.abundances[:, column_index].copy()


def restrict_to_taxa(table: AbundanceTable, names: Sequence[str]) -> AbundanceTable:
    name_to_index = {name: index for index, name in enumerate(table.taxa)}
    n_samples = table.abundances.shape[0]
    rows: list[NDArray[np.float64]] = []
    kept: list[str] = []
    for name in names:
        if name in name_to_index:
            rows.append(table.abundances[:, name_to_index[name]])
            kept.append(name)
    if not rows:
        return AbundanceTable(abundances=np.zeros((n_samples, 0), dtype=np.float64), taxa=())
    return AbundanceTable(abundances=np.stack(rows, axis=1), taxa=tuple(kept))


__all__ = [
    "filter_low_read_samples",
    "counts_to_relative_abundance",
    "rollup_to_genus",
    "taxon_relative_abundance",
    "restrict_to_taxa",
]
