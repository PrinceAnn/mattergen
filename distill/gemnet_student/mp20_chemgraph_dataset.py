"""MP-20 CSV dataset -> ChemGraph Batch for GemNet-based students.

Each row provides:
- material_id
- cif (full CIF text)
- target property column

We parse the CIF into a pymatgen Structure and convert to a `ChemGraph` with:
- pos: fractional coords in [0,1)
- cell: (1,3,3)
- atomic_numbers: (N_atoms,)
- num_atoms, num_nodes

We keep the loader resilient by allowing unparsable CIF rows to be skipped at
collate time (they're rare but do exist in real-world dumps).

"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
import torch
from pymatgen.core import Structure
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from mattergen.common.data.chemgraph import ChemGraph


@dataclass(frozen=True)
class DatasetConfig:
    target_property: str
    limit: int | None = None


def chemgraph_from_structure(structure: Structure) -> ChemGraph:
    num_atoms = torch.tensor(structure.num_sites)
    return ChemGraph(
        pos=torch.from_numpy(structure.frac_coords).float() % 1.0,
        cell=torch.tensor(structure.lattice.matrix, dtype=torch.float32).unsqueeze(0),
        atomic_numbers=torch.tensor(structure.atomic_numbers, dtype=torch.long),
        num_atoms=num_atoms,
        num_nodes=num_atoms,
    )


class MP20ChemGraphDataset(Dataset):
    def __init__(self, csv_path: str, cfg: DatasetConfig):
        self.df = pd.read_csv(csv_path)
        if cfg.limit is not None:
            self.df = self.df.iloc[: cfg.limit]

        if cfg.target_property not in self.df.columns:
            raise KeyError(
                f"target_property={cfg.target_property!r} not found in CSV columns ({list(self.df.columns)[:20]}...)"
            )

        self.target_property = cfg.target_property

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        material_id = str(row["material_id"])
        cif_str = str(row["cif"])
        target = float(row[self.target_property])

        try:
            structure = Structure.from_str(cif_str, fmt="cif")
            graph = chemgraph_from_structure(structure)
        except Exception:
            # Let collate() skip this row.
            graph = None

        target_t = torch.tensor([target], dtype=torch.float32)
        return graph, target_t, material_id


def collate_chemgraphs(batch_list):
    """Collate that skips items where CIF parsing failed."""

    graphs: List[ChemGraph] = []
    targets: List[torch.Tensor] = []
    ids: List[str] = []

    for graph, target, material_id in batch_list:
        if graph is None:
            continue
        graphs.append(graph)
        targets.append(target)
        ids.append(material_id)

    if not graphs:
        # All rows in this minibatch were invalid; upstream should handle by continuing.
        return None

    batch = Batch.from_data_list(graphs)
    target_t = torch.stack(targets, dim=0)
    return batch, target_t, ids
