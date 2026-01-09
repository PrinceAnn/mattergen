"""Dataset adapter for MP-20 CSV splits used in distillation.

MP-20 split CSV format provides:
- material_id
- cif (full CIF contents as string)
- target property columns

We reuse the graph construction code from original CrysXPP (CGCNN style):
- atom features from atom_init.json (JSON element embeddings)
- neighbor features from GaussianDistance expansion of distances

This dataset returns:
  (input_tuple, target, material_id)

Where input_tuple matches CrysXPP's `CrystalGraphConvNet.forward` signature:
  (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)

"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset


class GaussianDistance:
    def __init__(self, dmin: float, dmax: float, step: float, var: float | None = None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        self.var = step if var is None else var

    def expand(self, distances: np.ndarray) -> np.ndarray:
        return np.exp(-((distances[..., np.newaxis] - self.filter) ** 2) / (self.var**2))


class AtomCustomJSONInitializer:
    def __init__(self, elem_embedding_file: str):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value in elem_embedding.items()}
        self.embedding = {k: np.array(v, dtype=float) for k, v in elem_embedding.items()}

    def get_atom_fea(self, atom_type: int) -> np.ndarray:
        return self.embedding[int(atom_type)]


@dataclass(frozen=True)
class GraphConfig:
    max_num_nbr: int = 12
    radius: float = 8.0
    dmin: float = 0.0
    step: float = 0.2


class MP20CSVDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        atom_init_json: str,
        target_property: str,
        graph_cfg: GraphConfig = GraphConfig(),
        limit: int | None = None,
    ):
        self.df = pd.read_csv(csv_path)
        if limit is not None:
            self.df = self.df.iloc[:limit]

        if target_property not in self.df.columns:
            raise KeyError(f"target_property={target_property!r} not found in CSV columns")

        self.target_property = target_property
        self.graph_cfg = graph_cfg
        self.ari = AtomCustomJSONInitializer(atom_init_json)
        self.gdf = GaussianDistance(dmin=graph_cfg.dmin, dmax=graph_cfg.radius, step=graph_cfg.step)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        material_id = str(row["material_id"])
        target = float(row[self.target_property])
        cif_str = str(row["cif"])

        crystal = Structure.from_str(cif_str, fmt="cif")

        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))])
        atom_fea = torch.tensor(atom_fea, dtype=torch.float32)

        all_nbrs = crystal.get_all_neighbors(self.graph_cfg.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

        nbr_fea_idx: List[List[int]] = []
        nbr_fea: List[List[float]] = []
        for nbr in all_nbrs:
            if len(nbr) < self.graph_cfg.max_num_nbr:
                nbr_fea_idx.append([int(x[2]) for x in nbr] + [0] * (self.graph_cfg.max_num_nbr - len(nbr)))
                nbr_fea.append([float(x[1]) for x in nbr] + [self.graph_cfg.radius + 1.0] * (self.graph_cfg.max_num_nbr - len(nbr)))
            else:
                nbr_fea_idx.append([int(x[2]) for x in nbr[: self.graph_cfg.max_num_nbr]])
                nbr_fea.append([float(x[1]) for x in nbr[: self.graph_cfg.max_num_nbr]])

        nbr_fea_idx_arr = np.array(nbr_fea_idx, dtype=np.int64)
        nbr_fea_arr = np.array(nbr_fea, dtype=np.float32)

        nbr_fea_gdf = self.gdf.expand(nbr_fea_arr)

        nbr_fea_t = torch.tensor(nbr_fea_gdf, dtype=torch.float32)
        nbr_fea_idx_t = torch.tensor(nbr_fea_idx_arr, dtype=torch.long)

        target_t = torch.tensor([target], dtype=torch.float32)

        # We keep crystal_atom_idx construction in collate_fn (batch-level).
        return (atom_fea, nbr_fea_t, nbr_fea_idx_t), target_t, material_id


def collate_pool(dataset_list):
    """CrysXPP-compatible collate.

    Returns:
      (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx), targets, material_ids
    """

    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx = []
    batch_target = []
    batch_ids = []

    base_idx = 0
    for (atom_fea, nbr_fea, nbr_fea_idx), target, material_id in dataset_list:
        n_i = atom_fea.shape[0]
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        crystal_atom_idx.append(torch.arange(n_i, dtype=torch.long) + base_idx)
        batch_target.append(target)
        batch_ids.append(material_id)
        base_idx += n_i

    return (
        torch.cat(batch_atom_fea, dim=0),
        torch.cat(batch_nbr_fea, dim=0),
        torch.cat(batch_nbr_fea_idx, dim=0),
        crystal_atom_idx,
    ), torch.stack(batch_target, dim=0), batch_ids
