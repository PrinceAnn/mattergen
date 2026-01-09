"""Utilities for loading teacher feature .npz files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class TeacherFeatures:
    x: np.ndarray  # (N, F) float32
    id_to_row: Dict[str, int]
    meta: dict

    @property
    def dim(self) -> int:
        return int(self.x.shape[1])


def load_teacher_npz(path: str) -> TeacherFeatures:
    data = np.load(path, allow_pickle=True)
    material_id = data["material_id"]
    x = data["x"].astype(np.float32)
    meta = json.loads(str(data["meta"]))

    ids = [str(v) for v in material_id.tolist()]
    id_to_row = {mid: i for i, mid in enumerate(ids)}

    return TeacherFeatures(x=x, id_to_row=id_to_row, meta=meta)


def gather_teacher_batch(tf: TeacherFeatures, material_ids: list[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x_batch, mask) where mask=1 when teacher exists, else 0.

    x_batch: (B, F) float32
    mask: (B,) float32
    """

    f = tf.dim
    xb = np.zeros((len(material_ids), f), dtype=np.float32)
    mask = np.zeros((len(material_ids),), dtype=np.float32)
    for i, mid in enumerate(material_ids):
        j = tf.id_to_row.get(str(mid))
        if j is None:
            continue
        xb[i] = tf.x[j]
        mask[i] = 1.0
    return xb, mask
