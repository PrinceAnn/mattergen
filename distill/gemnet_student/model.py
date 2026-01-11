"""GemNet student model for distillation.

Design goal: property prediction should NOT depend on teacher diffusion times.

We therefore split the student into:
    1) A *base* GemNetT forward pass run once per structure with a fixed latent
         conditioning vector z=0 (no dependency on t_values).
    2) A supervised regression head on the base representation.
    3) A distillation projection head mapping base_repr -> teacher_feature_dim.

The teacher feature vector may still encode knowledge from multiple diffusion
times/blocks, but the student predictor itself stays time-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from mattergen.common.gemnet.gemnet import GemNetT
from mattergen.common.gemnet.layers.embedding_block import AtomEmbedding


def pool_h_per_crystal(h: torch.Tensor, batch_idx: torch.Tensor, num_graphs: int) -> torch.Tensor:
    # mean pool over atoms for each crystal in batch
    out = torch.zeros((num_graphs, h.shape[-1]), device=h.device, dtype=h.dtype)
    out.scatter_add_(0, batch_idx[:, None].expand(-1, h.shape[-1]), h)
    counts = torch.bincount(batch_idx, minlength=num_graphs).clamp(min=1).to(h.dtype)
    out = out / counts[:, None]
    return out


@dataclass(frozen=True)
class StudentConfig:
    # GemNet hyperparams (aligned with mattergen-base defaults/config)
    emb_size: int = 512
    latent_dim: int = 512
    num_blocks: int = 4
    cutoff: float = 7.0
    max_neighbors: int = 50
    max_cell_images_per_dim: int = 5
    otf_graph: bool = True
    regress_stress: bool = True


class GemNetStudent(nn.Module):
    def __init__(
        self,
        teacher_feature_dim: Optional[int] = None,
        blocks: Optional[List[int]] = None,
        cfg: StudentConfig = StudentConfig(),
        out_dim: int = 1,
    ):
        super().__init__()

        if teacher_feature_dim is not None and teacher_feature_dim <= 0:
            raise ValueError("teacher_feature_dim must be positive when provided")

        self.blocks = None if blocks is None else sorted(set(int(b) for b in blocks))

        self.gemnet = GemNetT(
            atom_embedding=AtomEmbedding(emb_size=cfg.emb_size, with_mask_type=True),
            num_targets=1,
            latent_dim=cfg.latent_dim,
            num_blocks=cfg.num_blocks,
            emb_size_atom=cfg.emb_size,
            emb_size_edge=cfg.emb_size,
            cutoff=cfg.cutoff,
            max_neighbors=cfg.max_neighbors,
            max_cell_images_per_dim=cfg.max_cell_images_per_dim,
            otf_graph=cfg.otf_graph,
            regress_stress=cfg.regress_stress,
        )

        self.hidden_dim = int(cfg.emb_size)
        self.latent_dim = int(cfg.latent_dim)
        self.base_repr_dim = self.hidden_dim
        self.teacher_feature_dim = None if teacher_feature_dim is None else int(teacher_feature_dim)

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(self.base_repr_dim, max(1, self.base_repr_dim // 2)),
            nn.SiLU(),
            nn.Linear(max(1, self.base_repr_dim // 2), out_dim),
        )

        # Optional distillation projection: base_repr -> teacher feature vector
        self.distill_head = None
        if self.teacher_feature_dim is not None:
            self.distill_head = nn.Linear(self.base_repr_dim, self.teacher_feature_dim)

    def forward(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (pred, repr).

        - If distillation is enabled (teacher_feature_dim provided): repr is the
          projected distillation vector with shape (B, teacher_feature_dim).
        - Otherwise: repr is the base representation with shape (B, base_repr_dim).

        batch must have fields used by GemNetT:
        - pos (frac coords)
        - cell
        - atomic_numbers
        - num_atoms
        - batch (atom->graph indices)
        """

        bsz = int(batch.num_graphs)
        device = batch.pos.device

        all_blocks = self.gemnet.int_blocks
        if self.blocks is None:
            blocks = [len(all_blocks) - 1]
        else:
            blocks = self.blocks
            if min(blocks) < 0 or max(blocks) >= len(all_blocks):
                raise ValueError(f"blocks must be within [0, {len(all_blocks)-1}] but got {blocks}")

        pooled_per_block: List[torch.Tensor] = []

        def hook_fn(_module, _inp, out):
            h, _m = out
            pooled = pool_h_per_crystal(h, batch.batch.to(h.device), bsz)
            pooled_per_block.append(pooled)

        handles = [all_blocks[i].register_forward_hook(hook_fn) for i in blocks]
        try:
            z = torch.zeros((bsz, self.latent_dim), dtype=torch.float32, device=device)
            _ = self.gemnet(
                z=z,
                frac_coords=batch.pos,
                atom_types=batch.atomic_numbers,
                num_atoms=batch.num_atoms,
                batch=batch.batch,
                lengths=None,
                angles=None,
                lattice=batch.cell,
                edge_index=None,
                to_jimages=None,
                num_bonds=None,
            )
        finally:
            for h in handles:
                h.remove()

        base_repr = pooled_per_block[-1]
        pred = self.head(base_repr)
        if self.distill_head is None:
            return pred, base_repr
        return pred, self.distill_head(base_repr)


def _infer_gemnet_prefix(state_dict: Dict[str, torch.Tensor]) -> str:
    # Try to find a key like "...gemnet.int_blocks.0..."
    for k in state_dict.keys():
        if ".gemnet.int_blocks." in k:
            return k.split(".gemnet.")[0] + ".gemnet."
    for k in state_dict.keys():
        if "gemnet.int_blocks." in k:
            return k.split("gemnet.")[0] + "gemnet."
    raise RuntimeError("Could not infer gemnet prefix in checkpoint state_dict")


def load_gemnet_weights_from_mattergen_ckpt(student: GemNetStudent, ckpt_path: str) -> None:
    """Initialize student.gemnet from a MatterGen Lightning checkpoint."""

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" not in ckpt:
        raise KeyError(f"Checkpoint missing 'state_dict': {ckpt_path}")

    sd: Dict[str, torch.Tensor] = ckpt["state_dict"]
    prefix = _infer_gemnet_prefix(sd)
    sub = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}

    incompatible = student.gemnet.load_state_dict(sub, strict=False)
    loaded = len(sub) - len(incompatible.unexpected_keys)
    if loaded < 100:
        raise RuntimeError(
            f"Loaded too few parameters into student.gemnet. prefix={prefix!r} matched={len(sub)} loaded~={loaded}"
        )
