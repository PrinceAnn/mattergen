"""Extract MatterGen (GemNet-T) intermediate features for MP-20 CSV splits.

We follow the split CSVs under:
  data-release/mp-20/unzipped/mp_20/{train,val,test}.csv

Each row contains:
  - material_id: unique id
  - cif: CIF content as a multi-line string

We compute teacher features by:
  1) parsing CIF -> pymatgen Structure
  2) sampling noisy states at specified diffusion times t
  3) running the pretrained MatterGen GemNet-T denoiser
  4) capturing (h, m) outputs from each int_block
  5) pooling per-crystal and concatenating across blocks and t

Output is an .npz with keys:
  - material_id: (N,) object array of strings
  - x: (N, F) float32 teacher features
  - meta: JSON string

Note: For robustness, we currently distill ONLY atom-level h representations.
      (m edge features are not guaranteed to be easy to pool without extra mappings.)

"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pymatgen.core import Structure
from torch_geometric.data import Batch

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.diffusion.corruption import LatticeVPSDE, NumAtomsVarianceAdjustedWrappedVESDE
from mattergen.common.gemnet.gemnet import GemNetT
from mattergen.common.gemnet.layers.embedding_block import AtomEmbedding
from mattergen.denoiser import GemNetTDenoiser
from mattergen.diffusion.corruption.d3pm_corruption import D3PMCorruption
from mattergen.diffusion.corruption.multi_corruption import MultiCorruption
from mattergen.diffusion.d3pm.d3pm import MaskDiffusion, create_discrete_diffusion_schedule


@dataclass(frozen=True)
class Config:
    ckpt_path: Path
    csv_path: Path
    out_path: Path
    t_values: List[float]
    device: str
    batch_size: int
    limit: int | None
    corrupt_atomic_numbers: bool


def parse_t_values(t_csv: str) -> List[float]:
    parts = [p.strip() for p in t_csv.split(",") if p.strip()]
    if not parts:
        raise ValueError("--t must be a comma-separated list")
    return [float(p) for p in parts]


def build_mattergen_base_components(device: torch.device) -> Tuple[GemNetTDenoiser, MultiCorruption]:
    gemnet = GemNetT(
        atom_embedding=AtomEmbedding(emb_size=512, with_mask_type=True),
        num_targets=1,
        latent_dim=512,
        num_blocks=4,
        emb_size_atom=512,
        emb_size_edge=512,
        cutoff=7.0,
        max_neighbors=50,
        max_cell_images_per_dim=5,
        otf_graph=True,
        regress_stress=True,
    )
    denoiser = GemNetTDenoiser(
        gemnet=gemnet,
        hidden_dim=512,
        denoise_atom_types=True,
        atom_type_diffusion="mask",
        property_embeddings={},
        property_embeddings_adapt={},
    )

    pos_sde = NumAtomsVarianceAdjustedWrappedVESDE(
        sigma_max=5.0,
        wrapping_boundary=1.0,
        limit_info_key="num_atoms",
    )
    cell_sde = LatticeVPSDE(
        beta_min=0.1,
        beta_max=20,
        limit_density=0.05771451654022283,
        limit_var_scaling_constant=0.25,
    )
    schedule = create_discrete_diffusion_schedule(kind="standard", num_steps=1000)
    atom_d3pm = MaskDiffusion(dim=101, schedule=schedule)
    atomic_numbers_corruption = D3PMCorruption(d3pm=atom_d3pm, offset=1)

    corruption = MultiCorruption(
        sdes={"pos": pos_sde, "cell": cell_sde},
        discrete_corruptions={"atomic_numbers": atomic_numbers_corruption},
    )

    denoiser = denoiser.to(device)
    denoiser.eval()
    return denoiser, corruption


def _infer_denoiser_prefix(state_dict: Dict[str, torch.Tensor]) -> str:
    for key in state_dict.keys():
        if ".gemnet." in key:
            return key.split(".gemnet.")[0] + "."
    for key in state_dict.keys():
        if "gemnet." in key:
            return key.split("gemnet.")[0]
    raise RuntimeError("Could not infer denoiser prefix in checkpoint state_dict")


def load_denoiser_weights_from_ckpt(denoiser: GemNetTDenoiser, ckpt_path: Path) -> None:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if "state_dict" not in ckpt:
        raise KeyError(f"Checkpoint missing 'state_dict': {ckpt_path}")

    sd: Dict[str, torch.Tensor] = ckpt["state_dict"]
    prefix = _infer_denoiser_prefix(sd)
    sub = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
    incompatible = denoiser.load_state_dict(sub, strict=False)
    loaded = len(sub) - len(incompatible.unexpected_keys)
    if loaded < 100:
        raise RuntimeError(
            f"Loaded too few parameters into denoiser. prefix={prefix!r} matched={len(sub)} loaded~={loaded}"
        )


def chemgraph_from_structure(structure: Structure) -> ChemGraph:
    # ChemGraph expects fractional coords in [0,1)
    num_atoms = torch.tensor(structure.num_sites)
    return ChemGraph(
        pos=torch.from_numpy(structure.frac_coords).float() % 1.0,
        cell=torch.tensor(structure.lattice.matrix, dtype=torch.float32).unsqueeze(0),
        atomic_numbers=torch.tensor(structure.atomic_numbers, dtype=torch.long),
        num_atoms=num_atoms,
        num_nodes=num_atoms,  # special attr used by PyG batching
    )


def parse_structure_from_cif_str(cif_str: str) -> Structure:
    # pymatgen Structure.from_str supports fmt="cif"
    return Structure.from_str(cif_str, fmt="cif")


def pool_h_per_crystal(h: torch.Tensor, batch_idx: torch.Tensor, num_graphs: int) -> torch.Tensor:
    # mean pool over atoms for each crystal in batch
    # h: (N_atoms, D), batch_idx: (N_atoms,) in [0, B)
    out = torch.zeros((num_graphs, h.shape[-1]), device=h.device, dtype=h.dtype)
    out.scatter_add_(0, batch_idx[:, None].expand(-1, h.shape[-1]), h)
    counts = torch.bincount(batch_idx, minlength=num_graphs).clamp(min=1).to(h.dtype)
    out = out / counts[:, None]
    return out


@torch.no_grad()
def extract_teacher_features(
    denoiser: GemNetTDenoiser,
    corruption: MultiCorruption,
    clean_batch: Batch,
    t_values: List[float],
    device: torch.device,
    corrupt_atomic_numbers: bool,
    blocks: List[int],
) -> torch.Tensor:
    """Return (B, F) teacher features for a batch."""

    bsz = int(clean_batch.num_graphs)

    all_blocks = denoiser.gemnet.int_blocks
    if not blocks:
        raise ValueError("blocks must contain at least one block index")
    if min(blocks) < 0 or max(blocks) >= len(all_blocks):
        raise ValueError(
            f"blocks must be within [0, {len(all_blocks)-1}] but got {blocks}"
        )
    blocks = sorted(set(int(i) for i in blocks))
    block_count = len(blocks)
    hidden_dim = int(denoiser.hidden_dim)

    feats_per_t: List[torch.Tensor] = []

    for t_value in t_values:
        t = torch.full((bsz,), float(t_value), dtype=torch.float32, device=device)
        noisy_batch = corruption.sample_marginal(clean_batch, t)
        if not corrupt_atomic_numbers:
            noisy_batch = noisy_batch.replace(atomic_numbers=clean_batch.atomic_numbers)

        h_block_pooled: List[torch.Tensor] = []

        def hook_fn(_module, _inp, out):
            h, _m = out
            # h: (N_atoms, D)
            pooled = pool_h_per_crystal(h, noisy_batch.batch.to(h.device), bsz)
            h_block_pooled.append(pooled)

        # Only hook the selected blocks
        handles = [all_blocks[i].register_forward_hook(hook_fn) for i in blocks]
        try:
            z_per_crystal = denoiser.noise_level_encoding(t)
            _ = denoiser.gemnet(
                z=z_per_crystal,
                frac_coords=noisy_batch.pos,
                atom_types=noisy_batch.atomic_numbers,
                num_atoms=noisy_batch.num_atoms,
                batch=noisy_batch.batch,
                lengths=None,
                angles=None,
                lattice=noisy_batch.cell,
                edge_index=None,
                to_jimages=None,
                num_bonds=None,
            )
        finally:
            for h in handles:
                h.remove()

        if len(h_block_pooled) != block_count:
            raise RuntimeError(
                f"Expected {block_count} block features, got {len(h_block_pooled)} at t={t_value}"
            )

        feats_per_t.append(torch.cat(h_block_pooled, dim=-1))

    # concat across t
    out = torch.cat(feats_per_t, dim=-1)
    expected_dim = len(t_values) * block_count * hidden_dim
    if out.shape[-1] != expected_dim:
        raise RuntimeError(f"Teacher dim mismatch: got {out.shape[-1]} expected {expected_dim}")

    return out


def iter_batches(material_ids: List[str], cif_strs: List[str], batch_size: int) -> Iterable[Tuple[List[str], Batch]]:
    for start in range(0, len(material_ids), batch_size):
        mids = material_ids[start : start + batch_size]
        cifs = cif_strs[start : start + batch_size]

        graphs = []
        kept_ids = []
        for mid, cif in zip(mids, cifs):
            try:
                s = parse_structure_from_cif_str(cif)
                graphs.append(chemgraph_from_structure(s))
                kept_ids.append(mid)
            except Exception:
                # Skip un-parseable CIF rows (rare). We don't crash the whole extraction.
                continue

        if not graphs:
            continue

        batch = Batch.from_data_list(graphs)
        yield kept_ids, batch


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract MatterGen teacher features for MP-20 CSV")
    p.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/mattergen_base/checkpoints/last.ckpt",
        help="MatterGen Lightning checkpoint",
    )
    p.add_argument("--csv", type=str, required=True, help="Path to MP-20 split CSV (train/val/test)")
    p.add_argument("--out", type=str, required=True, help="Output .npz")
    p.add_argument(
        "--t",
        type=str,
        default="1e-3,0.01,0.1,0.5,1.0",
        help="Comma-separated diffusion times in [0,1]",
    )
    p.add_argument(
        "--blocks",
        type=str,
        default="last",
        help="Which GemNet int_blocks to use: 'last' or comma-separated indices like '0,1,3'",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="Debug: only process first N rows")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--no_corrupt_atomic_numbers",
        action="store_true",
        help="Only corrupt pos/cell and keep atomic_numbers clean.",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()

    cfg = Config(
        ckpt_path=Path(args.ckpt),
        csv_path=Path(args.csv),
        out_path=Path(args.out),
        t_values=parse_t_values(args.t),
        device=args.device,
        batch_size=int(args.batch_size),
        limit=int(args.limit) if args.limit is not None else None,
        corrupt_atomic_numbers=not bool(args.no_corrupt_atomic_numbers),
    )

    df = pd.read_csv(cfg.csv_path)
    if cfg.limit is not None:
        df = df.iloc[: cfg.limit]

    material_id = df["material_id"].astype(str).tolist()
    cif_strs = df["cif"].astype(str).tolist()

    device = torch.device(cfg.device)
    denoiser, corruption = build_mattergen_base_components(device)
    load_denoiser_weights_from_ckpt(denoiser, cfg.ckpt_path)

    if args.blocks.strip().lower() == "last":
        blocks = [len(denoiser.gemnet.int_blocks) - 1]
    else:
        blocks = [int(x.strip()) for x in args.blocks.split(",") if x.strip()]

    all_feats: List[np.ndarray] = []
    all_ids: List[str] = []

    total_rows = len(material_id)
    kept_total = 0

    for kept_ids, batch in tqdm(
        iter_batches(material_id, cif_strs, cfg.batch_size),
        total=(total_rows + cfg.batch_size - 1) // cfg.batch_size,
        desc="Extract teacher features",
    ):
        batch = batch.to(device)
        feats = extract_teacher_features(
            denoiser=denoiser,
            corruption=corruption,
            clean_batch=batch,
            t_values=cfg.t_values,
            device=device,
            corrupt_atomic_numbers=cfg.corrupt_atomic_numbers,
            blocks=blocks,
        )
        all_feats.append(feats.detach().cpu().float().numpy())
        all_ids.extend(kept_ids)
        kept_total += len(kept_ids)

    # tqdm prints progress; show skip ratio summary too
    skipped = total_rows - kept_total
    print(f"Parsed CIFs kept={kept_total}/{total_rows} (skipped={skipped})")

    x = np.concatenate(all_feats, axis=0) if all_feats else np.zeros((0, 0), dtype=np.float32)

    meta = {
        "t_values": cfg.t_values,
        "num_blocks_used": len(blocks),
        "blocks": blocks,
        "hidden_dim": 512,
        "pooling": "mean_atoms_per_crystal",
        "feature": "concat(t, selected_blocks, h_pool)",
        "corrupt_atomic_numbers": cfg.corrupt_atomic_numbers,
        "csv": str(cfg.csv_path),
        "ckpt": str(cfg.ckpt_path),
    }

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cfg.out_path,
        material_id=np.asarray(all_ids, dtype=object),
        x=x.astype(np.float32),
        meta=json.dumps(meta),
    )

    print(f"Wrote {len(all_ids)} features to {cfg.out_path} with dim={x.shape[1] if x.ndim==2 else '??'}")


if __name__ == "__main__":
    main()
