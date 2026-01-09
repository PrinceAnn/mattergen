"""Train CrysXPP student with MatterGen feature distillation on MP-20 splits.

This is a lightweight trainer that:
- loads MP-20 CSV splits (exact official split files)
- loads teacher features (npz produced by distill/mattergen_teacher/extract_mp20_features.py)
- initializes student from the official CrysAE pretrained encoder (optional)
- optimizes supervised regression + distillation loss

We intentionally keep code self-contained instead of modifying the original `crysxpp/src/main.py`.

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import warnings
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import os
import sys

# Make direct script execution work when running `python distill/crysxpp_student/train_distill.py`
# by ensuring the repository root is on sys.path so `distill` package imports resolve.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from distill.crysxpp_student.mp20_csv_dataset import GraphConfig, MP20CSVDataset, collate_pool
from distill.crysxpp_student.teacher_features import gather_teacher_batch, load_teacher_npz
from distill.crysxpp_student.model import CrystalGraphConvNet, load_crysae_encoder_weights_into_student


# Keep logs readable. This is intentionally conservative (doesn't silence Exceptions).
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class Normalizer:
    def __init__(self, tensor: torch.Tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor: torch.Tensor) -> torch.Tensor:
        return normed_tensor * self.std + self.mean


def set_seed(seed: int) -> None:
    """Best-effort reproducibility across Python/NumPy/PyTorch.

    Notes:
      - Full determinism can be slower.
      - Some GPU ops may still be nondeterministic depending on versions/hardware.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # cuDNN / matmul determinism knobs (safe defaults)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # NOTE: We intentionally do NOT call torch.use_deterministic_algorithms(True) here,
    # because it can error out on some CUDA/cuBLAS setups unless extra environment
    # variables are set. Seeding + cudnn flags is usually enough for practical reproducibility.


@dataclass
class TrainConfig:
    train_csv: Path
    val_csv: Path
    test_csv: Path
    atom_init_json: Path
    target_property: str
    teacher_train: Path
    teacher_val: Path
    teacher_test: Path
    pretrained_ae: Path | None
    out_dir: Path
    batch_size: int
    epochs: int
    lr: float
    optim: str
    weight_decay: float
    lr_milestones: list[int]
    lambda_distill: float
    device: str
    max_nbr: int
    radius: float
    limit: int | None


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train CrysXPP with MatterGen distillation")

    p.add_argument("--train-csv", type=str, default="data-release/mp-20/unzipped/mp_20/train.csv")
    p.add_argument("--val-csv", type=str, default="data-release/mp-20/unzipped/mp_20/val.csv")
    p.add_argument("--test-csv", type=str, default="data-release/mp-20/unzipped/mp_20/test.csv")

    p.add_argument("--atom-init-json", type=str, default="crysxpp/custom_datasets/mp_20/atom_init.json")
    p.add_argument("--target-property", type=str, default="formation_energy_per_atom")

    p.add_argument("--teacher-train", type=str, required=True)
    p.add_argument("--teacher-val", type=str, required=True)
    p.add_argument("--teacher-test", type=str, required=True)

    p.add_argument("--pretrained-ae", type=str, default="crysxpp/model/model_pretrain.pth")
    p.add_argument("--no-pretrained-ae", action="store_true")

    p.add_argument("--out-dir", type=str, default="distill/outputs/mp20_distill")

    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--optim", type=str, default="Adam", choices=["Adam", "SGD"])
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--lr-milestones", type=int, nargs="+", default=[30, 40])

    p.add_argument("--lambda-distill", type=float, default=1.0)

    p.add_argument("--max-nbr", type=int, default=12)
    p.add_argument("--radius", type=float, default=8.0)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None, help="Debug: only first N samples")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return p


def evaluate(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    normalizer: Normalizer,
    device: torch.device,
    desc: str = "val",
):
    model.eval()
    losses = []
    maes = []

    with torch.no_grad():
        for (atom_fea, nbr_fea, nbr_fea_idx, crys_idx), target, material_ids in tqdm(
            loader,
            desc=desc,
            leave=False,
        ):
            atom_fea = atom_fea.to(device)
            nbr_fea = nbr_fea.to(device)
            nbr_fea_idx = nbr_fea_idx.to(device)
            crys_idx = [x.to(device) for x in crys_idx]

            target_normed = normalizer.norm(target).to(device)

            out = model(atom_fea, nbr_fea, nbr_fea_idx, crys_idx)
            pred = out[0] if isinstance(out, (tuple, list)) else out
            loss = criterion(pred, target_normed)
            losses.append(loss.detach().cpu().item())

            mae = torch.mean(torch.abs(normalizer.denorm(pred.detach().cpu()) - target.cpu())).item()
            maes.append(mae)

    return float(np.mean(losses)), float(np.mean(maes))


def main() -> None:
    args = build_argparser().parse_args()

    set_seed(int(args.seed))

    cfg = TrainConfig(
        train_csv=Path(args.train_csv),
        val_csv=Path(args.val_csv),
        test_csv=Path(args.test_csv),
        atom_init_json=Path(args.atom_init_json),
        target_property=str(args.target_property),
        teacher_train=Path(args.teacher_train),
        teacher_val=Path(args.teacher_val),
        teacher_test=Path(args.teacher_test),
        pretrained_ae=None if args.no_pretrained_ae else Path(args.pretrained_ae),
        out_dir=Path(args.out_dir),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        optim=str(args.optim),
        weight_decay=float(args.weight_decay),
        lr_milestones=list(args.lr_milestones),
        lambda_distill=float(args.lambda_distill),
        device=str(args.device),
        max_nbr=int(args.max_nbr),
        radius=float(args.radius),
        limit=int(args.limit) if args.limit is not None else None,
    )

    device = torch.device(cfg.device)

    # datasets
    graph_cfg = GraphConfig(max_num_nbr=cfg.max_nbr, radius=cfg.radius)

    train_ds = MP20CSVDataset(
        csv_path=str(cfg.train_csv),
        atom_init_json=str(cfg.atom_init_json),
        target_property=cfg.target_property,
        graph_cfg=graph_cfg,
        limit=cfg.limit,
    )
    val_ds = MP20CSVDataset(
        csv_path=str(cfg.val_csv),
        atom_init_json=str(cfg.atom_init_json),
        target_property=cfg.target_property,
        graph_cfg=graph_cfg,
        limit=cfg.limit,
    )
    test_ds = MP20CSVDataset(
        csv_path=str(cfg.test_csv),
        atom_init_json=str(cfg.atom_init_json),
        target_property=cfg.target_property,
        graph_cfg=graph_cfg,
        limit=cfg.limit,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate_pool)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_pool)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_pool)

    # normalizer (sample first up to 2000)
    sample_n = min(len(train_ds), 2000)
    sample_targets = torch.stack([train_ds[i][1] for i in range(sample_n)], dim=0)
    normalizer = Normalizer(sample_targets)

    # build model
    (atom_fea0, nbr_fea0, nbr_fea_idx0), _, _ = train_ds[0]
    orig_atom_fea_len = atom_fea0.shape[-1]
    nbr_fea_len = nbr_fea0.shape[-1]

    model = CrystalGraphConvNet(
        orig_atom_fea_len=orig_atom_fea_len,
        nbr_fea_len=nbr_fea_len,
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128,
        n_h=1,
        classification=False,
    ).to(device)

    if cfg.pretrained_ae is not None and cfg.pretrained_ae.exists():
        load_crysae_encoder_weights_into_student(model, str(cfg.pretrained_ae), device)

    # teacher features
    tf_train = load_teacher_npz(str(cfg.teacher_train))
    # tf_val = load_teacher_npz(str(cfg.teacher_val))
    # tf_test = load_teacher_npz(str(cfg.teacher_test))

    teacher_dim = tf_train.dim

    # distill head mapping student crystal embedding -> teacher_dim
    distill_head = nn.Linear(128, teacher_dim).to(device)

    criterion_sup = nn.MSELoss()
    criterion_distill = nn.MSELoss(reduction="none")

    params = list(model.parameters()) + list(distill_head.parameters())

    if cfg.optim == "SGD":
        optimizer = optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    else:
        optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    scheduler = MultiStepLR(optimizer, milestones=cfg.lr_milestones, gamma=0.1)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    best_val_mae = 1e9

    for epoch in range(cfg.epochs):
        model.train()
        distill_head.train()

        for (atom_fea, nbr_fea, nbr_fea_idx, crys_idx), target, material_ids in tqdm(
            train_loader,
            desc=f"train epoch {epoch+1}/{cfg.epochs}",
            leave=False,
        ):
            atom_fea = atom_fea.to(device)
            nbr_fea = nbr_fea.to(device)
            nbr_fea_idx = nbr_fea_idx.to(device)
            crys_idx = [x.to(device) for x in crys_idx]

            target_normed = normalizer.norm(target).to(device)

            pred, crys_fea = model(atom_fea, nbr_fea, nbr_fea_idx, crys_idx, return_repr=True)
            loss_sup = criterion_sup(pred, target_normed)

            # teacher batch
            t_x_np, t_mask_np = gather_teacher_batch(tf_train, material_ids)
            t_x = torch.tensor(t_x_np, device=device)
            t_mask = torch.tensor(t_mask_np, device=device)

            # student representation (crystal embedding)
            s_proj = distill_head(crys_fea)

            distill_per = criterion_distill(s_proj, t_x).mean(dim=1)
            loss_distill = (distill_per * t_mask).sum() / (t_mask.sum().clamp(min=1.0))

            loss = loss_sup + cfg.lambda_distill * loss_distill

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        scheduler.step()

        val_loss, val_mae = evaluate(
            val_loader,
            model,
            criterion_sup,
            normalizer,
            device,
            desc=f"val epoch {epoch+1}/{cfg.epochs}",
        )
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "distill_head": distill_head.state_dict(),
                    "normalizer_mean": float(normalizer.mean),
                    "normalizer_std": float(normalizer.std),
                    "cfg": cfg.__dict__,
                },
                cfg.out_dir / "best.pt",
            )

        print(
            f"epoch={epoch+1} val_loss={val_loss:.4f} val_mae={val_mae:.4f} best_val_mae={best_val_mae:.4f}"
        )

    # final test eval
    ckpt = torch.load(cfg.out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss, test_mae = evaluate(test_loader, model, criterion_sup, normalizer, device, desc="test")
    print(f"test_loss={test_loss:.4f} test_mae={test_mae:.4f}")


if __name__ == "__main__":
    main()
