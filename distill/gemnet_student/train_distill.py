"""Train a GemNet (MatterGen GemNetT) student with MatterGen feature distillation.

Compared with distill/crysxpp_student/train_distill.py:
- Student backbone is MatterGen's original `GemNetT`.
- Inputs are `ChemGraph` batches (frac coords + cell + atomic_numbers).
- Representation is built to *match teacher feature format* (concat over teacher t/blocks).

Teacher features must be produced by:
  distill/mattergen_teacher/extract_mp20_features.py

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make direct script execution work when running `python distill/gemnet_student/train_distill.py`
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from distill.crysxpp_student.teacher_features import gather_teacher_batch, load_teacher_npz
from distill.gemnet_student.mp20_chemgraph_dataset import DatasetConfig, MP20ChemGraphDataset, collate_chemgraphs
from distill.gemnet_student.model import GemNetStudent, StudentConfig, load_gemnet_weights_from_mattergen_ckpt


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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrainConfig:
    train_csv: Path
    val_csv: Path
    test_csv: Path
    target_property: str
    teacher_train: Path
    teacher_val: Path
    teacher_test: Path
    out_dir: Path
    batch_size: int
    epochs: int
    lr: float
    optim: str
    weight_decay: float
    lr_milestones: list[int]
    lambda_distill: float
    device: str
    limit: int | None
    seed: int
    init_mattergen_ckpt: Path | None


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train GemNet student with MatterGen distillation")

    p.add_argument("--train-csv", type=str, default="data-release/mp-20/unzipped/mp_20/train.csv")
    p.add_argument("--val-csv", type=str, default="data-release/mp-20/unzipped/mp_20/val.csv")
    p.add_argument("--test-csv", type=str, default="data-release/mp-20/unzipped/mp_20/test.csv")

    p.add_argument("--target-property", type=str, default="formation_energy_per_atom")

    p.add_argument("--teacher-train", type=str, required=True)
    p.add_argument("--teacher-val", type=str, required=True)
    p.add_argument("--teacher-test", type=str, required=True)

    p.add_argument("--out-dir", type=str, default="distill/outputs/mp20_distill_gemnet")

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--optim", type=str, default="Adam", choices=["Adam", "SGD"])
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--lr-milestones", type=int, nargs="+", default=[30, 40])

    p.add_argument("--lambda-distill", type=float, default=1.0)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None, help="Debug: only first N samples")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--init-mattergen-ckpt",
        type=str,
        default=None,
        help="Optional: initialize student.gemnet weights from a MatterGen Lightning checkpoint (e.g. checkpoints/mattergen_base/checkpoints/last.ckpt)",
    )

    return p


@torch.no_grad()
def evaluate(
    loader: DataLoader,
    model: GemNetStudent,
    criterion: nn.Module,
    normalizer: Normalizer,
    device: torch.device,
    desc: str = "val",
):
    model.eval()
    losses: list[float] = []
    maes: list[float] = []

    for batch_out in tqdm(loader, desc=desc, leave=False):
        if batch_out is None:
            continue
        batch, target, _ids = batch_out
        batch = batch.to(device)
        target = target.to(device)
        target_normed = normalizer.norm(target)

        pred, _repr = model(batch)
        loss = criterion(pred, target_normed)
        losses.append(float(loss.detach().cpu().item()))

        mae = torch.mean(torch.abs(normalizer.denorm(pred.detach().cpu()) - target.detach().cpu())).item()
        maes.append(float(mae))

    return float(np.mean(losses)) if losses else float("nan"), float(np.mean(maes)) if maes else float("nan")


def main() -> None:
    args = build_argparser().parse_args()

    cfg = TrainConfig(
        train_csv=Path(args.train_csv),
        val_csv=Path(args.val_csv),
        test_csv=Path(args.test_csv),
        target_property=str(args.target_property),
        teacher_train=Path(args.teacher_train),
        teacher_val=Path(args.teacher_val),
        teacher_test=Path(args.teacher_test),
        out_dir=Path(args.out_dir),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        optim=str(args.optim),
        weight_decay=float(args.weight_decay),
        lr_milestones=list(args.lr_milestones),
        lambda_distill=float(args.lambda_distill),
        device=str(args.device),
        limit=int(args.limit) if args.limit is not None else None,
        seed=int(args.seed),
        init_mattergen_ckpt=Path(args.init_mattergen_ckpt) if args.init_mattergen_ckpt else None,
    )

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # teacher features (train meta defines t/blocks)
    tf_train = load_teacher_npz(str(cfg.teacher_train))
    # tf_val = load_teacher_npz(str(cfg.teacher_val))
    # tf_test = load_teacher_npz(str(cfg.teacher_test))

    blocks = tf_train.meta.get("blocks")
    hidden_dim = int(tf_train.meta.get("hidden_dim", 512))
    if not isinstance(blocks, list):
        raise ValueError("Teacher meta must include list field 'blocks'")

    # build datasets
    ds_cfg = DatasetConfig(target_property=cfg.target_property, limit=cfg.limit)
    train_ds = MP20ChemGraphDataset(str(cfg.train_csv), ds_cfg)
    val_ds = MP20ChemGraphDataset(str(cfg.val_csv), ds_cfg)
    test_ds = MP20ChemGraphDataset(str(cfg.test_csv), ds_cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate_chemgraphs)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_chemgraphs)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_chemgraphs)

    # normalizer from CSV targets (fast, no CIF parsing)
    sample_df = train_ds.df
    sample_targets = torch.tensor(sample_df[cfg.target_property].values[: min(len(sample_df), 2000)], dtype=torch.float32).view(-1, 1)
    normalizer = Normalizer(sample_targets)

    model = GemNetStudent(
        teacher_feature_dim=int(tf_train.dim),
        blocks=[int(b) for b in blocks],
        cfg=StudentConfig(emb_size=hidden_dim, latent_dim=hidden_dim),
        out_dim=1,
    ).to(device)

    # sanity: teacher/student repr dims
    # distill repr must match teacher feature dim by construction
    if int(tf_train.dim) != int(model.teacher_feature_dim):
        raise RuntimeError(f"Teacher dim mismatch: teacher_dim={tf_train.dim} student.teacher_feature_dim={model.teacher_feature_dim}")

    if cfg.init_mattergen_ckpt is not None:
        load_gemnet_weights_from_mattergen_ckpt(model, str(cfg.init_mattergen_ckpt))

    criterion_sup = nn.MSELoss()
    criterion_distill = nn.MSELoss(reduction="none")

    params = list(model.parameters())
    if cfg.optim == "SGD":
        optimizer = optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    else:
        optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    scheduler = MultiStepLR(optimizer, milestones=cfg.lr_milestones, gamma=0.1)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    best_val_mae = 1e9

    for epoch in range(cfg.epochs):
        model.train()

        for batch_out in tqdm(train_loader, desc=f"train epoch {epoch+1}/{cfg.epochs}", leave=False):
            if batch_out is None:
                continue
            batch, target, material_ids = batch_out
            batch = batch.to(device)
            target = target.to(device)
            target_normed = normalizer.norm(target)

            pred, repr_vec = model(batch)
            loss_sup = criterion_sup(pred, target_normed)

            # teacher batch (B, F)
            t_x_np, t_mask_np = gather_teacher_batch(tf_train, material_ids)
            t_x = torch.tensor(t_x_np, device=device)
            t_mask = torch.tensor(t_mask_np, device=device)

            distill_per = criterion_distill(repr_vec, t_x).mean(dim=1)
            loss_distill = (distill_per * t_mask).sum() / (t_mask.sum().clamp(min=1.0))

            loss = loss_sup + cfg.lambda_distill * loss_distill

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        scheduler.step()

        val_loss, val_mae = evaluate(val_loader, model, criterion_sup, normalizer, device, desc=f"val epoch {epoch+1}/{cfg.epochs}")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "normalizer_mean": float(normalizer.mean),
                    "normalizer_std": float(normalizer.std),
                    "teacher_meta": tf_train.meta,
                    "cfg": cfg.__dict__,
                },
                cfg.out_dir / "best.pt",
            )

        print(f"epoch={epoch+1} val_loss={val_loss:.4f} val_mae={val_mae:.4f} best_val_mae={best_val_mae:.4f}")

    # final test eval
    ckpt = torch.load(cfg.out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss, test_mae = evaluate(test_loader, model, criterion_sup, normalizer, device, desc="test")
    print(f"test_loss={test_loss:.4f} test_mae={test_mae:.4f}")


if __name__ == "__main__":
    main()
