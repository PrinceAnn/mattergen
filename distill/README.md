# MatterGen → CrysXPP distillation

This folder contains a lightweight distillation pipeline that:

1) Extracts **teacher features** from MatterGen (GemNet-T intermediate hidden states at chosen diffusion times `t`).
2) Trains a **student** (CrysXPP / CGCNN-style graph model or Gemnet) with **supervised regression** + **feature distillation**.

The goal is to keep distillation code minimally coupled to the original repos:

- Teacher extraction lives in `distill/mattergen_teacher/`.
- Student training lives in `distill/crysxpp_student/`.

## Data splits

We follow the official MP-20 CSV splits from:

- `data-release/mp-20/unzipped/mp_20/train.csv`
- `data-release/mp-20/unzipped/mp_20/val.csv`
- `data-release/mp-20/unzipped/mp_20/test.csv`

Each CSV contains a `material_id` column and a `cif` column (CIF content as a string).

## Output feature format

Teacher features are stored as `npz`:

- `material_id`: array of strings, shape `(N,)`
- `x`: float32 array, shape `(N, F)`
- `meta`: a JSON string with metadata (t list, blocks, hidden dim, pooling).

## Next steps

- Run feature extraction for train/val/test.
- Train student with `--teacher-features-*` paths.


## How to run
### Teacher extraction
```bash
python distill/mattergen_teacher/extract_mp20_features.py   --csv data-release/mp-20/unzipped/mp_20/val.csv   --out distill/teacher_features/train_lastblock.npz   --blocks last
```
### Student training

#### Student A: CrysXPP (CGCNN-style)

```bash
python distill/crysxpp_student/train_distill.py \
  --teacher-train distill/teacher_features/train_lastblock.npz \
  --teacher-val distill/teacher_features/val_lastblock.npz \
  --teacher-test distill/teacher_features/test_lastblock.npz \
  --target-property dft_band_gap \
  --out-dir distill/outputs/mp20_distill_dft_band_gap \
  --seed 42 > dft_band_gap_distil.log
```

#### Student B: GemNet (MatterGen GemNetT backbone)

This student uses MatterGen's original `GemNetT` architecture as the backbone.
It builds the student representation to match the teacher feature format using
the `t_values` and `blocks` stored in the teacher `.npz` metadata.


```bash
python distill/gemnet_student/train_distill.py \
  --teacher-train distill/teacher_features/train_lastblock.npz \
  --teacher-val distill/teacher_features/val_lastblock.npz \
  --teacher-test distill/teacher_features/test_lastblock.npz \
  --target-property formation_energy_per_atom \
  --out-dir distill/outputs/mp20_distill_gemnet_formation_energy_per_atom \
  --init-mattergen-ckpt checkpoints/mattergen_base/checkpoints/last.ckpt \
  --device cuda --batch-size 32 --epochs 200
  ```

  ```bash
  CUDA_VISIBLE_DEVICES=1 python distill/gemnet_student/train_distill.py \
  --teacher-train distill/teacher_features/train_lastblock.npz \
  --teacher-val distill/teacher_features/val_lastblock.npz \
  --teacher-test distill/teacher_features/test_lastblock.npz \
  --target-property dft_band_gap \
  --out-dir distill/outputs/mp20_distill_gemnet_dft_band_gap \
  --init-mattergen-ckpt checkpoints/mattergen_base/checkpoints/last.ckpt \
  --device cuda --batch-size 32 --epochs 200
  ```

### Gemnet Baseline for student B

  ```bash
CUDA_VISIBLE_DEVICES=1 python distill/gemnet_student/train_baseline.py \
  --target-property dft_band_gap \
  --out-dir distill/outputs/mp20_baseline_gemnet_dft_band_gap \
  --init-mattergen-ckpt checkpoints/mattergen_base/checkpoints/last.ckpt \
  --device cuda --batch-size 32 --epochs 200
  ``` 
### Plot loss curves from logs
  ```bash


python plot_log_metrics.py \
  mp20_distill_gemnet_formation_energy_per_atom.log \
  mp20_baseline_gemnet_formation_energy_per_atom.log \
  mp20_distill_nopretrain_gemnet_formation_energy_per_atom.log \
  --metrics val_mae \
  --max-epoch 100 \
  --out distill/outputs/_plots/compare_3runs_val_mae_first100.png \
  --title "Formation energy: val_mae (first 100 epochs)"
  ```