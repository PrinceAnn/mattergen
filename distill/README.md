# MatterGen → CrysXPP distillation

This folder contains a lightweight distillation pipeline that:

1) Extracts **teacher features** from MatterGen (GemNet-T intermediate hidden states at chosen diffusion times `t`).
2) Trains a **student** (CrysXPP / CGCNN-style graph model) with **supervised regression** + **feature distillation**.

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
python distill/mattergen_teacher/extract_mp20_features.py   --csv data-release/mp-20/unzipped/mp_20/val.csv   --out val_lastblock.npz   --blocks last
```
### Student training

```bash
 python distill/crysxpp_student/train_distill.py   --teacher-train train_lastblock.npz   --teacher-val val_lastblock.npz   --teacher-test test_lastblock.npz
```
```bash
python distill/crysxpp_student/train_distill.py \
  --teacher-train train_lastblock.npz \
  --teacher-val val_lastblock.npz \
  --teacher-test test_lastblock.npz \
  --target-property dft_band_gap \
  --out-dir distill/outputs/mp20_distill_dft_band_gap \
  --seed 42 > dft_band_gap_distil.log

python distill/crysxpp_student/train_distill.py \
  --teacher-train train_lastblock.npz \
  --teacher-val val_lastblock.npz \
  --teacher-test test_lastblock.npz \
  --target-property e_above_hull \
  --out-dir distill/outputs/mp20_distill_e_above_hull \
  --seed 42 > e_above_hull_distil.log
```