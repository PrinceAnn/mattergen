"""Local copy of the CrysXPP/CGCNN-style model, with a stable API for distillation.

Why a copy?
- Keeps distillation code decoupled from the original `crysxpp/` folder.
- Lets you add new student architectures later without touching upstream code.

Compatibility goal
- Be able to load encoder weights from the official CrysAE pretrained model
  (the file produced by `crysxpp/src/pretrain.py`, typically `model/model_pretrain.pth`).

Changes vs upstream
- `CrystalGraphConvNet.forward(..., return_repr=False)` can optionally return `crys_fea`.
- Added helper `load_crysae_encoder_weights_into_student()`.

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """Convolutional operation on graphs (CGCNN style)."""

    def __init__(self, atom_fea_len: int, nbr_fea_len: int):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len, 2 * self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        # atom_in_fea: (N, atom_fea_len)
        # nbr_fea: (N, M, nbr_fea_len)
        # nbr_fea_idx: (N, M)
        n, m = nbr_fea_idx.shape
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(n, m, self.atom_fea_len), atom_nbr_fea, nbr_fea], dim=2
        )
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(-1, self.atom_fea_len * 2)).view(
            n, m, self.atom_fea_len * 2
        )
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalGraphConvNet(nn.Module):
    """Crystal graph convolutional neural network for regression/classification."""

    def __init__(
        self,
        orig_atom_fea_len: int,
        nbr_fea_len: int,
        atom_fea_len: int = 64,
        n_conv: int = 3,
        h_fea_len: int = 128,
        n_h: int = 1,
        classification: bool = False,
    ):
        super().__init__()

        # Feature selector mask (as in CrysXPP)
        self.mask = nn.Parameter(torch.ones(orig_atom_fea_len), requires_grad=True)

        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList(
            [ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len) for _ in range(n_conv)]
        )
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()

        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])

        self.fc_out = nn.Linear(h_fea_len, 2 if classification else 1)

        if classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def pooling(self, atom_fea, crystal_atom_idx):
        # mean pooling per crystal
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True) for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, return_repr: bool = False):
        # Feature selector
        n = atom_fea.shape[0]
        atom_fea = atom_fea * self.mask.repeat(n, 1)

        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)

        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)

        if self.classification:
            crys_fea = self.dropout(crys_fea)

        if hasattr(self, "fcs") and hasattr(self, "softpluses"):
            for fc, sp in zip(self.fcs, self.softpluses):
                crys_fea = sp(fc(crys_fea))

        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)

        if return_repr:
            return out, crys_fea
        return out


def load_crysae_encoder_weights_into_student(student: CrystalGraphConvNet, ae_path: str, device: torch.device) -> None:
    """Load official CrysAE pretrained weights into the student encoder.

    This mirrors the logic in `crysxpp/src/main.py`.
    """

    # The official CrysXPP pretrain script often saves the entire nn.Module object
    # via `torch.save(save_model, ...)`. That pickle references a module named `model`
    # (i.e., `from model import CrystalAE`). When we load from a different package
    # layout, unpickling can fail with:
    #   AttributeError: Can't get attribute 'CrystalAE' on <module 'model' ...>
    #
    # To be robust, we temporarily alias sys.modules['model'] to the original
    # `crysxpp.src.model` module (if available), so unpickling can resolve.
    import sys

    alias_added = False
    if "model" not in sys.modules:
        try:
            import crysxpp.src.model as crysxpp_model

            sys.modules["model"] = crysxpp_model
            alias_added = True
        except Exception:
            alias_added = False

    obj = torch.load(ae_path, map_location=device)
    if alias_added:
        # Clean up: avoid surprising global side effects
        sys.modules.pop("model", None)

    # Support both formats:
    #  1) saved nn.Module
    #  2) saved dict with state_dict
    if isinstance(obj, dict) and "state_dict" in obj:
        model_dict = obj["state_dict"]
    else:
        model_dict = obj.state_dict()

    # remove decoder / non-encoder weights
    for k in [
        "embedding.weight",
        "fc_adj.weight",
        "fc_adj.bias",
        "fc_atom_feature.weight",
        "fc_atom_feature.bias",
        "fc1.weight",
        "fc1.bias",
    ]:
        if k in model_dict:
            model_dict.pop(k)

    pmodel_dict = student.state_dict()
    pmodel_dict.update(model_dict)
    student.load_state_dict(pmodel_dict)
