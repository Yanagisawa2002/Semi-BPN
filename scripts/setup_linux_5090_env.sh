#!/usr/bin/env bash
set -euo pipefail

conda create -n bpnver python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bpnver

python -m pip install --upgrade pip setuptools wheel ninja
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
python -m pip install torchdrug==0.2.1 easydict pyyaml pandas numpy scipy scikit-learn tqdm pytest

python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
x = torch.randn(1024, 1024, device="cuda")
print("matmul mean", (x @ x).mean().item())
PY
