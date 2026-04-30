# Cloud Training Guide

This guide is for Linux GPU machines. It avoids the Windows fallback runner and
uses the original BioPathNet implementation under `biopathnet/original/`.

## 1. Clone Code

```bash
git clone --recursive <YOUR_GITHUB_REPO_URL> BPNVer
cd BPNVer
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

## 2. Copy Data

Copy the local data bundle into the cloned project:

```text
BPNVer/data/cloud_run/
```

The expected layout is documented in `docs/cloud_data_manifest.md`.

## 3. Create Environment

For RTX 5090 Linux, use Python 3.10 and PyTorch CUDA 12.8 wheels:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
bash scripts/setup_linux_5090_env.sh
conda activate bpnver
```

Check the GPU:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
PY
```

For RTX 5090, the capability should be `(12, 0)`.

## 4. Run Original BioPathNet Smoke

This uses `mechrep.training.run_original_biopathnet_linux`, which contains no
Windows sparse-operator fallback.

```bash
conda activate bpnver
bash scripts/run_biopathnet_linux_smoke.sh
```

The script writes a runtime config under `results/runtime_configs/` with
absolute `dataset.path` and `output_dir` values. This is needed because the
original BioPathNet script changes into the experiment output directory before
loading data.

The smoke config is intentionally small:

- `batch_size: 1`
- `hidden_dims: [16, 16]`
- `num_negative: 4`
- `full_batch_eval: no`
- `num_epoch: 1`

## 5. Run 5 Minute Profile

```bash
conda activate bpnver
PROFILE_SECONDS=300 bash scripts/run_biopathnet_linux_profile_5min.sh
```

After it ends, check:

- peak GPU memory from `nvidia-smi`
- latest log under `results/biopathnet_linux_profile/`
- whether epoch 0 completed
- seconds per epoch if available

## 6. Run Lightweight Template Pipeline

This is not the original BioPathNet full GNN. It runs the current endpoint pair
baseline, gold-template, pseudo assignment, and semi-template training over the
prepared endpoint-OOD data.

```bash
conda activate bpnver
bash scripts/run_cloud_light_pipeline.sh
```

Main outputs:

- `results/cloud_baseline/`
- `results/cloud_gold_template/`
- `results/cloud_pseudo_template_main/`
- `results/cloud_semi_template_main/`

## 7. Pulling Updates Later

On the cloud server:

```bash
git pull --recurse-submodules
git submodule update --init --recursive
```

The `data/cloud_run/` folder is local data and is not overwritten by Git.

## 8. First Scaling Rules

Do not start with the original large BioPathNet config. Scale in this order:

1. keep `batch_size=1`
2. confirm the smoke run compiles TorchDrug CUDA extensions
3. profile 5 minutes
4. increase `hidden_dims` from `[16, 16]` to `[32, 32]`
5. increase layers only after memory is stable
6. keep `full_batch_eval: no` until the final evaluation stage

If the 5090 runs out of memory on the full graph, use it for smoke/profiling
only and move full training to an 80GB GPU.

## 9. Path-Subgraph K=50 Variant

The K=50 accelerated variant builds a standard BioPathNet data directory from
the union of top-50 train evidence paths per pair:

```bash
bash scripts/build_path_subgraph_k50.sh
```

The output is:

```text
data/cloud_run/biopathnet_path_subgraph_k50/
```

Then run:

```bash
bash scripts/run_biopathnet_linux_subgraph_k50_smoke.sh
PROFILE_SECONDS=300 bash scripts/run_biopathnet_linux_subgraph_k50_profile_5min.sh
```

This is a train-evidence path-union graph, not a dynamic per-pair subgraph
loader. It is intended to test the speed and coverage tradeoff before deeper
model changes.

## 10. No-Evidence Report And Fallback Subgraph

To diagnose train positives without retrieved evidence paths:

```bash
bash scripts/analyze_no_evidence_pairs.sh
```

Outputs:

```text
data/cloud_run/reports/no_evidence_pairs_train.tsv
data/cloud_run/reports/no_evidence_pair_report.json
```

Then build the K=50 graph with fallback structural support:

```bash
bash scripts/build_path_subgraph_k50_fallback.sh
```

The fallback variant adds local train-graph structure for no-evidence train
positive pairs only. It does not create pseudo-template labels and it rejects
non-train or negative fallback pairs.

Output:

```text
data/cloud_run/biopathnet_path_subgraph_k50_fallback/
```

Run the fallback smoke / profiling jobs:

```bash
bash scripts/run_biopathnet_linux_subgraph_k50_fallback_smoke.sh
PROFILE_SECONDS=300 bash scripts/run_biopathnet_linux_subgraph_k50_fallback_profile_5min.sh
```

Compare graph sizes and coverage for the full graph, K=50 evidence-only graph,
and K=50 plus fallback support:

```bash
bash scripts/compare_graph_variants.sh
```

Outputs:

```text
data/cloud_run/reports/graph_variant_comparison.tsv
data/cloud_run/reports/graph_variant_comparison.json
```
