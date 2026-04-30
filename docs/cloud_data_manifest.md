# Cloud Data Bundle

The cloud training data bundle is local-only and lives under:

```text
data/cloud_run/
```

It is intentionally not tracked by Git because it contains large KG and evidence
path files. Copy this directory to the cloud server next to the cloned code.

## Contents

- `all_data/`
  - `pairs.tsv`
  - `gold_paths.tsv`
  - `gold_paths_rejected.tsv`
  - `conversion_report.json`
- `splits/`
  - `train.tsv`
  - `valid.tsv`
  - `test.tsv`
  - `leakage_report.json`
- `biopathnet_full/`
  - `train1.txt`
  - `train2.txt`
  - `valid.txt`
  - `test.txt`
  - `entity_names.txt`
  - `entity_types.txt`
  - `prepare_report.json`
- `templates/`
  - `templates.tsv`
  - `gold_template_labels_train.tsv`
  - `gold_template_labels_valid.tsv`
  - `gold_template_labels_test.tsv`
  - `gold_template_labels.tsv`
  - `template_extraction_report.json`
- `gold_template/`
  - `template_vocab.json`
  - `predictions_train.tsv`
  - `predictions_valid.tsv`
  - `predictions_test.tsv`
  - `evidence_paths_train.tsv`
  - `evidence_paths_train_report.json`
- `pseudo_template_main/`
  - frozen stricter pseudo labels from `tau_match=0.85`
- `pseudo_template_high_coverage/`
  - frozen current/high-coverage pseudo labels

## Current Real Data Counts

- Pair splits:
  - train: 14,812 pairs
  - valid: 2,082 pairs
  - test: 3,676 pairs
- BioPathNet full graph:
  - `train1.txt`: 8,129,318 KG/BRG edges
  - `train2.txt`: 7,406 train positive endpoint edges
  - `valid.txt`: 1,041 valid positive endpoint edges
  - `test.txt`: 1,838 test positive endpoint edges
- Templates:
  - 502 template classes
- Frozen pseudo labels:
  - main stricter pseudo set: 99 labels
  - high-coverage ablation pseudo set: 332 labels

## Copy Pattern

From the project root on the local machine, copy only the bundle:

```bash
rsync -av data/cloud_run/ USER@SERVER:/path/to/BPNVer/data/cloud_run/
```

On Windows, use `scp`, `rsync` from WSL, or the cloud provider file upload UI.
