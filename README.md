# Template-guided semi-supervised BioPathNet

## Goal

This project extends BioPathNet for endpoint-level OOD drug repurposing using sparse curated gold mechanism paths.

The minimal method is:

BioPathNet baseline  
+ gold mechanism template supervision  
+ high-confidence pseudo-template assignment for no-gold positive pairs  
+ endpoint-OOD evaluation  
+ mechanism audit metrics

## Current development stage

Stage 1: endpoint-OOD data, template extraction, frozen pseudo labels, and
Linux cloud training scaffolding are available.

## Planned milestones

1. Run BioPathNet baseline on endpoint-level drug-endpoint pairs.
2. Implement endpoint-OOD split with leakage checks.
3. Extract typed templates from gold mechanism paths.
4. Add gold-template supervised loss.
5. Add high-confidence pseudo-template assignment.
6. Train semi-supervised model.
7. Run ablations.
8. Run mechanism audit.

## Main evaluation

Prediction:
- AUROC
- AUPRC
- Recall@K
- Hits@K

Mechanism audit:
- Gold path recall@K
- Template match@K
- Assignment coverage
- Shuffled-template control gap

## Cloud Training

Linux training instructions are in `docs/cloud_training.md`.

The cloud data bundle is local-only:

```text
data/cloud_run/
```

It is not tracked by Git. See `docs/cloud_data_manifest.md` for the required
layout and current real-data counts.
