# AGENTS.md

## Project goal

This project extends BioPathNet for endpoint-level OOD drug repurposing.

The target method is:

Template-guided semi-supervised mechanism regularization on top of BioPathNet.

Main idea:
- Use BioPathNet as a path-aware drug-endpoint pair encoder.
- Convert sparse curated gold mechanism paths into typed mechanism templates.
- Use gold templates as supervised mechanism labels.
- Assign high-confidence pseudo-template labels to no-gold positive training pairs.
- Improve endpoint-level OOD prediction.
- Output mechanism-template assignments and top-K evidence paths for audit.

## Main model variants

The project must support these variants:

1. BioPathNet baseline.
2. BioPathNet + gold-template supervision.
3. BioPathNet + gold-template supervision + high-confidence pseudo-template assignment.
4. Shuffled-template control.
5. Random-pseudo control.

## Development rules

- Do not rewrite the original BioPathNet implementation unless explicitly requested.
- Prefer adding new code under `mechrep/`.
- Keep the original BioPathNet code under `biopathnet/original/` as stable as possible.
- Every new module must include a minimal unit test or sanity check.
- Every training script must support config-driven execution.
- Every experiment must save:
  - config
  - random seed
  - metrics
  - prediction file
  - checkpoint path, when applicable
- All random operations must use a fixed seed.
- Do not silently drop failed examples.
- Do not change train/validation/test split logic without adding a leakage check.

## Data leakage rules

These rules are mandatory:

- In endpoint-OOD evaluation, no test endpoint may appear in training pairs.
- Test labels must never be used for pseudo-template assignment.
- Test evidence paths must not be used to create training pseudo-labels.
- Gold mechanism paths from the test set must not be used as training mechanism supervision.
- Thresholds for pseudo-template assignment must be selected on train/validation data only.
- Any split builder must output a leakage report.

## Method components

The minimal method is:

Pair prediction loss:

`L_pred = BCE(pair_score, pair_label)`

Gold-template loss:

`L_gold = CE(template_logits, gold_template_label)`

Pseudo-template loss:

`L_pseudo = CE(template_logits, pseudo_template_label)`

Total loss:

`L = L_pred + lambda_gold * L_gold + lambda_pseudo * L_pseudo`

Masking rules:
- `L_gold` applies only to examples with curated gold-template labels.
- `L_pseudo` applies only to no-gold positive training examples with high-confidence pseudo-template labels.
- Negative pairs do not receive pseudo-template labels.
- Test examples never receive pseudo-template labels during training.

## Evaluation metrics

Prediction metrics:
- AUROC
- AUPRC
- Recall@K
- Hits@K
- MRR, if the task is evaluated as ranking

Mechanism audit metrics:
- Gold path recall@K
- Template match@K
- Assignment coverage
- Assignment confidence
- Shuffled-template control gap
- Random-pseudo control gap

## Protected files

Avoid modifying:
- `biopathnet/original/` unless explicitly required.
- Existing BioPathNet model internals unless a wrapper cannot solve the problem.

Preferred new files:
- `mechrep/data/build_pairs.py`
- `mechrep/data/build_ood_splits.py`
- `mechrep/templates/extract_templates.py`
- `mechrep/templates/match_paths_to_templates.py`
- `mechrep/templates/pseudo_assign.py`
- `mechrep/models/biopathnet_wrapper.py`
- `mechrep/models/template_head.py`
- `mechrep/models/losses.py`
- `mechrep/training/train_baseline.py`
- `mechrep/training/train_gold_template.py`
- `mechrep/training/train_semi_supervised.py`
- `mechrep/evaluation/eval_ood.py`
- `mechrep/evaluation/eval_audit.py`

## Testing

Preferred test command:

```bash
pytest tests/
```

When adding a new module, add or update a test.

Minimum tests:

- Endpoint-OOD split leakage test.
- Template extraction test.
- Loss masking test.
- Pseudo-template assignment test.
- Audit metric sanity test.

## Coding style

Use simple, readable Python.

Prefer explicit data formats.

Prefer deterministic outputs.

Prefer small functions with clear inputs and outputs.

Do not hide errors with broad except blocks.

Do not suppress warnings that indicate data quality or leakage problems.
