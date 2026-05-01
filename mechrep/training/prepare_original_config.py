"""Prepare runtime config files for the original BioPathNet runner."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _absolute_path(path_value: str, root: Path) -> str:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = root / path
    return str(path.resolve(strict=False))


def materialize_config(config_path: Path, output_path: Path, root: Path) -> dict[str, Any]:
    """Write a runtime config with paths made absolute for BioPathNet.

    The original BioPathNet script changes into the experiment output directory
    before constructing the dataset. Relative dataset paths therefore stop
    resolving from the project root. This helper keeps checked-in configs
    portable and generates an absolute-path runtime copy immediately before
    launch.
    """

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping in config: {config_path}")

    dataset = config.get("dataset")
    if isinstance(dataset, dict) and dataset.get("path"):
        dataset["path"] = _absolute_path(str(dataset["path"]), root)

    if config.get("output_dir"):
        config["output_dir"] = _absolute_path(str(config["output_dir"]), root)

    runtime = config.get("runtime")
    if isinstance(runtime, dict):
        pairwise_eval = runtime.get("pairwise_eval")
        if isinstance(pairwise_eval, dict):
            for key in ("split_dir", "output_dir"):
                if pairwise_eval.get(key):
                    pairwise_eval[key] = _absolute_path(str(pairwise_eval[key]), root)

    pairwise_training = config.get("pairwise_training")
    if isinstance(pairwise_training, dict):
        for key in ("split_dir", "output_dir"):
            if pairwise_training.get(key):
                pairwise_training[key] = _absolute_path(str(pairwise_training[key]), root)

    template_training = config.get("template_training")
    if isinstance(template_training, dict):
        for key in (
            "template_vocab",
            "gold_labels_train",
            "gold_labels_valid",
            "gold_labels_test",
            "pseudo_labels_train",
            "pseudo_labels_valid",
            "pseudo_labels_test",
            "assignment_report",
        ):
            if template_training.get(key):
                template_training[key] = _absolute_path(str(template_training[key]), root)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()

    materialize_config(args.config, args.output, args.root.resolve(strict=False))


if __name__ == "__main__":
    main()
