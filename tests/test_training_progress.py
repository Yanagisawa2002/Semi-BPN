import pytest

from mechrep.training.progress import EarlyStopper, infer_metric_mode, normalize_monitor_metric


def test_early_stopper_tracks_max_metric_with_patience():
    stopper = EarlyStopper("valid_auprc", patience=2)

    assert normalize_monitor_metric("valid_auprc") == "auprc"
    assert infer_metric_mode("valid_auprc") == "max"
    assert stopper.observe(0.4, epoch=1) == (True, False)
    assert stopper.observe(0.39, epoch=2) == (False, False)
    assert stopper.observe(0.38, epoch=3) == (False, True)
    assert stopper.best_epoch == 1


def test_early_stopper_infers_loss_as_min_metric():
    stopper = EarlyStopper("valid_loss_total", patience=1)

    assert infer_metric_mode("valid_loss_total") == "min"
    assert stopper.observe(0.5, epoch=1) == (True, False)
    assert stopper.observe(0.6, epoch=2) == (False, True)


def test_early_stopper_rejects_bad_mode():
    with pytest.raises(ValueError, match="early stopping mode"):
        EarlyStopper("valid_auprc", patience=1, mode="median")
