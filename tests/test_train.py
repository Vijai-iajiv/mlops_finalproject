from train import evaluate


def test_evaluate_perfect_predictions():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 1, 0]
    y_proba = [0.05, 0.9, 0.8, 0.1]

    metrics = evaluate(y_true, y_pred, y_proba)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_evaluate_returns_expected_keys():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 0, 0, 1]
    y_proba = [0.2, 0.4, 0.1, 0.7]

    metrics = evaluate(y_true, y_pred, y_proba)

    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
