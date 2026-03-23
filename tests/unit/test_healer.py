import numpy as np

from src.healer import Healer


class FakeModel:
    """
    Simple test double implementing evaluate(x, y) and retrain(x, y).
    retrain returns a new FakeModel with a configured accuracy improvement (delta).
    evaluate returns a dict {'accuracy': ...}.
    """

    def __init__(self, base_accuracy: float = 0.8, retrain_delta: float = 0.0):
        self._accuracy = float(base_accuracy)
        self._retrain_delta = float(retrain_delta)

    def evaluate(self, x_holdout, y_holdout):
        # ignore data; return accuracy and dummy metrics
        return {"accuracy": self._accuracy}

    def retrain(self, x_train, y_train):
        # produce a challenger with adjusted accuracy
        return FakeModel(base_accuracy=self._accuracy + self._retrain_delta, retrain_delta=0.0)


def make_dummy_data(n=50, features=3):
    rng = np.random.RandomState(0)
    x = rng.normal(size=(n, features))
    y = rng.normal(size=(n,))
    return x, y


def test_promote_when_challenger_is_better():
    # champion at 0.80, retrain produces challenger at 0.85 -> should PROMOTE (default min_improvement 0.02)
    champion = FakeModel(base_accuracy=0.80, retrain_delta=0.05)
    healer = Healer(champion)
    x_train, y_train = make_dummy_data()
    x_holdout, y_holdout = make_dummy_data()
    outcome = healer.heal(x_train, y_train, x_holdout, y_holdout)
    assert outcome.action == "PROMOTE"
    assert outcome.challenger_metrics.accuracy > outcome.champion_metrics.accuracy


def test_rollback_when_challenger_is_worse():
    champion = FakeModel(base_accuracy=0.90, retrain_delta=-0.05)
    healer = Healer(champion)
    x_train, y_train = make_dummy_data()
    x_holdout, y_holdout = make_dummy_data()
    outcome = healer.heal(x_train, y_train, x_holdout, y_holdout)
    assert outcome.action == "ROLLBACK"
    assert outcome.challenger_metrics.accuracy < outcome.champion_metrics.accuracy


def test_no_action_when_improvement_below_threshold():
    # champion 0.80, challenger 0.815 (improvement 0.015) -> NO_ACTION if min_improvement default 0.02
    champion = FakeModel(base_accuracy=0.80, retrain_delta=0.015)
    healer = Healer(champion)
    x_train, y_train = make_dummy_data()
    x_holdout, y_holdout = make_dummy_data()
    outcome = healer.heal(x_train, y_train, x_holdout, y_holdout)
    assert outcome.action == "NO_ACTION"
    assert 0 <= (outcome.challenger_metrics.accuracy - outcome.champion_metrics.accuracy) < 0.02
