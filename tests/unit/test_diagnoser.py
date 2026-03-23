import numpy as np
from sklearn.dummy import DummyRegressor

from src.diagnoser import Diagnoser, DiagnosisReport


def make_data(n_samples=200, n_features=5, shift_idx=0, shift_amount=2.0):
    rng = np.random.RandomState(42)
    baseline = rng.normal(loc=50.0, scale=5.0, size=(n_samples, n_features))
    current = baseline.copy()
    current[:, shift_idx] += shift_amount
    return baseline, current

def test_basic_diagnosis_no_model():
    baseline, current = make_data(n_features=4, shift_idx=2, shift_amount=3.0)
    names = [f"f{i}" for i in range(4)]
    d = Diagnoser(baseline, names)
    report = d.diagnose(current, model=None, top_k=2)
    assert isinstance(report, DiagnosisReport)
    assert len(report.feature_diagnoses) == 2
    # top feature should be shift_idx (2)
    assert report.feature_diagnoses[0].feature_name == "f2"
    assert report.feature_diagnoses[0].psi_score >= report.feature_diagnoses[1].psi_score

def test_diagnosis_with_dummy_model():
    baseline, current = make_data(n_features=3, shift_idx=1, shift_amount=4.0)
    names = [f"f{i}" for i in range(3)]
    # train a dummy regressor on baseline
    x = baseline
    y = x[:, 0] * 0.1 + x[:, 1] * 0.2 + x[:, 2] * 0.3
    model = DummyRegressor(strategy="mean")
    model.fit(x, y)
    d = Diagnoser(baseline, names)
    report = d.diagnose(current, model=model, top_k=1)
    assert isinstance(report, DiagnosisReport)
    assert len(report.feature_diagnoses) == 1
    assert report.feature_diagnoses[0].feature_name == "f1"
