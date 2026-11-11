import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.data import apply_label, process_data
from ml.model import compute_model_metrics, inference, train_model

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="module")
def processed_data():
    """Load a small sample of census data and process it for tests."""
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "census.csv")
    data = pd.read_csv(data_path).sample(n=1000, random_state=42)
    train, test = train_test_split(
        data, test_size=0.20, random_state=42, stratify=data["salary"]
    )
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    return X_train, y_train, X_test, y_test


def test_process_data_shapes(processed_data):
    """Processed train and test data have matching rows and binary labels."""
    X_train, y_train, X_test, y_test = processed_data
    assert X_train.shape[0] == y_train.shape[0] == 800
    assert X_test.shape[0] == y_test.shape[0] == 200
    assert X_train.shape[1] == X_test.shape[1]
    assert set(np.unique(y_train)).issubset({0, 1})


def test_train_model_algorithm(processed_data):
    """train_model returns a RandomForestClassifier."""
    X_train, y_train, _, _ = processed_data
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


def test_inference_output(processed_data):
    """inference returns a numpy array of 0/1 predictions, one per row."""
    X_train, y_train, X_test, _ = processed_data
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_test.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics():
    """compute_model_metrics returns the expected values on a known example."""
    y = np.array([1, 1, 0, 0])
    preds = np.array([1, 0, 1, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(0.5)


def test_apply_label():
    """apply_label converts binary predictions to salary strings."""
    assert apply_label(np.array([1])) == ">50K"
    assert apply_label(np.array([0])) == "<=50K"
