"""Module for the baseline model."""
from dataclasses import dataclass

import numpy as np


@dataclass
class BaselineResult:
    """Holds a result of a baseline evaluation on a validation set."""

    # Accuracy score of the baseline model
    score: float

    # Actual labels for the predictions
    labels: np.ndarray

    # Predicted labels of the baseline model
    predictions: np.ndarray
