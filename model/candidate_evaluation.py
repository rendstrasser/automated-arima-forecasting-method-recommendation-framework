"""Module for fit evaluation functions."""
import numpy as np
from sklearn.metrics import mean_absolute_error

VALID_FIT_EVALUATION_CRITERIA = ["mape", "mase", "aic", "bic"]


def mean_absolute_scaled_error(y_true, y_pred, y_train):
    """Calculates the mean absolute scaled error (MASE) [1].

    MASE is an alternative to MAPE which has similar properties,
    but is robust if the predicated data point is 0,
    and many consider it as the best error metric in modern time series
    forecasting evaluation [2].

    Parameters
    ----------
    candidate: Candidate[CANDIDATE_TYPE]
        Candidate to calculate the MASE for the time series
    y_train: np.ndarray
        Time series to fit candidate on.
    y_test:
        Actual succeeding values of y_train that can be used to check the forecast
        against.

    Returns
    -------
    mase: float
        MASE value.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
    [2] https://robjhyndman.com/papers/foresight.pdf
    """

    e_t = y_true - y_pred
    scale = mean_absolute_error(y_train[1:], y_train[:-1])
    return np.mean(np.abs(e_t / scale))
