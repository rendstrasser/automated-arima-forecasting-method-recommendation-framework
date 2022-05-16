"""Module for SARIMA baseline candidate based on Auto-ARIMA."""
import numpy as np
from numpy.random import default_rng
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

from sample_generation.generators.candidates.sarima_candidate import SARIMACandidate


class AutoSARIMACandidate(SARIMACandidate):
    """Seasonal baseline candidate based on auto ARIMA.

    Auto-ARIMA is a method that automatically finds the best-fitting ARIMA model
    for a valid range of model orders [1].

    Parameters
    ----------
    min_order_p: int
        Minimum number of autoregressive terms p.
    max_order_p: int
        Maximum number of autoregressive terms p.
    min_order_d: int
        Minimum number of differences d.
    max_order_d: int
        Maximum number of differences d.
    min_order_q: int
        Minimum number of moving-average terms q.
    max_order_q: int
        Maximum number of moving-average terms q.
    min_order_P: int
        Minimum number of seasonal autoregressive terms P.
    max_order_P: int
        Maximum number of seasonal autoregressive terms P.
    min_order_D: int
        Minimum number of seasonal differences D.
    max_order_D: int
        Maximum number of seasonal differences D.
    min_order_Q: int
        Minimum number of seasonal moving-average terms Q.
    max_order_Q: int
        Maximum number of seasonal moving-average terms Q.
    m: int
        Seasonality.

    References
    ----------
    [1] https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html  # noqa
    """

    def __init__(
        self,
        min_order_p: int,
        max_order_p: int,
        min_order_d: int,
        max_order_d: int,
        min_order_q: int,
        max_order_q: int,
        min_order_P: int,
        max_order_P: int,
        min_order_D: int,
        max_order_D: int,
        min_order_Q: int,
        max_order_Q: int,
        m: int,
    ):
        self.min_order_p = min_order_p
        self.max_order_p = max_order_p
        self.min_order_d = min_order_d
        self.max_order_d = max_order_d
        self.min_order_q = min_order_q
        self.max_order_q = max_order_q
        self.min_order_P = min_order_P
        self.max_order_P = max_order_P
        self.min_order_D = min_order_D
        self.max_order_D = max_order_D
        self.min_order_Q = min_order_Q
        self.max_order_Q = max_order_Q
        self.m = m

    def fit(self, y: np.ndarray, seed: int = None) -> SARIMAXResults:
        """Fits a SARIMA model on the given time series using auto arima.

        Parameters
        ----------
        y: np.ndarray
            Time series to fit on.
        seed: SeedSequence
            Seed for random number generation for fitting.

        Returns
        -------
        model: CANDIDATE_TYPE
            The fitted model.
        """
        rng = default_rng(seed)
        return auto_arima(
            y,
            start_p=self.min_order_p,
            max_p=self.max_order_p,
            start_q=self.min_order_q,
            max_q=self.max_order_q,
            start_P=self.min_order_P,
            max_P=self.max_order_P,
            start_Q=self.min_order_Q,
            max_Q=self.max_order_Q,
            max_d=self.max_order_D,
            max_D=self.max_order_D,
            seasonal=True,
            criterion="bic",
            stepwise=False,
            m=self.m,
            random=rng,
            suppress_warnings=True,
            error_action="ignore",
        ).arima_res_
