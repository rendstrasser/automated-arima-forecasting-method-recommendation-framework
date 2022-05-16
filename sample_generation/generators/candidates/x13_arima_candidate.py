"""Module for ARIMA baseline candidate based on X13."""
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from statsmodels.tsa.x13 import x13_arima_select_order

from sample_generation.generators.candidates.sarima_candidate import SARIMACandidate


class X13ARIMACandidate(SARIMACandidate):
    """Baseline candidate based on X13-ARIMA [1].

    References
    ----------
    [1] https://www.census.gov/data/software/x13as.html
    """

    def fit(self, y: np.ndarray, seed: int = None) -> SARIMAXResults:
        """Fits an ARIMA model on the given time series using the X13 method.

        Parameters
        ----------
        y: np.ndarray
            Time series to fit on.
        seed: SeedSequence
            Seed for random number generation for fitting.

        Returns
        -------
        model: SARIMAXResults
            The fitted model.
        """
        dummy_date = "2018-01-01"
        dummy_freq = "M"

        # X13 requires a pandas series with a date index
        # we choose an arbitrary one
        y_series = pd.Series(
            y, index=pd.date_range(dummy_date, periods=len(y), freq=dummy_freq)
        )

        order_results = x13_arima_select_order(
            y_series, maxorder=(4, 2), maxdiff=(2, 1), start="1/1/2011", freq="D",
            print_stdout=False
        )

        print(order_results)

        return SARIMAX(y, order=order_results["order"]).fit(disp=False)
