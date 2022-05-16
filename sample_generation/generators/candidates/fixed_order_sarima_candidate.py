"""Module for SARIMA baseline candidate for a fixed order."""
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

from sample_generation.generators.candidates.sarima_candidate import SARIMACandidate


class FixedOrderSARIMACandidate(SARIMACandidate):
    """Baseline candidate for a SARIMA model of fixed order.

    Parameters
    ----------
    model_params: dict
        Model parameters for the SARIMA model.
        Needs to contain 'order' and if seasonality is required also 'seasonal_order'.
    """

    def __init__(self, model_params: dict):
        self.model_params = model_params

    def fit(self, y: np.ndarray, seed: int = None) -> SARIMAXResults:
        """Fits a SARIMA model with a fixed order on the given time series.

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
        return SARIMAX(y, **self.model_params).fit(disp=False)
