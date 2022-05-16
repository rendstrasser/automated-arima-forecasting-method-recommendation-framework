"""Module foran abstract SARIMA baseline candidate."""
from abc import abstractmethod

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

from sample_generation.generators.candidates.baseline_candidate import BaselineCandidate


class SARIMACandidate(BaselineCandidate[SARIMAXResults]):
    """Abstract candidate for SARIMA which handles the forecasting and AIC."""

    @abstractmethod
    def fit(self, y: np.ndarray, seed: int = None) -> SARIMAXResults:
        """Fits a SARIMA model on the given time series.

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

    def forecast(self, model: SARIMAXResults, steps: int) -> np.ndarray:
        """Performs a forecast for the given model.

        Parameters
        ----------
        model: SARIMAXResults
            Fitted model.
        steps: int
            Steps to forecast into the future.

        Returns
        -------
        forecast: np.ndarray
            A forecast array.
        """
        return model.forecast(steps)

    def aic(self, model: SARIMAXResults):
        """Calculate the AIC for the given SARIMA model.

        Parameters
        ----------
        model: SARIMAXResults
            Fitted model.

        Returns
        -------
        aic: float
            AIC value of the fitted model.
        """
        return model.aic

    def bic(self, model: SARIMAXResults):
        """Calculate the BIC for the given SARIMA model.

        Parameters
        ----------
        model: SARIMAXResults
            Fitted model.

        Returns
        -------
        aic: float
            BIC value of the fitted model.
        """
        return model.bic
