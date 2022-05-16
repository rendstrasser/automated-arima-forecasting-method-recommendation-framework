"""Module for a linear baseline candidate."""
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLSResults, OLS

from sample_generation.generators.candidates.baseline_candidate import BaselineCandidate


class LinearCandidate(BaselineCandidate[OLSResults]):
    """Baseline candidate for a linear model.

    Parameters
    ----------
    model_params: dict
        Additional model parameters for the OLS model.
    """

    def __init__(self, model_params: dict = None):
        if model_params is None:
            model_params = {}

        self.model_params = model_params

    def fit(self, y: np.ndarray, seed: int = None) -> OLSResults:
        """Fits an OLS model on the given time series.

        Parameters
        ----------
        y: np.ndarray
            Time series to fit on.
        seed: SeedSequence
            Seed for random number generation for fitting.

        Returns
        -------
        model: OLSResults
            The fitted model.
        """
        df = pd.DataFrame({"x": np.arange(len(y)), "y": y})
        return OLS.from_formula("y ~ x", df, **self.model_params).fit()

    def forecast(self, model: OLSResults, steps: int) -> np.ndarray:
        """Performs a forecast for the given model (not implemented here).

        Parameters
        ----------
        model: CANDIDATE_TYPE
            Fitted model.
        steps: int
            Steps to forecast into the future.

        Returns
        -------
        forecast: np.ndarray
            A forecast array.
        """
        raise NotImplementedError()

    def aic(self, model: OLSResults):
        """Calculate the AIC for the given linear model.

        Parameters
        ----------
        model: OLSResults
            Fitted model.

        Returns
        -------
        aic: float
            AIC value of the fitted model.
        """
        return model.aic

    def bic(self, model: OLSResults):
        """Calculate the BIC for the given linear model.

        Parameters
        ----------
        model: OLSResults
            Fitted model.

        Returns
        -------
        aic: float
            BIC value of the fitted model.
        """
        return model.bic
