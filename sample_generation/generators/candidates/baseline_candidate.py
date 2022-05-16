"""Module for baseline candidate interface."""
from abc import abstractmethod
from typing import Generic, TypeVar

import numpy as np
from numpy.random import SeedSequence

CANDIDATE_TYPE = TypeVar("CANDIDATE_TYPE")


class BaselineCandidate(Generic[CANDIDATE_TYPE]):
    """Interface for candidate that might be the best-fit for a time series."""

    @abstractmethod
    def fit(self, y: np.ndarray, seed: SeedSequence = None) -> CANDIDATE_TYPE:
        """Fits a model on the given time series.

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

    @abstractmethod
    def forecast(self, model: CANDIDATE_TYPE, steps: int) -> np.ndarray:
        """Performs a forecast for the given model.

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

    @abstractmethod
    def aic(self, model: CANDIDATE_TYPE) -> float:
        """Calculate the AIC for the given model.

        Parameters
        ----------
        model: CANDIDATE_TYPE
            Fitted model.

        Returns
        -------
        aic: float
            AIC value of the fitted model.
        """

    @abstractmethod
    def bic(self, model: CANDIDATE_TYPE) -> float:
        """Calculate the BIC for the given model.

        Parameters
        ----------
        model: CANDIDATE_TYPE
            Fitted model.

        Returns
        -------
        aic: float
            BIC value of the fitted model.
        """
