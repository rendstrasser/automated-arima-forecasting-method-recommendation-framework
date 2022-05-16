"""Module for gaussian white noise sample generation."""
from typing import List

import numpy as np
from numpy.random import Generator
from statsmodels.tsa.arima.model import ARIMAResults

from .candidates.baseline_candidate import BaselineCandidate
from .candidates.fixed_order_sarima_candidate import FixedOrderSARIMACandidate
from .default_settings import DEFAULT_MIN_SD, DEFAULT_MAX_SD
from .sample_generation_params import SampleGenerationParams
from .sample_generator import SampleGenerator


class GaussianWhiteNoiseGenerator(SampleGenerator):
    """A sample generator that generates gaussian white noise.

    Parameters
    ----------
    params: SampleGenerationParams
        Parameters that configure the sample generator.
    seed:
        Seed that is used to randomly generate data.
        (default is no seed -> nondeterministic results)
    rng: Generator
        Random number generator. If not given, will be created based on seed.
    """

    def __init__(
        self, params: SampleGenerationParams, seed: int = None, rng: Generator = None
    ):
        super().__init__(params, seed, rng)

    def get_best_fit_candidates(
        self, fitting_params: dict = None
    ) -> List[BaselineCandidate[ARIMAResults]]:
        """Returns baseline candidates that may be the best fit for a time series.

        For Gaussian White Noise, the best fit is known to be an ARIMA(0,0,0) process.

        Parameters
        ----------
        fitting_params: dict
            Parameters that were used for simulation.
            Might be used to infer the hyperparameters for fitting.

        Returns
        -------
        candidates: List[BaselineCandidate[SARIMAXResults]]
            A list of candidates that may yield the best fitting
            model for a time series.
        """
        return [FixedOrderSARIMACandidate({"order": (0, 0, 0)})]

    def simulate(self) -> (np.ndarray, dict):
        """Simulates a gaussian white noise time series.

        The variance for the white noise is uniformly random within [0.1, 5).

        Returns
        -------
        y: np.ndarray
            Simulated 1-dimensional time series.
        debugging_params: dict
            A dictionary containing debugging information for the simulation.
        """
        sd = self.rng.uniform(low=DEFAULT_MIN_SD, high=DEFAULT_MAX_SD)
        return self.rng.normal(size=self.params.n_samples, scale=sd), {}
