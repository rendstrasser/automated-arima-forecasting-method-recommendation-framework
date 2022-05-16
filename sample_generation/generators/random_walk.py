"""Module for random walk sample generation."""
from typing import List

import numpy as np
from statsmodels.tsa.arima.model import ARIMAResults

from .candidates.baseline_candidate import BaselineCandidate
from .candidates.fixed_order_sarima_candidate import FixedOrderSARIMACandidate
from .gaussian_white_noise import GaussianWhiteNoiseGenerator
from .sample_generator import SampleGenerator


class RandomWalkGenerator(SampleGenerator):
    """A sample generator that generates random walks.

    Parameters
    ----------
    params: SampleGenerationParams
        Parameters that configure the sample generator.
    seed:
        Seed that is used to randomly generate data.
        (default is no seed -> nondeterministic results)
    """

    def get_best_fit_candidates(
        self, fitting_params: dict = None
    ) -> List[BaselineCandidate[ARIMAResults]]:
        """Returns baseline candidates that may be the best fit for a time series.

        For random walk, the best fit is known to be an ARIMA(0,1,0) process.

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
        return [FixedOrderSARIMACandidate({"order": (0, 1, 0)})]

    def simulate(self) -> (np.ndarray, dict):
        """Simulates a random walk time series.

        The variance for the white noise is uniformly random within [0.1, 5).

        Returns
        -------
        y: np.ndarray
            Simulated 1-dimensional time series.
        debugging_params: dict
            A dictionary containing debugging information for the simulation.
        """
        generator = GaussianWhiteNoiseGenerator(self.params, rng=self.rng)
        white_noise, _ = generator.simulate()
        return np.cumsum(white_noise), {}
