"""Module for linear sample generation."""
from typing import List

import numpy as np
from statsmodels.regression.linear_model import OLSResults

from .candidates.baseline_candidate import BaselineCandidate
from .candidates.linear_candidate import LinearCandidate
from .sample_generation_params import SampleGenerationParams
from .sample_generator import SampleGenerator


class LinearGenerator(SampleGenerator):
    """A sample generator that generates random linear time series.

    The time series are generated for a linear model, i.e.,
    y = kx + d + e,
    where y is the simulated time series, x is time, k is the slope,
    d is the intercept and e is normally distributed random error.

    Parameters
    ----------
    params: SampleGenerationParams
        Parameters that configure the sample generator.
    seed:
        Seed that is used to randomly generate data.
        (default is no seed -> nondeterministic results)
    include_error:
        Defines if a normally distributed random error should be added to the
        linear simulation. (default: True)
    """

    def __init__(
        self, params: SampleGenerationParams, seed: int = None, include_error=True
    ):
        super().__init__(params, seed)

        self.include_error = include_error

    def get_best_fit_candidates(
        self, fitting_params: dict = None
    ) -> List[BaselineCandidate[OLSResults]]:
        """Returns baseline candidates that may be the best fit for a time series.

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
        return [LinearCandidate(self.params.model_params)]

    def simulate(self) -> (np.ndarray, dict):
        """Simulates a random linear time series sample.

        Samples a slope and intercept uniformly within [-3, 3) and normally distributed
        error for which the variance is uniformly sampled within [0.01, 5).

        Returns
        -------
        y: np.ndarray
            Simulated 1-dimensional time series.
        debugging_params: dict
            A dictionary containing debugging information for the simulation.
        """
        n_samples = self.params.n_samples

        # Linear regression models assume normally distributed error
        if self.include_error:
            error_sd = self.rng.uniform(low=0.01, high=5)
            error = self.rng.normal(scale=error_sd, size=n_samples)
        else:
            error_sd = 0
            error = np.zeros(n_samples)

        beta = self.rng.uniform(low=-3, high=3)
        intercept = self.rng.uniform(low=-3, high=3)
        y_true = beta * np.arange(n_samples) + intercept
        y = y_true + error

        debugging_params = {"beta": beta, "intercept": intercept, "error_sd": error_sd}

        return y, debugging_params
