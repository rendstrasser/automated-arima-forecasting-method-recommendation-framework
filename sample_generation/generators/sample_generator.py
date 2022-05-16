"""Module for sample generator interface."""
from abc import abstractmethod, ABC
from typing import List

import numpy as np
from numpy.random import Generator, SeedSequence
from sklearn.preprocessing import StandardScaler
from statsmodels.base.model import Results

from model.baseline_single import (
    find_best_idx_fit_for_series,
    spawn_seeds_for_series_fit,
)
from .candidates.baseline_candidate import BaselineCandidate
from .sample_generation_params import SampleGenerationParams


class SampleGenerator(ABC):
    """Abstract class for sample generators to simulate random time series.

    This class is intended to be implemented for specific model classes to simulate
    random 1-dimensional time series for models in their model class.

    Parameters
    ----------
    params: SampleGenerationParams
        Parameters that configure the sample generator.
    seed: int
        Seed that is used to randomly generate data.
        Only used if no rng is given.
        (default is no seed -> nondeterministic results)
    rng: Generator
        Random number generator. If not given, will be created based on seed.
    """

    def __init__(
        self, params: SampleGenerationParams, seed: int = None, rng: Generator = None
    ):
        self.params = params
        self.seed = seed
        self.rng = rng if rng else np.random.default_rng(seed)

    def fit(
        self,
        y: np.ndarray,
        ss: SeedSequence,
        evaluation_criteria: str = "bic",
        fitting_params: dict = None,
        n_splits: int = 5,
    ) -> Results:
        """Fits a model on the given time series.

        Parameters
        ----------
        y: np.ndarray
            Time series that the model should be fit on.
        ss: SeedSequence
            Seed sequence for random number generation that is used within fitting.
        evaluation_criteria: str
            The evaluation criteria to use for evaluating candidates of the best fit.
            Default is 'aic', possible are 'aic', 'mape', 'mase'.
        fitting_params: dict
            Parameters that were used for simulation.
            Might be used to infer the hyperparameters for fitting.
        n_splits: int
            Number of CV splits - only used if evaluation_criteria is 'mape' or 'mase'.
            Default is 5.

        Returns
        -------
        fitted_model
            A fitted model results class from the library statsmodels.
        """

        # use the best fit candidates to find the best-fitting model
        candidates = self.get_best_fit_candidates(fitting_params)
        n_candidates_total = len(candidates)

        final_fit_seed = ss.spawn(1)[0]

        if n_candidates_total == 1:
            # early exit as performance improvement
            return candidates[0].fit(y, seed=final_fit_seed)

        # reshape such that our best-fit-finding-method
        # selects the best candidate idx, as it expects a list of lists
        candidates = np.reshape(candidates, (-1, 1))

        fitting_seeds = spawn_seeds_for_series_fit(ss, len(candidates), n_splits)
        best_indices = find_best_idx_fit_for_series(
            y, candidates, fitting_seeds, [evaluation_criteria]
        )

        # only one index will be > -1 as evaluation-criteria is fixed to one item
        best_candidate_idx = best_indices[best_indices > -1][0]

        return candidates[best_candidate_idx][0].fit(y, seed=final_fit_seed)

    @abstractmethod
    def get_best_fit_candidates(
        self, fitting_params: dict = None
    ) -> List[BaselineCandidate[Results]]:
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

    @abstractmethod
    def simulate(self) -> (np.ndarray, dict):
        """Simulates a random time series sample.

        Returns
        -------
        y: np.ndarray
            Simulated 1-dimensional time series.
        debugging_params: dict
            A dictionary containing debugging information for the simulation.
        """

    def simulate_normalized(self) -> np.ndarray:
        """Generates a random time series sample that is normalized to the standard scale.

        Returns
        -------
        y: np.ndarray
            Simulated and 1-dimensional time series that is normalized
            to the standard scale.
        """
        y, _ = self.simulate()

        # scale to ensure comparable simulations
        return self.to_standard_scale(y)

    @staticmethod
    def to_standard_scale(y: np.ndarray) -> np.ndarray:
        """Normalizes a time series to standard scale.

        Parameters
        ----------
        y: np.ndarray
            1-dimensional time series to normalize.

        Returns
        -------
        normalized_y: np.ndarray
            Time series normalized to standard scale
        """
        return StandardScaler().fit_transform(y[..., np.newaxis]).reshape(len(y))
