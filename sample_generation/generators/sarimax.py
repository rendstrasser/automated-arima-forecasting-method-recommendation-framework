"""Module for SARIMA sample generation."""
from typing import List

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

from .candidates.auto_arima_candidate import AutoARIMACandidate
from .candidates.auto_sarima_candidate import AutoSARIMACandidate
from .candidates.baseline_candidate import BaselineCandidate
from .candidates.fixed_order_sarima_candidate import FixedOrderSARIMACandidate
from .candidates.x13_arima_candidate import X13ARIMACandidate
from .default_settings import (
    DEFAULT_SARIMAX_MIN_p,
    DEFAULT_SARIMAX_MAX_p,
    DEFAULT_SARIMAX_MIN_d,
    DEFAULT_SARIMAX_MAX_d,
    DEFAULT_SARIMAX_MIN_q,
    DEFAULT_SARIMAX_MAX_q,
    DEFAULT_SARIMAX_MIN_P,
    DEFAULT_SARIMAX_MAX_P,
    DEFAULT_SARIMAX_MIN_D,
    DEFAULT_SARIMAX_MAX_D,
    DEFAULT_SARIMAX_MIN_Q,
    DEFAULT_SARIMAX_MAX_Q,
    DEFAULT_SARIMAX_M_CANDIDATES,
)
from .fixed_order_sarimax import FixedOrderSARIMAXGenerator
from .sample_generation_params import SampleGenerationParams
from .sample_generator import SampleGenerator


class SARIMAXGenerator(SampleGenerator):
    """A sample generator that generates random SARIMA(p,d,q)(P,D,Q,m) time series.

    A SARIMA(p,d,q)(P,D,Q,m) model is an extension of ARIMA which supports seasonality.
    The hyperparameter m gives the seasonal length of the model, e.g., 31 means for
    a time series frequency of 1 day that every month is one season.

    Parameters
    ----------
    params: SampleGenerationParams
        Parameters that configure the sample generator.
    min_order_p: int
        Minimum integer value for hyperparameter p - inclusive. (default: 1)
    max_order_p: int
        Maximum integer value for hyperparameter p - inclusive. (default: 5)
    min_order_d: int
        Minimum integer value for hyperparameter d - inclusive. (default: 1)
    max_order_d: int
        Maximum integer value for hyperparameter d - inclusive. (default: 2)
    min_order_q: int
        Minimum integer value for hyperparameter q - inclusive. (default: 1)
    max_order_q: int
        Maximum integer value for hyperparameter q - inclusive. (default: 5)
    min_order_P: int
        Minimum integer value for hyperparameter P - inclusive. (default: 1)
    max_order_P: int
        Maximum integer value for hyperparameter P - inclusive. (default: 2)
    min_order_D: int
        Minimum integer value for hyperparameter D - inclusive. (default: 0)
    max_order_D: int
        Maximum integer value for hyperparameter D - inclusive. (default: 1)
    min_order_Q: int
        Minimum integer value for hyperparameter Q - inclusive. (default: 1)
    max_order_Q: int
        Maximum integer value for hyperparameter Q - inclusive. (default: 2)
    min_order_m: int
        Minimum integer value for hyperparameter m - inclusive. (default: 7)
    max_order_m: int
        Maximum integer value for hyperparameter m - inclusive. (default: 31)
    seed:
        Seed that is used to randomly generate data.
        (default is no seed -> nondeterministic results)

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
    """

    def __init__(
        self,
        params: SampleGenerationParams,
        min_order_p=DEFAULT_SARIMAX_MIN_p,
        max_order_p=DEFAULT_SARIMAX_MAX_p,
        min_order_d=DEFAULT_SARIMAX_MIN_d,
        max_order_d=DEFAULT_SARIMAX_MAX_d,
        min_order_q=DEFAULT_SARIMAX_MIN_q,
        max_order_q=DEFAULT_SARIMAX_MAX_q,
        min_order_P=DEFAULT_SARIMAX_MIN_P,
        max_order_P=DEFAULT_SARIMAX_MAX_P,
        min_order_D=DEFAULT_SARIMAX_MIN_D,
        max_order_D=DEFAULT_SARIMAX_MAX_D,
        min_order_Q=DEFAULT_SARIMAX_MIN_Q,
        max_order_Q=DEFAULT_SARIMAX_MAX_Q,
        m_candidates=DEFAULT_SARIMAX_M_CANDIDATES,
        seed: int = None,
    ):
        super().__init__(params, seed)

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
        self.m_candidates = m_candidates
        self.is_seasonal = len(self.m_candidates) != 0

    def get_best_fit_candidates(
        self, fitting_params: dict = None
    ) -> List[BaselineCandidate[SARIMAXResults]]:
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

        if True:
            return [X13ARIMACandidate()]

        if fitting_params is not None:
            model_params = fitting_params["model_params"]
            return [FixedOrderSARIMACandidate(model_params)]

        if not self.is_seasonal:
            return [
                AutoARIMACandidate(
                    self.min_order_p,
                    self.max_order_p,
                    self.min_order_d,
                    self.max_order_d,
                    self.min_order_q,
                    self.max_order_q,
                )
            ]

        candidates = [None] * len(self.m_candidates)

        for i, m in enumerate(self.m_candidates):
            candidates[i] = AutoSARIMACandidate(
                self.min_order_p,
                self.max_order_p,
                self.min_order_d,
                self.max_order_d,
                self.min_order_q,
                self.max_order_q,
                self.min_order_P,
                self.max_order_P,
                self.min_order_D,
                self.max_order_D,
                self.min_order_Q,
                self.max_order_Q,
                m,
            )

        return candidates

    def simulate(self) -> (np.ndarray, dict):
        """Simulates a random SARIMAX time series sample.

        The SARIMAX model is created by choosing uniformly
        random integers in the ranges that are given at the instantiation of the
        generator (min and max are both inclusive) for the model order.

        A seasonal order is only sampled if the min_order_m is larger than 1.

        To create random parameters for the model,
        the FixedOrderSARIMAXGenerator is used.

        Returns
        -------
        y: np.ndarray
            Simulated 1-dimensional time series.
        debugging_params: dict
            A dictionary containing debugging information for the simulation.
        """

        p = self.rng.integers(low=self.min_order_p, high=self.max_order_p + 1)
        d = self.rng.integers(low=self.min_order_d, high=self.max_order_d + 1)
        q = self.rng.integers(low=self.min_order_q, high=self.max_order_q + 1)

        extended_model_params = dict(self.params.model_params)
        extended_model_params["order"] = (p, d, q)

        if self.is_seasonal:
            P = self.rng.integers(low=self.min_order_P, high=self.max_order_P + 1)
            D = self.rng.integers(low=self.min_order_D, high=self.max_order_D + 1)
            Q = self.rng.integers(low=self.min_order_Q, high=self.max_order_Q + 1)
            m = self.rng.choice(self.m_candidates)
            extended_model_params["seasonal_order"] = (P, D, Q, m)

        params = SampleGenerationParams(
            n_samples=self.params.n_samples,
            model_params=extended_model_params,
            burnin=self.params.burnin,
        )
        generator = FixedOrderSARIMAXGenerator(
            params, rng=self.rng, param_condition=self._are_params_valid
        )
        y, sarimax_debugging_params = generator.simulate()

        return y, sarimax_debugging_params

    def _are_params_valid(self, params: np.ndarray, order: List[int]) -> bool:
        """Decides if the full parameters of a SARIMA model are valid.

        This function is intended to be overwritten by subclasses.
        The default implementation renders every parameter set valid.

        If the parameters are not valid, the parameters will be randomly regenerated,
        until they are valid.

        Parameters
        ----------
        params: np.ndarray
            The parameters of the model (coefficients + error variance).
        order:
            Model order, e.g. (2,1,2) for ARIMA.

        Returns
        -------
        params_valid: bool
            Indicator, if params are valid.
        """
        return True
