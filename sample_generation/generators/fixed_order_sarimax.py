"""Module for SARIMAX sample generation with a fixed model order."""
from typing import List

import numpy as np
from numpy.random import Generator
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

from .candidates.baseline_candidate import BaselineCandidate
from .candidates.fixed_order_sarima_candidate import FixedOrderSARIMACandidate
from .default_settings import DEFAULT_MIN_VARIANCE, DEFAULT_MAX_VARIANCE
from .sample_generation_params import SampleGenerationParams
from .sample_generator import SampleGenerator


class FixedOrderSARIMAXGenerator(SampleGenerator):
    """A sample generator that generates random SARIMA time series for a fixed order.

    A SARIMA(p,d,q)(P,D,Q,m) model is an extension of ARIMA which supports seasonality.
    The hyperparameter m gives the seasonal length of the model, e.g., 31 means for
    a time series frequency of 1 day that every month is one season.

    The model order is expected to be given in the model_params field of the params.
    model_params needs to contain an 'order' tuple of 3 elements and might
    contain a 'seasonal_order' tuple of 4 elements.

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
        self,
        params: SampleGenerationParams,
        seed: int = None,
        rng: Generator = None,
        param_condition=None,
    ):

        super().__init__(params, seed, rng)
        empty_dataset = np.zeros(0)

        # initialization param is used in statsmodels\tsa\statespace\representation.py
        self.model = SARIMAX(
            empty_dataset, **self.params.model_params, initialization="diffuse"
        )

        if not param_condition:

            def default_param_condition(_, __):
                return True

            param_condition = default_param_condition

        self.param_condition = param_condition

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
        return [FixedOrderSARIMACandidate(self.params.model_params)]

    def simulate(self) -> (np.ndarray, dict):
        """Simulates a random SARIMAX time series sample.

        The SARIMAX model is created for the model order based on the
        params given at the time of instantiation of this generator.

        Model coefficients are created uniformly in random within [-2, 2).
        The variance of the random error (white noise) is created uniformly in
        random within [0.1, 5). As the coefficients might violate assumptions
        to make the model stationary and invertible, they are transformed with
        statsmodels.tsa.statespace.sarimax.SARIMAX.transform_params to ensure
        stationarity and invertability.

        The actual state shocks are created as a Gaussian white noise time series
        with the variance sampled as described in the previous paragraph.

        Returns
        -------
        y: np.ndarray
            Simulated 1-dimensional time series.
        debugging_params: dict
            A dictionary containing debugging information for the simulation.
        """
        assert self.params.model_params["order"]

        while True:
            (
                full_simulation_params,
                simulation_params,
                state_error_var,
            ) = self._simulate_params()

            # transformed=False ensures that
            # statsmodels.tsa.statespace.sarimax.SARIMAX.transform_params
            # is called to correct the randomly generated params
            # to be valid for the model
            full_simulation_params = self.model.update(
                full_simulation_params, transformed=False, includes_fixed=False
            )

            if self.param_condition(full_simulation_params, self.params.model_params["order"]):
                break

        n_simulations = self.params.n_samples + self.params.burnin

        # generate shocks with rng instead of within
        # simulate where it uses global random functions
        # effectively, no measurement error as of now as obs_cov=0
        measurement_shocks = self.rng.multivariate_normal(
            mean=np.zeros(1), cov=np.atleast_2d(0), size=n_simulations
        )

        state_shocks = self.rng.multivariate_normal(
            mean=np.zeros(1), cov=self.model.ssm.state_cov[0], size=n_simulations
        )

        y = self.model.simulate(
            full_simulation_params,
            n_simulations,
            measurement_shocks=measurement_shocks,
            state_shocks=state_shocks,
            transformed=True,
        )[self.params.burnin :]

        debugging_params = dict()
        debugging_params["model_params"] = self.params.model_params
        debugging_params["not_transformed_coefficients"] = simulation_params
        debugging_params["not_transformed_state_error_var"] = state_error_var
        debugging_params["transformed_params"] = full_simulation_params

        return y, debugging_params

    def _simulate_params(self):
        order = self.params.model_params["order"]
        order_params_count = order[0] + order[2]  # p, q have params

        seasonal_order_contained = "seasonal_order" in self.params.model_params
        seasonal_order_params_count = 0
        if seasonal_order_contained:
            seasonal_order = self.params.model_params["seasonal_order"]
            seasonal_order_params_count = (
                seasonal_order[0] + seasonal_order[2]
            )  # P, Q have parameters

        params_count = order_params_count + seasonal_order_params_count

        simulation_params = self.rng.uniform(low=-2, high=2, size=params_count)

        state_error_var = self.rng.uniform(
            low=DEFAULT_MIN_VARIANCE, high=DEFAULT_MAX_VARIANCE
        )

        full_simulation_params = [*simulation_params, state_error_var]
        return full_simulation_params, simulation_params, state_error_var
