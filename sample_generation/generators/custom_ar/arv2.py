"""Module for legacy class."""
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ..sample_generation_params import SampleGenerationParams
from ..sarimax import SARIMAXGenerator


class SpecialARCustomStationaryImplGenerator(SARIMAXGenerator):
    """AR(p) generator that enforces stationarity through brute-force.

    This implementation was used to test how we could do without the param
    transformation from statsmodels. The result was that the brute-force method
    goes through a lot of random parameter combinations until it finds a valid one,
    therefore it is not feasible.
    """

    def __init__(
        self,
        params: SampleGenerationParams,
        min_order_p=1,
        max_order_p=10,
        seed: int = None,
    ):
        super().__init__(
            params, min_order_p, max_order_p, 0, 0, 0, 0, 0, 0, seed  # d  # q  # m
        )

    def simulate(self) -> (np.ndarray, dict):
        """Not documented."""
        p = self.rng.integers(low=self.min_order_p, high=self.max_order_p + 1)

        initial = self.rng.normal(size=self.params.burnin)
        model = SARIMAX(initial, order=(p, 0, 0))

        while True:
            ar_coefficients = self.rng.normal(size=p) - 0.5

            # formula for unit root check is
            # 1 - AR coefficients
            # therefore prepend 1 and subtract lag coefficients
            root_coeff = np.append(1, -ar_coefficients)

            if np.max(np.abs(np.roots(root_coeff))) < 1:
                break

        state_error_variance = self.rng.uniform(low=0.1, high=2)

        params = np.append(ar_coefficients, state_error_variance)

        model.update(params=params)

        n_simulations = self.params.n_samples + self.params.burnin

        # generate shocks with rng instead of within simulate
        # where it uses global random functions
        # effectively, no measurement error as of now as obs_cov=0
        measurement_shocks = self.rng.multivariate_normal(
            mean=np.zeros(1), cov=np.atleast_2d(0), size=n_simulations
        )

        state_shocks = self.rng.multivariate_normal(
            mean=np.zeros(1), cov=[[state_error_variance]], size=n_simulations
        )

        y = model.simulate(
            params,
            n_simulations,
            measurement_shocks=measurement_shocks,
            state_shocks=state_shocks,
            transformed=True,
        )[self.params.burnin :]

        debugging_data = {
            "model_params": {"order": (p, 0, 0)},
            "ar_coefficients": ar_coefficients,
            "state_error_variance:": state_error_variance,
        }

        return y, debugging_data
