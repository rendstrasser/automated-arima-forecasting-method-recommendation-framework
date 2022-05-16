"""Module for legacy class."""
import numpy as np
from statsmodels.tsa.arima_process import arma_generate_sample

from ..sample_generation_params import SampleGenerationParams
from ..sarimax import SARIMAXGenerator


class SpecialARGenerateARMASamplesGenerator(SARIMAXGenerator):
    """AR(p) generator that uses arma_generate_sample from statsmodels.

    This implementation was used to check how time series samples look when
    generated with arma_generate_sample compared to SARIMAX(...).simulate.

    Stationarity is enforced through brute-force.

    No longer used.
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

        # ensure stationary coefficients
        while True:
            lag_coefficients = self.rng.normal(size=p) - 0.5

            # formula for unit root check is
            # 1 - AR coefficients
            # therefore prepend 1 and subtract lag coefficients
            root_coeff = np.append(1, -lag_coefficients)

            if np.max(np.abs(np.roots(root_coeff))) < 1:
                break

        ar_coefficients = np.append(1, -lag_coefficients)
        noise_sd = self.rng.uniform(low=0.1, high=2)

        y = arma_generate_sample(
            ar=ar_coefficients,
            ma=[1],
            nsample=self.params.n_samples,
            scale=noise_sd,
            distrvs=self.rng.standard_normal,
            burnin=self.params.burnin,
        )

        debugging_data = {
            "model_params": {"order": (p, 0, 0)},
            "ar_coefficients": lag_coefficients,
            "noise_sd:": noise_sd,
        }

        return y, debugging_data
