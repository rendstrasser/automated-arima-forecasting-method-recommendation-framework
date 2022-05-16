"""Module for legacy class."""
import numpy as np

from ..sample_generation_params import SampleGenerationParams
from ..sarimax import SARIMAXGenerator


class SpecialCustomAr1ImplGenerator(SARIMAXGenerator):
    """Completely custom AR(1) simulation implementation, no use of statsmodels.

    This implementation was used to check custom AR(1) time series against those
    from statsmodels SARIMAX(1,0,0).simulate.

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
        n_samples = self.params.n_samples + self.params.burnin

        white_noise_sigma = self.rng.uniform(low=0.1, high=2)
        white_noise = self.rng.normal(scale=white_noise_sigma, size=n_samples)

        first_coeff = 0.75

        y = white_noise
        y[1:] += first_coeff * white_noise[:-1]

        debugging_data = {
            "model_params": {"order": (1, 0, 0)},
            "ar_coefficients": [first_coeff],
            "noise_sd:": white_noise_sigma,
        }

        return y[self.params.burnin :], debugging_data
