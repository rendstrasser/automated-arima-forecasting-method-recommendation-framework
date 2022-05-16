"""Module for ARMA sample generation."""
from typing import List

import numpy as np

from sample_generation.generators.arma import ARMAGenerator
from sample_generation.generators.sample_generation_params import SampleGenerationParams


class ARMASignificantGenerator(ARMAGenerator):
    """A sample generator that generates random ARMA(p,q) time series samples.

    The difference to ARMAGenerator is that a condition is applied on the parameters
    of the ARMA model. The absolute value of the last AR and MA coefficient needs to
    be higher than 0.8 to make them 'significant'.

    An ARMA(p,q) model is a mixed autoregressive-moving-average model
    with p autoregressive and q moving average terms [1].

    It combines the AR(p) and MA(q) into one model.

    Parameters
    ----------
    params: SampleGenerationParams
        Parameters that configure the sample generator.
    min_order_p: int
        Minimum integer value for hyperparameter p - inclusive. (default: 1)
    max_order_p: int
        Maximum integer value for hyperparameter p - inclusive. (default: 5)
    min_order_q: int
        Minimum integer value for hyperparameter q - inclusive. (default: 1)
    max_order_q: int
        Maximum integer value for hyperparameter q - inclusive. (default: 5)
    seed:
        Seed that is used to randomly generate data.
        (default is no seed -> nondeterministic results)

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model
    """

    def __init__(
        self, params: SampleGenerationParams, seed: int = None,
    ):
        super().__init__(params, seed=seed)

    def _are_params_valid(self, params: np.ndarray, order: List[int]) -> bool:
        """Decides if the full parameters of a SARIMA model are valid.

        For this implementation, the parameters are valid if the last AR and last MA
        coefficient are both greater than 0.8 in absolute terms, i.e., they
        cause "significant" AR and MA behavior in their last coefficient.

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

        significance_threshold = 0.8

        n_ar = order[0]

        # last element is variance
        coefficients = params[:-1]

        last_ar_coeff = coefficients[n_ar - 1]
        last_ma_coeff = coefficients[-1]

        return (
            abs(last_ar_coeff) >= significance_threshold
            and abs(last_ma_coeff) >= significance_threshold
        )
