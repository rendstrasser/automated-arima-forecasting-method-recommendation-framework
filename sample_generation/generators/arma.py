"""Module for ARMA sample generation."""

from .arima import ARIMAGenerator
from .default_settings import (
    DEFAULT_SARIMAX_MIN_p,
    DEFAULT_SARIMAX_MAX_p,
    DEFAULT_SARIMAX_MIN_q,
    DEFAULT_SARIMAX_MAX_q,
)
from .sample_generation_params import SampleGenerationParams


class ARMAGenerator(ARIMAGenerator):
    """A sample generator that generates random ARMA(p,q) time series samples.

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
        self,
        params: SampleGenerationParams,
        min_order_p=DEFAULT_SARIMAX_MIN_p,
        max_order_p=DEFAULT_SARIMAX_MAX_p,
        min_order_q=DEFAULT_SARIMAX_MIN_q,
        max_order_q=DEFAULT_SARIMAX_MAX_q,
        seed: int = None,
    ):
        super().__init__(
            params, min_order_p, max_order_p, 0, 0, min_order_q, max_order_q, seed
        )
