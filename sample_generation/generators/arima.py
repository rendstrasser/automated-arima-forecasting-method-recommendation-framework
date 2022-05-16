"""Module for ARIMA sample generation."""

from .default_settings import (
    DEFAULT_SARIMAX_MIN_d,
    DEFAULT_SARIMAX_MAX_d,
    DEFAULT_SARIMAX_MIN_p,
    DEFAULT_SARIMAX_MAX_p,
    DEFAULT_SARIMAX_MIN_q,
    DEFAULT_SARIMAX_MAX_q,
)
from .sample_generation_params import SampleGenerationParams
from .sarimax import SARIMAXGenerator


class ARIMAGenerator(SARIMAXGenerator):
    """A sample generator that generates random ARIMA(p,d,q) time series samples.

    An ARIMA(p,d,q) model is a mixed autoregressive integrated moving-average model
    with p autoregressive and q moving average terms and d differences [1].

    It combines the AR(p) and MA(q) into one model and differences the ARMA(p,q) model
    d times to get the ARIMA model.

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
        seed: int = None,
    ):
        super().__init__(
            params,
            min_order_p,
            max_order_p,
            min_order_d,
            max_order_d,
            min_order_q,
            max_order_q,
            0,
            0,  # P
            0,
            0,  # D
            0,
            0,  # Q
            [],  # m candidates
            seed,
        )
