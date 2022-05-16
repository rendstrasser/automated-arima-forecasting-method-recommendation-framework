"""Module for ARMA sample generation with lower model order."""
from sample_generation.generators.arima import ARIMAGenerator
from sample_generation.generators.default_settings import (
    DEFAULT_SARIMAX_MIN_p,
    DEFAULT_SARIMAX_MIN_q,
)
from sample_generation.generators.sample_generation_params import SampleGenerationParams


class ARMALowGenerator(ARIMAGenerator):
    """A sample generator that generates random ARMA(p,q) time series samples.

    The difference to ARMAGenerator is that it generates lower order ARMA models
    by default.

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
        Maximum integer value for hyperparameter p - inclusive. (default: 2)
    min_order_q: int
        Minimum integer value for hyperparameter q - inclusive. (default: 1)
    max_order_q: int
        Maximum integer value for hyperparameter q - inclusive. (default: 2)
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
        max_order_p=2,
        min_order_q=DEFAULT_SARIMAX_MIN_q,
        max_order_q=2,
        seed: int = None,
    ):
        super().__init__(
            params, min_order_p, max_order_p, 0, 0, min_order_q, max_order_q, seed
        )
