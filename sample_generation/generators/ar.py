"""Module for AR sample generation."""
from .arma import ARMAGenerator
from .default_settings import DEFAULT_SARIMAX_MIN_p, DEFAULT_SARIMAX_MAX_p
from .sample_generation_params import SampleGenerationParams


class ARGenerator(ARMAGenerator):
    """A sample generator that generates random AR(p) time series samples.

    An AR(p) model is an autoregressive model with p autoregressive terms [1].

    Parameters
    ----------
    params: SampleGenerationParams
        Parameters that configure the sample generator.
    min_order_p: int
        Minimum integer value for hyperparameter p - inclusive. (default: 1)
    max_order_p: int
        Maximum integer value for hyperparameter p - inclusive. (default: 5)
    seed:
        Seed that is used to randomly generate data.
        (default is no seed -> nondeterministic results)

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Autoregressive_model
    """

    def __init__(
        self,
        params: SampleGenerationParams,
        min_order_p: int = DEFAULT_SARIMAX_MIN_p,
        max_order_p: int = DEFAULT_SARIMAX_MAX_p,
        seed: int = None,
    ):
        super().__init__(params, min_order_p, max_order_p, 0, 0, seed)
