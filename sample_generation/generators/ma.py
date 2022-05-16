"""Module for MA sample generation."""
from .arma import ARMAGenerator
from .default_settings import DEFAULT_SARIMAX_MIN_q, DEFAULT_SARIMAX_MAX_q
from .sample_generation_params import SampleGenerationParams


class MAGenerator(ARMAGenerator):
    """A sample generator that generates random MA(q) time series samples.

    An MA(q) model is a moving-average model with p moving average terms [1].
    Note: Don't be confused with moving average models and the moving average
    formula, they are not the same! A moving average model is better explained as a
    model that uses past errors (past noise values) to calculate the current value.

    Parameters
    ----------
    params: SampleGenerationParams
        Parameters that configure the sample generator.
    min_order_q: int
        Minimum integer value for hyperparameter q - inclusive. (default: 1)
    max_order_q: int
        Maximum integer value for hyperparameter q - inclusive. (default: 5)
    seed:
        Seed that is used to randomly generate data.
        (default is no seed -> nondeterministic results)

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Moving-average_model
    """

    def __init__(
        self,
        params: SampleGenerationParams,
        min_order_q=DEFAULT_SARIMAX_MIN_q,
        max_order_q=DEFAULT_SARIMAX_MAX_q,
        seed: int = None,
    ):
        super().__init__(params, 0, 0, min_order_q, max_order_q, seed)
