"""Module for common sample generation parameters."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SampleGenerationParams:
    """Parameters for sample generators that are independent from the model class."""

    # How many data points should be simulated
    n_samples: int

    # Params of the model that fits the data points
    # Example: May contain 'order' for ARIMA models.
    model_params: dict[str, Any] = field(default_factory=dict)

    # Number of observation at the beginning of the sample to drop.
    # Used to reduce dependence on initial values.
    burnin: int = 100
