"""Module for the label selection."""

from sample_generation.generators.ar import ARGenerator
from sample_generation.generators.arima import ARIMAGenerator
from sample_generation.generators.arma import ARMAGenerator
from sample_generation.generators.custom_ar.arma_high import ARMAHighGenerator
from sample_generation.generators.custom_ar.arma_low import ARMALowGenerator
from sample_generation.generators.custom_ar.arma_non_significant import ARMANonSignificantGenerator
from sample_generation.generators.custom_ar.arma_significant import ARMASignificantGenerator
from sample_generation.generators.gaussian_white_noise import (
    GaussianWhiteNoiseGenerator,
)
from sample_generation.generators.linear import LinearGenerator
from sample_generation.generators.ma import MAGenerator
from sample_generation.generators.random_walk import RandomWalkGenerator
from sample_generation.generators.sarimax import SARIMAXGenerator

DEFAULT_SAMPLE_GENERATOR_FACTORIES = (
    RandomWalkGenerator,
    GaussianWhiteNoiseGenerator,
    ARGenerator,
    MAGenerator,
    ARMAGenerator,
    ARIMAGenerator,
    SARIMAXGenerator
)

TRAIN_SAMPLE_GENERATOR_FACTORIES = (
    RandomWalkGenerator,
    GaussianWhiteNoiseGenerator,
    ARGenerator,
    MAGenerator,
    ARMAGenerator,
    ARIMAGenerator,
    SARIMAXGenerator,
    LinearGenerator,
    ARMASignificantGenerator,
    ARMANonSignificantGenerator,
    ARMAHighGenerator,
    ARMALowGenerator
)

DEFAULT_SAMPLE_GENERATOR_NAMES = [
    gen.__name__ for gen in DEFAULT_SAMPLE_GENERATOR_FACTORIES
]

TRAIN_SAMPLE_GENERATOR_NAMES = [
    gen.__name__ for gen in TRAIN_SAMPLE_GENERATOR_FACTORIES
]
