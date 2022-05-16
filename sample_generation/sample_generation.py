"""Module for sample generation helper functions."""
from math import ceil
from typing import List

import numpy as np
import pandas as pd

from .generators.sample_generation_params import SampleGenerationParams
from .generators.sample_generator import SampleGenerator


def prepare_sample_generator(
    sample_generator_factory, n_samples=200, seed=None, **model_params
) -> SampleGenerator:
    """Creates a sample generator for the given parameters.

    Parameters
    ----------
    sample_generator_factory:
        Sample generator factory to use to instantiate sample generator.
    n_samples:
        Number of data points for a single time series sample.
    seed:
        Seed for randomness.
    model_params:
        Model parameters for sample generator.

    Returns
    -------
    sample_generator
        Created sample generator.
    """
    params = SampleGenerationParams(n_samples, model_params=model_params)
    sample_generator = sample_generator_factory(params=params, seed=seed)

    return sample_generator


def generate_samples(
    n: int, generators: List[SampleGenerator], sample_size=365 * 4, seed=None
):
    """Generates a fixed amount of time series samples.

    Parameters
    ----------
    n: int
        Number of samples that we should reach at least.
        Will be more if n is not dividable by the length of generators to
        ensure an equal distribution of generators.
    generators: List[SampleGenerator]
        Sample generators that generate the time series samples.
    sample_size: int
        Number of datapoints for a single time series sample.
    seed:
        Seed for randomness.
    n_jobs: int
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    Returns
    -------
    df: pd.DataFrame
        Pandas DataFrame containing the generated time series and their labels.
    """
    if seed:
        np.random.seed(seed)

    n_samples_per_generator = max(1, ceil(n / len(generators)))
    n = n_samples_per_generator * len(generators)

    df = pd.DataFrame(index=range(n))

    total_y = np.zeros(shape=(n, sample_size))
    total_params = np.zeros(shape=n, dtype=object)
    total_labels = np.zeros(n, dtype=np.int16)
    for gen_idx, generator in enumerate(generators):
        start_idx = int(gen_idx * n_samples_per_generator)
        end_idx = int((gen_idx + 1) * n_samples_per_generator)

        for sim_idx in range(n_samples_per_generator):
            y, params = generator.simulate()
            y = SampleGenerator.to_standard_scale(y)

            total_y[start_idx + sim_idx] = y
            total_params[start_idx + sim_idx] = params

        total_labels[start_idx:end_idx] = np.repeat(
            gen_idx, repeats=n_samples_per_generator
        )

    df["series"] = pd.Series(list(total_y))
    df["labels"] = total_labels
    df["params"] = total_params

    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    return df
