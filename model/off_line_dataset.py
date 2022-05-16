"""Module for an off line dataset class."""
from math import ceil
from typing import List

import numpy as np
from torch.utils.data import Dataset

from model.dataset_utils import generate_dataset_datapoint
from sample_generation.generators.sample_generator import SampleGenerator


class OffLineDataset(Dataset):
    """Dataset that creates a predefined amount of random time series and keeps them.

    Parameters
    ----------
    generators: List[SampleGenerator]
        List of sample generators which should generate data points.
    n: int
        Number of samples that we should reach at least.
        Will be more if n is not dividable by the length of generators to
        ensure an equal distribution of generators.
    sample_size:
        Size of a single time series sample.
    n_jobs: int
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    fft_mode: str
        Configures the mode of using the real FFT of time series in outputted data.

    """

    def __init__(
        self,
        generators: List[SampleGenerator],
        n: int,
        sample_size: int,
        n_jobs: int = 1,
        fft_mode: str = "only",
    ):
        self.generators = generators

        n_samples_per_generator = max(1, ceil(n / len(generators)))
        n = n_samples_per_generator * len(generators)

        adjusted_sample_size = self._calc_adjusted_sample_size(fft_mode, sample_size)
        self.X = np.zeros(shape=(n, adjusted_sample_size))
        self.y = np.zeros(n, dtype=np.int16)

        for gen_idx, generator in enumerate(generators):
            start_idx = int(gen_idx * n_samples_per_generator)
            end_idx = int((gen_idx + 1) * n_samples_per_generator)

            # not parallelized atm because of rng issues
            self.X[start_idx:end_idx] = np.asarray(
                [
                    generate_dataset_datapoint(generator, fft_mode)
                    for _ in range(n_samples_per_generator)
                ]
            )
            self.y[start_idx:end_idx] = np.repeat(
                gen_idx, repeats=n_samples_per_generator
            )

        # We need to add a dimension in the middle as Conv1d layers expect
        # this dimension, to be syntactically equal with Conv2d layers that
        # need it.
        self.X = np.expand_dims(self.X, axis=1)

    def __len__(self):
        """Length of data set."""
        return len(self.X)

    def __getitem__(self, index: int):
        """Retrieves time series at the given index.

        Parameters
        ----------
        index: int
            Index of time series to retrieve.

        Returns
        -------
        x: np.ndarray
            Time series at the given index
        y: int
            Label of the time series at given index (index of the sample generator
            that created the time series)
        """
        return self.X[index], self.y[index]

    @staticmethod
    def _calc_adjusted_sample_size(fft_mode, sample_size):
        if fft_mode == "none":
            return sample_size

        if fft_mode == "include":
            return sample_size + (sample_size // 2)

        if fft_mode == "only":
            return sample_size // 2

        raise ValueError(f"Unknown fft_mode '{fft_mode}'")
