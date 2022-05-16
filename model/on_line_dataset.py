"""Module for an on line dataset class."""
from typing import List

import numpy as np
from torch.utils.data import Dataset

from model.dataset_utils import generate_dataset_datapoint
from sample_generation.generators.sample_generator import SampleGenerator


class OnLineDataset(Dataset):
    """Dataset that creates random data on the fly.

    Parameters
    ----------
    generators: List[SampleGenerator]
        List of sample generators which should generate data points.
    n: int
        Number of samples that should be created within this data loader instance.
        Needs to be at least the batch size and at most the size of an epoch.
        It is advised to be higher than the batch size to
        allow multithreading to be used properly.
    fft_mode: str
        Configures the mode of using the real FFT of time series in outputted data.
    """

    def __init__(self, generators: List[SampleGenerator], n: int, fft_mode: str):
        self.generators = generators
        self.n_classes = len(self.generators)
        self.n = n
        self.fft_mode = fft_mode

    def __len__(self):
        """Length of data set (not for an epoch!)."""
        return self.n

    def __getitem__(self, index):
        """Retrieves a new time series sample.

        Parameters
        ----------
        index: int
            Index of time series to retrieve. The index is only used to guarantee an
            equal distribution of sample generators. For each given index,
            a completely new random time series is generated based on
            a deterministically chosen sample generator.

        Returns
        -------
        x: np.ndarray
            Generated time series sample.
        y: int
            Label of the time series (index of the sample generator
            that created the time series)
        """
        generator_idx = index % self.n_classes
        generator = self.generators[generator_idx]

        x = generate_dataset_datapoint(generator, self.fft_mode)

        return x[np.newaxis, ...], generator_idx
