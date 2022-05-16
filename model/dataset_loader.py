"""Module for dataset loading that allows to create infinite training samples."""
import os
from typing import List

from torch.utils.data import DataLoader

from sample_generation.generators.sample_generator import SampleGenerator
from utils import debugger_is_active
from .off_line_dataset import OffLineDataset
from .on_line_dataset import OnLineDataset


def on_the_fly_loader_factory(
    generators: List[SampleGenerator],
    batch_size: int = 32,
    n_test_samples: int = 20_000,
    sample_size: int = 365 * 4,
    n_jobs: int = -1,
) -> (int, bool, DataLoader, DataLoader):
    """Data loader factory that is focused on creating data on the fly.

    Parameters
    ----------
    generators: List[SampleGenerator]
        List of sample generators which should generate time series data.
        The indices of the list are the labels.
    batch_size: int
        Batch size for data loaders. (default: 32)
    n_test_samples: int
        Number of test samples for validation set and test set. (default: 20000)
    sample_size: int
        Size of a single time series sample. (default: 365 * 4
         - assumption: day frequency for 4 years)
    n_jobs: int
        Number of jobs to run in parallel for data loading.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    Returns
    -------
    n_classes: int
        Number of classes to predict.
    is_data_loader_size_equal_length: bool
        False - tells the consumer that the length of the train data loader
        is not equal to the actual length of the data and therefore
        a run through the data loader is not necessarily an epoch.
    train_loader: DataLoader
        Data loader for training data.
        Uses OnLineDataset to create random data on the fly.
    val_loader: DataLoader
        Data loader for validation data.
        Uses OffLineDataset to create random data once and reuse it
        for comparable evaluations.
    """

    if debugger_is_active():
        # override n_jobs for debug mode
        # (otherwise no breakpoints possible in parallel code)
        print("Overriding n_jobs for data loaders to 1 as debugger is active.")
        n_jobs = 1
        n_workers = 0
    else:
        n_workers = n_jobs

    if n_workers == -1:
        # num_workers does not support -1
        n_workers = os.cpu_count()

    print("Initializing data loaders")

    n_classes = len(generators)

    # ensure equal amount of samples for each class in training set
    assert batch_size % n_classes == 0

    training_set = OnLineDataset(
        generators, n=batch_size * n_workers ** 2, fft_mode="only"
    )
    train_loader = DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=False,  # no shuffling necessary, random data anyways
        num_workers=n_workers,
        # required to ensure that rng is not recycled -> iid for train/test split holds!
        persistent_workers=True,
    )

    print("Finished initializing training loader")

    val_loader = init_validation_loader(
        batch_size, generators, n_test_samples, n_workers, sample_size, "only"
    )

    print("Finished initializing validation loader")

    return n_classes, False, train_loader, val_loader


def init_validation_loader(
    batch_size: int,
    generators: List[SampleGenerator],
    n_test_samples: int,
    n_workers: int,
    sample_size: int,
    fft_mode: str,
):
    """Initializes a validation set loader.

    Parameters
    ----------
    batch_size: int
        Batch size for data loaders.
    generators: List[SampleGenerator]
        List of sample generators which should generate time series data.
        The indices of the list are the labels.
    n_test_samples: int
        Number of test samples for validation set and test set.
    n_workers: int
        Number of jobs to run in parallel for data loading.
    sample_size: int
        Size of a single time series sample.
    fft_mode: str
        Configures to the FFT mode of how the FFT of the time series
        should be added to the data points.

    Returns
    -------
    val_loader: DataLoader
        Data loader for validation set.
    """
    if n_workers == -1:
        # num_workers does not support -1
        n_workers = os.cpu_count()

    validation_set = OffLineDataset(
        generators,
        n=n_test_samples,
        sample_size=sample_size,
        n_jobs=n_workers,
        fft_mode=fft_mode,
    )

    val_loader = DataLoader(
        validation_set, batch_size=batch_size, shuffle=True, num_workers=n_workers,
    )

    return val_loader
