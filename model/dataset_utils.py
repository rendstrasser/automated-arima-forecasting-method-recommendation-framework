"""Utilities for dataset generation."""
import numpy as np
import torch

from sample_generation.generators.sample_generator import SampleGenerator


def generate_dataset_datapoint(generator: SampleGenerator, fft_mode: str):
    """Generates a data point for a generator.

    Parameters
    ----------
    generator: SampleGenerator
        Generates a data point for the model.
    fft_mode: str
        Configures the mode how a real FFT should be calculcated
        on the simulated time series and concatenated to the result.

    Returns
    -------
    y: np.ndarray
        Generated data point.

    """
    y, _ = generator.simulate()

    return _transform_sample(y, fft_mode)


def convert_sample_to_model_input(y: np.ndarray, fft_mode: str) -> torch.Tensor:
    """Converts a sample to model input that can be directly given to a pytorch model.

    Parameters
    ----------
    y: np.ndarray
        Time series to convert.
    fft_mode: str
        Configures the mode how a real FFT should be calculcated
        on the simulated time series and concatenated to the result.

    Returns
    -------
    model_input: torch.Tensor
        Input for model.
    """
    datapoint = _transform_sample(y, fft_mode)[np.newaxis, np.newaxis, ...]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.from_numpy(datapoint).to(device=device)


def _transform_sample(y: np.ndarray, fft_mode: str) -> np.ndarray:
    y = SampleGenerator.to_standard_scale(y)

    if fft_mode == "none":
        return y

    if fft_mode == "only":
        return _calc_fft(y)

    if fft_mode == "include":
        return _concat_fft(y)

    raise ValueError(f"Unknown fft_mode '{fft_mode}'")


def _concat_fft(y):
    amplitudes = _calc_fft(y)

    return np.concatenate((y, amplitudes))


def _calc_fft(x):
    fft = np.fft.rfft(x)

    # transform to amplitudes and take only real values,
    # because the imaginary values mostly represent phase shift,
    # which we don't need for a classification which doesn't rely on phase shifts
    return np.abs(fft ** 2)[: len(x) // 2]
