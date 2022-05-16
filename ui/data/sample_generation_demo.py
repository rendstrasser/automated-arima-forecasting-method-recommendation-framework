"""UI functions for sample generation demo."""
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from numpy.random import SeedSequence

from sample_generation.generators.ar import ARGenerator
from sample_generation.generators.arima import ARIMAGenerator
from sample_generation.generators.arma import ARMAGenerator
from sample_generation.generators.gaussian_white_noise import (
    GaussianWhiteNoiseGenerator,
)
from sample_generation.generators.ma import MAGenerator
from sample_generation.generators.random_walk import RandomWalkGenerator
from sample_generation.generators.sample_generation_params import SampleGenerationParams
from sample_generation.sample_generation import generate_samples


def print_sample_generation(seed: int, ss: SeedSequence):
    """Prints a sample generation demo.

    Parameters
    ----------
    seed: int
        Seed for randomness.
    path: str
        Path, where to store the generated samples.
    """

    st.header("Write samples")
    n_samples = st.number_input("Sample count", min_value=10, value=100_000)
    sample_size = st.number_input("Sample size", min_value=10, value=365 * 4)

    params = SampleGenerationParams(sample_size)

    valid_sample_generators = [
        RandomWalkGenerator,
        GaussianWhiteNoiseGenerator,
        ARGenerator,
        MAGenerator,
        ARMAGenerator,
        ARIMAGenerator,
    ]

    sample_generator_options = [gen.__name__ for gen in valid_sample_generators]

    selected_sample_generator_names = st.multiselect(
        "Which sample generators should generate the data points",
        sample_generator_options,
        sample_generator_options,
    )

    selected_sample_generators = [
        gen(params)
        for gen in valid_sample_generators
        if gen.__name__ in selected_sample_generator_names
    ]

    path = ""
    file_name = ""

    should_store_samples = st.checkbox("Store samples?")

    if should_store_samples:
        path = st.text_input("Path")
        file_name = st.text_input("File name", value="data")

    is_generate_samples = st.button("Generate")

    if is_generate_samples:
        df = generate_samples(
            n=n_samples,
            generators=selected_sample_generators,
            seed=seed,
            sample_size=sample_size,
        )

        all_coeffs = []
        for params in df["params"].values:
            if "transformed_params" in params:
                coefficients = params["transformed_params"][:-1]
                all_coeffs.extend(coefficients)

        all_coeffs = np.array(all_coeffs)

        fig, (ax) = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))

        ax.hist(all_coeffs, density=True, bins=50)
        ax.set_title(f"Histogram of {selected_sample_generator_names} coefficients")
        ax.set_ylabel("Frequency")
        ax.set_ylim(0, 1)
        ax.set_yticks([])

        ax.set_xlabel("Coefficient bins")
        ax.xaxis.set_ticks(
            np.linspace(start=all_coeffs.min(), stop=all_coeffs.max(), num=10)
        )

        st.pyplot(fig)

        if should_store_samples:
            full_path = path + "\\" + file_name + ".pl"
            df.to_pickle(full_path)
