"""UI methods for time series simulation demo."""
import pickle
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from numpy.random import SeedSequence
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from model.baseline_single import (
    find_best_idx_fit_for_series,
    spawn_seeds_for_series_fit,
)
from model.candidate_evaluation import VALID_FIT_EVALUATION_CRITERIA
from model.dataset_utils import convert_sample_to_model_input
from model.label_selection import (
    TRAIN_SAMPLE_GENERATOR_NAMES,
    TRAIN_SAMPLE_GENERATOR_FACTORIES, DEFAULT_SAMPLE_GENERATOR_NAMES,
)
from sample_generation.generators.ar import ARGenerator
from sample_generation.generators.arima import ARIMAGenerator
from sample_generation.generators.arma import ARMAGenerator
from sample_generation.generators.custom_ar.arma_high import ARMAHighGenerator
from sample_generation.generators.custom_ar.arma_low import ARMALowGenerator
from sample_generation.generators.custom_ar.arma_non_significant import (
    ARMANonSignificantGenerator,
)
from sample_generation.generators.custom_ar.arma_significant import (
    ARMASignificantGenerator,
)
from sample_generation.generators.custom_ar.arv2 import (
    SpecialARCustomStationaryImplGenerator,
)
from sample_generation.generators.custom_ar.arv3 import (
    SpecialARGenerateARMASamplesGenerator,
)
from sample_generation.generators.custom_ar.arv4 import SpecialCustomAr1ImplGenerator
from sample_generation.generators.fixed_order_sarimax import FixedOrderSARIMAXGenerator
from sample_generation.generators.gaussian_white_noise import (
    GaussianWhiteNoiseGenerator,
)
from sample_generation.generators.linear import LinearGenerator
from sample_generation.generators.ma import MAGenerator
from sample_generation.generators.random_walk import RandomWalkGenerator
from sample_generation.generators.sample_generation_params import SampleGenerationParams
from sample_generation.generators.sample_generator import SampleGenerator
from sample_generation.generators.sarimax import SARIMAXGenerator
from sample_generation.sample_generation import prepare_sample_generator
from ui.formatting import format_period_in_s
from ui.simulation.lcr_demo import print_lcr_demo

ALL_SAMPLE_GENERATOR_FACTORIES = [
    RandomWalkGenerator,
    GaussianWhiteNoiseGenerator,
    FixedOrderSARIMAXGenerator,
    SARIMAXGenerator,
    ARGenerator,
    ARMAHighGenerator,
    ARMALowGenerator,
    ARMASignificantGenerator,
    ARMANonSignificantGenerator,
    SpecialARCustomStationaryImplGenerator,
    SpecialARGenerateARMASamplesGenerator,
    SpecialCustomAr1ImplGenerator,
    MAGenerator,
    ARMAGenerator,
    ARIMAGenerator,
    LinearGenerator,
]

DEFAULT_SHOW_LCR_DEMO = False
DEFAULT_SELECTED_SAMPLE_GENERATOR_NAMES = [
    name
    for name in DEFAULT_SAMPLE_GENERATOR_NAMES
    if not name == LinearGenerator.__name__
]
DEFAULT_SAMPLE_GENERATOR_FACTORY = ARGenerator
DEFAULT_N_SAMPLES = 500
DEFAULT_N_SAMPLES = 1460


def print_simulation_demo(seed: int, ss: SeedSequence):
    """Prints the time series simulation demo.

    Parameters
    ----------
    seed: int
        Seed for randomness.
    """
    show_lcr_demo = st.sidebar.checkbox("Show LCR demo", value=DEFAULT_SHOW_LCR_DEMO)

    fitting_demo_container = st.container()
    fitting_demo_container.title("Time-series simulation demo")
    fitting_demo_container.text("Simulate a time series for a given distribution")
    col1, col2 = fitting_demo_container.columns([3, 1])

    sample_generator_factory_names = [
        gen.__name__ for gen in ALL_SAMPLE_GENERATOR_FACTORIES
    ]
    default_factory_idx = sample_generator_factory_names.index(
        DEFAULT_SAMPLE_GENERATOR_FACTORY.__name__
    )
    sample_generator_factory_name = col1.selectbox(
        "Distribution (/model class) generator",
        sample_generator_factory_names,
        index=default_factory_idx,
    )

    n_samples = col2.number_input("Number of samples", value=DEFAULT_N_SAMPLES)

    with fitting_demo_container:
        params = _print_model_class_specific_settings(sample_generator_factory_name)

    simulation_start_time = time()
    y, debugging_params, sample_gen = _simulate(
        n_samples, params, sample_generator_factory_name, seed
    )
    simulation_time_in_s = time() - simulation_start_time

    fitting_start = time()
    best_fit_gen, best_fit_params = _print_and_find_best_fit_generator(
        sample_gen, seed, ss, y, debugging_params
    )
    model = None
    if best_fit_gen is not None:
        model = best_fit_gen.fit(y, ss, fitting_params=best_fit_params)
    fitting_time_in_s = time() - fitting_start

    pred_placeholder = st.empty()
    _print_simulation_line_chart(y, best_fit_gen, model)

    model_loading_container = st.container()
    col1, col2 = model_loading_container.columns([1, 4])
    col1.text("")  # pad vertically
    col1.text("")  # pad vertically
    should_load_model = col1.checkbox("Load model?")

    if should_load_model:
        path_to_model = col2.text_input(
            "Path to model", value=r"data\best_model.pt",  # noqa
        )

        with open(r"data\labels.pickle", "rb") as handle:
            label_lookup = pickle.load(handle)

        if len(path_to_model) > 0:
            stored_model = _load_model(path_to_model)

            datapoint = convert_sample_to_model_input(y, "only")

            prediction_start_time = time()
            orig_pred = stored_model(datapoint)
            _, pred = torch.max(orig_pred.data, 1)
            pred = pred.cpu().numpy().astype(int)[0]
            pred = label_lookup[pred]
            prediction_duration_in_s = time() - prediction_start_time
            prediction_time_fmt = format_period_in_s(prediction_duration_in_s)

            a = pred_placeholder.container()
            with a:
                st.write(f"Prediction: {pred} (took {prediction_time_fmt})")

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    plot_acf(y, ax=ax1, lags=10)
    plot_pacf(y, ax=ax2, lags=10, method="ywm")
    st.pyplot(fig)

    st.subheader("Debugging params of simulation")
    simulation_time_fmt = format_period_in_s(simulation_time_in_s)
    st.write(f"The simulation took {simulation_time_fmt}.")
    st.write(debugging_params)

    st.subheader("Summary of best-fitting model")
    fitting_time_fmt = format_period_in_s(fitting_time_in_s)
    st.write(f"The fitting took {fitting_time_fmt}.")
    if model is not None and hasattr(model, "summary"):
        st.write(model.summary())

    if show_lcr_demo:
        print_lcr_demo(y)


@st.cache
def _load_model(path_to_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.load(path_to_model, map_location=device)


def _print_and_find_best_fit_generator(sample_gen, seed, ss, y, debugging_params):
    best_fit_search_space_selection = st.radio(
        "Best fit search space",
        options=[
            "same hyperparameters as simulation",
            "same model class",
            "any model class",
            "no fit",
        ],
    )

    if best_fit_search_space_selection == "no fit":
        return None, None

    if best_fit_search_space_selection == "same hyperparameters as simulation":
        return sample_gen, debugging_params

    if best_fit_search_space_selection == "same model class":
        return sample_gen, None

    expander = st.expander("Search space settings")
    with expander:
        selected_sample_generator_names = st.multiselect(
            "Which sample generators should be tested against the time series",
            [
                name
                for name in TRAIN_SAMPLE_GENERATOR_NAMES
                if not name == LinearGenerator.__name__
            ],
            default=DEFAULT_SELECTED_SAMPLE_GENERATOR_NAMES,
        )

        fitting_params = SampleGenerationParams(len(y), model_params={})
        selected_sample_generators = [
            gen(fitting_params, seed=seed + 1)
            for i, gen in enumerate(TRAIN_SAMPLE_GENERATOR_FACTORIES)
            if gen.__name__ in selected_sample_generator_names
        ]

        fit_evaluation_criteria = st.selectbox(
            "Which evaluation criteria should be used to find the best fit",
            VALID_FIT_EVALUATION_CRITERIA,
            index=3,
        )

        candidates_by_sample_generators = np.zeros(
            shape=(len(selected_sample_generators)), dtype="O"
        )
        n_candidates_total = 0
        for i, sample_generator in enumerate(selected_sample_generators):
            candidates_by_sample_generators[
                i
            ] = sample_generator.get_best_fit_candidates()
            n_candidates_total += len(candidates_by_sample_generators[i])

        fitting_seeds = spawn_seeds_for_series_fit(ss, n_candidates_total)

        best_indices = find_best_idx_fit_for_series(
            y, candidates_by_sample_generators, fitting_seeds, [fit_evaluation_criteria]
        )

        best_idx = best_indices[best_indices > -1][0]

        return selected_sample_generators[best_idx], None


def _print_model_class_specific_settings(sample_generator_factory_name):
    if sample_generator_factory_name == FixedOrderSARIMAXGenerator.__name__:
        return _print_fixed_order_sarimax_settings()

    return {}


def _print_fixed_order_sarimax_settings():
    params = {}

    sarimax_expander = st.expander("SARIMAX-specific settings")

    sarimax_expander.write("Order")
    col1, col2, col3 = sarimax_expander.columns(3)
    p = int(col1.number_input("p", value=3, min_value=0))
    d = int(col2.number_input("d", value=1, min_value=0))
    q = int(col3.number_input("q", value=3, min_value=0))

    sarimax_expander.write("Seasonal order (activated if m>1)")
    col1, col2, col3, col4 = sarimax_expander.columns(4)
    P = int(col1.number_input("P", value=0, min_value=0))
    D = int(col2.number_input("D", value=0, min_value=0))
    Q = int(col3.number_input("Q", value=0, min_value=0))
    m = int(col4.number_input("m", value=1, min_value=1))

    params["order"] = (p, d, q)
    if m > 1:
        params["seasonal_order"] = (P, D, Q, m)

    return params


@st.cache
def _simulate(n_samples, debugging_params, sample_generator_factory_name, seed):
    sample_generator_factory = None
    for clz in ALL_SAMPLE_GENERATOR_FACTORIES:
        if clz.__name__ == sample_generator_factory_name:
            sample_generator_factory = clz
            break

    if not sample_generator_factory:
        st.error(
            "Sample generator factory could not be found "
            + f"for {sample_generator_factory_name}."
        )
        return

    sample_gen = prepare_sample_generator(
        sample_generator_factory, n_samples=n_samples, seed=seed, **debugging_params
    )

    y, debugging_params = sample_gen.simulate()

    return y, debugging_params, sample_gen


def _print_simulation_line_chart(y: np.ndarray, best_fit_gen: SampleGenerator, model):
    y_scaled = SampleGenerator.to_standard_scale(y)

    if best_fit_gen is None:
        df = pd.DataFrame({"1. sample": y_scaled})
        st.line_chart(df)
        return

    fitted_values_scaled = best_fit_gen.to_standard_scale(model.fittedvalues)
    df = pd.DataFrame({"2. fitted": fitted_values_scaled, "1. sample": y_scaled})
    st.line_chart(df)
