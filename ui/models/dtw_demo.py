"""UI functions for printing training demo."""
import os
import pickle
from collections import defaultdict
from os.path import isfile
from time import time

import numpy as np
import streamlit as st
from joblib import Parallel, delayed
from numpy.random import SeedSequence
from sklearn.metrics import accuracy_score

from model.baseline_result import BaselineResult
from model.baseline_single import (
    find_best_idx_fit_for_series,
    spawn_seeds_for_series_fit,
)
from model.candidate_evaluation import VALID_FIT_EVALUATION_CRITERIA
from model.dataset_loader import init_validation_loader
from model.dtw_model import DtwKnnModel
from model.label_selection import (
    TRAIN_SAMPLE_GENERATOR_FACTORIES,
    TRAIN_SAMPLE_GENERATOR_NAMES, DEFAULT_SAMPLE_GENERATOR_NAMES,
)
from sample_generation.generators.sample_generation_params import SampleGenerationParams
from ui.formatting import format_period_in_s
from ui.plotting import plot_confusion_matrix

SAMPLE_SIZE = 365 * 4

# actual default values
DEFAULT_N_TEST_SAMPLES = 100
DEFAULT_BATCH_SIZE = 70
DEFAULT_SELECTED_SAMPLE_GENERATOR_NAMES = DEFAULT_SAMPLE_GENERATOR_NAMES
DEFAULT_N_JOBS = 6
DEFAULT_EVALUATION_CRITERIA = ["bic"]


def print_dtw_demo(seed: int, ss: SeedSequence):
    """Prints training demo.

    Parameters
    ----------
    seed: int
        Seed for randomness.
    ss: SeedSequence
        Seed sequence for randomness.
    """
    has_errors = False

    st.header("DTW evaluation")

    # Dataset settings
    # [
    st.subheader("Dataset settings")

    st.write(
        f"Sample size = {SAMPLE_SIZE} "
        + "(hardcoded, because model needs to be manually adapted if changed)"
    )

    col1, col2 = st.columns([1, 1])
    n_test_samples = col1.number_input(
        "Sample count for validation/test set",
        min_value=10,
        value=DEFAULT_N_TEST_SAMPLES,
    )
    batch_size = col2.number_input("Batch size", min_value=1, value=DEFAULT_BATCH_SIZE)

    batch_size_error_placeholder = st.empty()

    selected_sample_generator_names = st.multiselect(
        "Which sample generators should be classified",
        TRAIN_SAMPLE_GENERATOR_NAMES,
        default=DEFAULT_SELECTED_SAMPLE_GENERATOR_NAMES,
    )

    params = SampleGenerationParams(SAMPLE_SIZE)
    selected_sample_generators = [
        gen(params, seed=seed + i + 1)
        for i, gen in enumerate(TRAIN_SAMPLE_GENERATOR_FACTORIES)
        if gen.__name__ in selected_sample_generator_names
    ]
    # ]

    if batch_size % len(selected_sample_generators) != 0:
        batch_size_error_placeholder.error(
            f"Batch size ({batch_size}) needs to be dividable "
            + f"by the label count ({len(selected_sample_generators)})"
        )
        has_errors = True

    k = col1.number_input(
        "Use k nearest neighbors",
        min_value=1,
        max_value=100,
        value=5
    )
    n_knn_per_gen = col2.number_input(
        "KNN fit count per label",
        min_value=5,
        value=100
    )

    # Miscellaneous
    # [
    st.subheader("Miscellaneous")
    col1, col2 = st.columns([1, 1])
    n_jobs = col1.number_input(
        "Number of processes (-1 means max)",
        min_value=-1,
        max_value=os.cpu_count(),
        value=DEFAULT_N_JOBS,
    )

    is_start_training = False
    if not has_errors:
        is_start_training = st.button("Start training")

    if is_start_training:
        val_loader = init_validation_loader(
            batch_size,
            selected_sample_generators,
            n_test_samples,
            1,
            SAMPLE_SIZE,
            "none",
        )

        st.subheader("DTW model evaluation")
        confusion_matrix_placeholder = st.empty()

        st.subheader("Logging output")

        result, val_labels = _eval_baseline_models(
            confusion_matrix_placeholder,
            val_loader,
            selected_sample_generators,
            n_jobs,
            k,
            n_knn_per_gen
        )

        with confusion_matrix_placeholder:
            a = st.container()
            with a:
                plot_confusion_matrix(
                    val_labels,
                    result.predictions,
                    selected_sample_generators,
                    result.score,
                    f"DTW",
                )


def _eval_baseline_models(
        placeholder,
        val_loader,
        sample_generators,
        n_jobs,
        k,
        n_knn_per_gen
):
    n_samples = len(val_loader.dataset)
    all_predictions = np.zeros(n_samples, dtype=int)
    all_labels = np.zeros(n_samples, dtype=int)

    n_candidates_total = 0
    candidates_by_sample_generators = np.zeros(
        shape=(len(sample_generators)), dtype="O"
    )
    for i, sample_generator in enumerate(sample_generators):
        candidates_by_sample_generators[i] = sample_generator.get_best_fit_candidates()
        n_candidates_total += len(candidates_by_sample_generators[i])

    st.write(f"Executing {len(val_loader)} iterations (batches)")

    model = DtwKnnModel(generators=sample_generators,
                        n_samples_per_generator=n_knn_per_gen,
                        sample_size=SAMPLE_SIZE,
                        n_jobs=n_jobs,
                        n_neighbors=k)

    for val_idx, (inputs, labels) in enumerate(val_loader):
        batch_start_time = time()

        # for baseline we dont need tensors, but numpy arrays
        inputs = inputs.cpu().numpy().astype(float)
        labels = labels.cpu().numpy().astype(int)

        # the extra dimension is not necessary for baseline
        # - it is only required by pytorch Conv1D layers
        inputs = np.squeeze(inputs, axis=1)

        result_start_idx = val_idx * val_loader.batch_size
        result_end_idx = result_start_idx + len(labels)

        all_labels[result_start_idx:result_end_idx] = labels

        parallel_start_time = time()

        all_predictions[result_start_idx:result_end_idx] = model.predict(inputs)

        parallel_duration_in_s = time() - parallel_start_time

        score = accuracy_score(
            all_labels[:result_end_idx], all_predictions[:result_end_idx]
        )
        result = BaselineResult(score, all_labels, all_predictions)

        with placeholder:
            a = st.container()
            with a:
                plot_confusion_matrix(
                    all_labels[:result_end_idx],
                    result.predictions[:result_end_idx],
                    sample_generators,
                    result.score,
                    f"dtw (i={val_idx})",
                )

        st.write(
            f"Batch {val_idx + 1} has overall score {result.score * 100:.2f} %."
        )

        batch_duration_in_s = time() - batch_start_time

        batch_duration_fmt = format_period_in_s(batch_duration_in_s)
        parallel_duration_fmt = format_period_in_s(parallel_duration_in_s)
        st.write(
            f"Batch {val_idx + 1} took {batch_duration_fmt} "
            f"for which parallel computation took {parallel_duration_fmt}."
        )

    return result, all_labels
