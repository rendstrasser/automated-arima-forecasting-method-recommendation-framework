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


def print_baseline_demo(seed: int, ss: SeedSequence):
    """Prints training demo.

    Parameters
    ----------
    seed: int
        Seed for randomness.
    ss: SeedSequence
        Seed sequence for randomness.
    """
    has_errors = False

    st.header("Baseline evaluation")

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

    selected_evaluation_criteria = st.multiselect(
        "Which evaluation criteria should be used",
        VALID_FIT_EVALUATION_CRITERIA,
        default=DEFAULT_EVALUATION_CRITERIA,
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

    prev_results = dict()

    if isfile("results.pickle"):
        with open("results.pickle", "rb") as handle:
            prev_results = pickle.load(handle)

    if len(prev_results) > 0 and len(prev_results) != len(selected_evaluation_criteria):
        batch_size_error_placeholder.error(
            "Evaluation criteria needs to match between previous result set"
            " and selected evaluation criteria."
        )
        has_errors = True

    for key, value in prev_results.items():
        if len(value.predictions) % batch_size != 0:
            batch_size_error_placeholder.error(
                f"Batch size ({batch_size}) needs to be dividable "
                + f"by the previous result set length ({len(value.predictions)})"
            )
            has_errors = True

        if key not in selected_evaluation_criteria:
            batch_size_error_placeholder.error(
                f"Evaluation criteria needs to match between previous result set"
                f" and selected evaluation criteria (prev contains {key})."
            )
            has_errors = True

    # ]

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

        st.subheader("Baseline model evaluation")
        confusion_matrix_placeholder = st.empty()

        st.subheader("Logging output")

        baseline_results, val_labels = _eval_baseline_models(
            confusion_matrix_placeholder,
            val_loader,
            selected_sample_generators,
            selected_evaluation_criteria,
            ss,
            n_jobs,
            5,
            prev_results,
        )

        with confusion_matrix_placeholder:
            a = st.container()
            with a:
                for method_name, result in baseline_results.items():
                    plot_confusion_matrix(
                        val_labels,
                        result.predictions,
                        selected_sample_generators,
                        result.score,
                        f"baseline: {method_name}",
                    )


def _eval_baseline_models(
    placeholder,
    val_loader,
    sample_generators,
    evaluation_criteria,
    ss,
    n_jobs,
    n_splits,
    prev_results,
):
    n_samples = len(val_loader.dataset)
    all_predictions = defaultdict(lambda: np.zeros(n_samples, dtype=int))
    all_labels = np.zeros(n_samples, dtype=int)

    n_candidates_total = 0
    candidates_by_sample_generators = np.zeros(
        shape=(len(sample_generators)), dtype="O"
    )
    for i, sample_generator in enumerate(sample_generators):
        candidates_by_sample_generators[i] = sample_generator.get_best_fit_candidates()
        n_candidates_total += len(candidates_by_sample_generators[i])

    st.write(f"Executing {len(val_loader)} iterations (batches)")

    prev_executed_preds = 0
    if evaluation_criteria[0] in prev_results:
        prev_executed_preds = len(prev_results[evaluation_criteria[0]].predictions)

    for method_name, result in prev_results.items():
        all_labels[:prev_executed_preds] = result.labels
        all_predictions[method_name][:prev_executed_preds] = result.predictions

    start_info_written = False
    for val_idx, (inputs, labels) in enumerate(val_loader):
        if (val_idx + 1) * val_loader.batch_size <= prev_executed_preds:
            continue

        if not start_info_written:
            st.write(f"Starting from index {val_idx}")

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
        pred = np.asarray(
            Parallel(n_jobs=n_jobs)(
                delayed(find_best_idx_fit_for_series)(
                    y,
                    candidates_by_sample_generators,
                    spawn_seeds_for_series_fit(ss, n_candidates_total, n_splits),
                    evaluation_criteria,
                    n_splits,
                )
                for y in inputs
            )
        ).T
        parallel_duration_in_s = time() - parallel_start_time

        all_predictions["aic"][result_start_idx:result_end_idx] = pred[0]
        all_predictions["bic"][result_start_idx:result_end_idx] = pred[1]
        all_predictions["mase"][result_start_idx:result_end_idx] = pred[2]
        all_predictions["mape"][result_start_idx:result_end_idx] = pred[3]

        results = dict()
        partial_results = dict()
        for method_name, predictions in all_predictions.items():
            if method_name in evaluation_criteria:
                score = accuracy_score(
                    all_labels[:result_end_idx], predictions[:result_end_idx]
                )
                results[method_name] = BaselineResult(score, all_labels, predictions)
                partial_results[method_name] = BaselineResult(
                    score, all_labels[:result_end_idx], predictions[:result_end_idx]
                )

        with open("results.pickle", "wb") as handle:
            pickle.dump(partial_results, handle)

        with placeholder:
            a = st.container()
            with a:
                for method_name, result in results.items():
                    plot_confusion_matrix(
                        all_labels[:result_end_idx],
                        result.predictions[:result_end_idx],
                        sample_generators,
                        result.score,
                        f"baseline (i={val_idx}): {method_name}",
                    )

        for method_name, result in results.items():
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

    results = dict(prev_results)
    for method_name, predictions in all_predictions.items():
        if method_name in evaluation_criteria:
            score = accuracy_score(all_labels, predictions)
            results[method_name] = BaselineResult(score, all_labels, predictions)

    return results, all_labels
