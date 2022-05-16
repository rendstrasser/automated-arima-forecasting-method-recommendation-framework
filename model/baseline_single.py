"""Module for the baseline functions that operate on a single input."""
import sys
from typing import List

import numpy as np
from numpy.random import SeedSequence
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

from sample_generation.generators.candidates.baseline_candidate import BaselineCandidate
from .candidate_evaluation import (
    mean_absolute_scaled_error,
    VALID_FIT_EVALUATION_CRITERIA,
)


def spawn_seeds_for_series_fit(
    ss: SeedSequence, n_candidates_total: int, n_splits: int = 5
) -> List[SeedSequence]:
    """Spawns child seed sequences for a single input fitting procedure.

    Parameters
    ----------
    ss: SeedSequence
        Parent seed sequence that is used to spawn the child seed sequences.
    n_candidates_total: int
        Number of candidates over all labels.
    n_splits: int
        Number of CV splits.

    Returns
    -------
    fitting_seeds: List[SeedSequence]
        Child seed sequences for fitting.
    """

    n_fits_per_candidate = 1 + n_splits
    n_fits_per_input = n_fits_per_candidate * n_candidates_total
    return ss.spawn(n_fits_per_input)


def find_best_idx_fit_for_series(
    input: np.ndarray,
    candidates_by_sample_generators: List[List[BaselineCandidate]],
    fitting_seeds: List[SeedSequence],
    evaluation_criteria: List[str] = VALID_FIT_EVALUATION_CRITERIA,
    n_splits: int = 5,
) -> np.ndarray:
    """Evaluates baseline models on a single input.

    Three different baseline models can be evaluated:
    - aic (Akaike information criterion):
        Fits all of the best fits from the sample generators on a time series and
        the one with the best aic wins. AIC combines model fit with model complexity.
    - mape (Mean average percentage error):
        Performs cross-validation on a time series to forecast and calculate the
        mape on the forecast. MAPE focuses on the forecasting performance.
    - mase (Mean average scaled error):
        Performs cross-validation on a time series to forecast and calculate the
        mase on the forecast. MASE focuses on the forecasting performance.

    Parameters
    ----------
    val_loader: DataLoader
        Data loader for validation set.
    candidates_by_sample_generators: List[List[BaselineCandidate]]
        Candidates for best-fit. First dimension gives the label
        and second dimension gives the candidates for the label.
    fitting_seeds: List[SeedSequence]
        Individual seed sequences that should be used for
         random number generation in individual fitting procedures.
    evaluation_criteria: List[str]
        The evaluation criteria to use for predictions.
        Possible are 'aic', 'mape', 'mase'.
    n_splits: int
        Number of CV splits. Only used for evaluation criteria 'mape' and 'mase'.

    Returns
    -------
    results: np.ndarray
        Array of size 3 containing index of best-fitting label for aic, mase and mape.
        -1 is returned for all disabled fits.
    """
    perform_aic = "aic" in evaluation_criteria
    perform_bic = "bic" in evaluation_criteria
    perform_mape = "mape" in evaluation_criteria
    perform_mase = "mase" in evaluation_criteria
    perform_cv = perform_mape or perform_mase

    best_aic = sys.float_info.max
    best_bic = sys.float_info.max
    best_mape = sys.float_info.max
    best_mase = sys.float_info.max

    best_aic_idx = -1
    best_bic_idx = -1
    best_mase_idx = -1
    best_mape_idx = -1

    cur_seed_idx = 0

    for sample_gen_idx, candidates in enumerate(candidates_by_sample_generators):
        for candidate_idx, candidate in enumerate(candidates):
            if perform_aic or perform_bic:
                model = candidate.fit(input, seed=fitting_seeds[cur_seed_idx])
                cur_seed_idx += 1

                aic = candidate.aic(model)
                bic = candidate.bic(model)

                if perform_aic and aic < best_aic:
                    best_aic = aic
                    best_aic_idx = sample_gen_idx

                if perform_bic and bic < best_bic:
                    best_bic = bic
                    best_bic_idx = sample_gen_idx

            if perform_cv:
                split = TimeSeriesSplit(n_splits)

                mase_split_values = np.zeros(shape=n_splits)
                mape_split_values = np.zeros(shape=n_splits)

                for split_idx, (train_indices, test_indices) in enumerate(
                    split.split(input)
                ):
                    y_train = input[train_indices]
                    y_test = input[test_indices]

                    cv_model = candidate.fit(y_train, seed=fitting_seeds[cur_seed_idx])
                    cur_seed_idx += 1

                    y_pred = candidate.forecast(cv_model, len(y_test))

                    if perform_mase:
                        mase = mean_absolute_scaled_error(y_test, y_pred, y_train)
                        mase_split_values[split_idx] = mase

                    if perform_mape:
                        mape = mean_absolute_percentage_error(y_test, y_pred)
                        mape_split_values[split_idx] = mape

                if perform_mase:
                    mase = mase_split_values.mean()
                    if mase < best_mase:
                        best_mase = mase
                        best_mase_idx = sample_gen_idx

                if perform_mape:
                    mape = mape_split_values.mean()
                    if mape < best_mape:
                        best_mape = mape
                        best_mape_idx = sample_gen_idx

    return np.array((best_aic_idx, best_bic_idx, best_mase_idx, best_mape_idx))
