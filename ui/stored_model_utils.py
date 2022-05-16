"""Module for stored model utilities."""

import pickle
from time import time

import numpy as np
import streamlit as st
import torch

from model.dataset_utils import convert_sample_to_model_input
from ui.formatting import format_period_in_s


@st.cache
def load_pred(path_to_model: str, y: np.ndarray) -> (str, str):
    """Loads a prediction for the given time series with the given model path.

    Parameters
    ----------
    path_to_model: str
        Path to model which should be loaded
    y: np.ndarray
        Time series for which to predict the model class

    Returns
    -------
    pred_label: str
        Label of predicted model class
    pred_time: str
        Formatted duration of prediction
    """
    with open(r"data\labels.pickle", "rb") as handle:
        label_lookup = pickle.load(handle)

    stored_model = _load_model(path_to_model)

    datapoint = convert_sample_to_model_input(y, "only")

    prediction_start_time = time()
    orig_pred = stored_model(datapoint)
    _, pred = torch.max(orig_pred.data, 1)
    pred = pred.cpu().numpy().astype(int)[0]
    pred = label_lookup[pred]
    prediction_duration_in_s = time() - prediction_start_time
    prediction_time_fmt = format_period_in_s(prediction_duration_in_s)

    return pred, prediction_time_fmt


def _load_model(path_to_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.load(path_to_model, map_location=device)
