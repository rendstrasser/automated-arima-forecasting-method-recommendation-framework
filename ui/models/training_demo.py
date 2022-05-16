"""UI functions for printing training demo."""
import os
import pickle

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from numpy.random import SeedSequence

from model.dataset_loader import on_the_fly_loader_factory
from model.label_selection import (
    TRAIN_SAMPLE_GENERATOR_FACTORIES,
    TRAIN_SAMPLE_GENERATOR_NAMES, DEFAULT_SAMPLE_GENERATOR_NAMES,
)
from model.training import (
    prepare_training,
    train_epoch,
    evaluate_accuracy_on_validation_set,
)
from sample_generation.generators.sample_generation_params import SampleGenerationParams
from ui.plotting import plot_confusion_matrix, prettify_label_array

SAMPLE_SIZE = 365 * 4

# actual default values
DEFAULT_N_TEST_SAMPLES = 10_000
DEFAULT_BATCH_SIZE = 70
DEFAULT_SELECTED_SAMPLE_GENERATOR_NAMES = DEFAULT_SAMPLE_GENERATOR_NAMES
DEFAULT_LR = 0.0001
DEFAULT_WEIGHT_DECAY = 0.00001
DEFAULT_N_EPOCHS = 1000
DEFAULT_N_UPDATES_PER_EPOCH = 60_000
DEFAULT_N_JOBS = 6
DEFAULT_IS_STORE_BEST_MODEL_ENABLED = False
DEFAULT_MODEL_STORE_LOCATION = r"data\new_model.pt"

# debugging default values (for quick epochs)
# DEFAULT_N_JOBS = 1
# DEFAULT_BATCH_SIZE = 20
# DEFAULT_SELECTED_SAMPLE_GENERATOR_NAMES = [ARGenerator.__name__, MAGenerator.__name__]
# DEFAULT_N_EPOCHS = 5
# DEFAULT_N_TEST_SAMPLES = 100
# DEFAULT_N_UPDATES_PER_EPOCH = 100
# DEFAULT_IS_STORE_BEST_MODEL_ENABLED = False


def print_training_demo(seed: int, ss: SeedSequence):
    """Prints training demo.

    Parameters
    ----------
    seed: int
        Seed for randomness.
    """
    has_errors = False

    st.header("Training a model")
    st.write(
        "Simulate time series data using the selected sample generators "
        + "and train a convolutional neural network "
        + "with the given settings to classify samples against their generator."
    )

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

    # Training settings
    # [
    st.subheader("Training settings")
    col1, col2, col3, col4 = st.columns([3, 3, 4, 5])
    lr = float(
        col1.text_input("Learning rate", value=np.format_float_positional(DEFAULT_LR))
    )
    weight_decay = float(
        col2.text_input(
            "Weight decay", value=np.format_float_positional(DEFAULT_WEIGHT_DECAY)
        )
    )
    n_epochs = col3.number_input("Number of epochs", value=DEFAULT_N_EPOCHS)
    n_updates_per_epoch = col4.number_input(
        "Number of updates per epoch", value=DEFAULT_N_UPDATES_PER_EPOCH
    )
    # ]

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
    col2.write("")
    col2.write("")
    is_store_best_model_enabled = col2.checkbox(
        "Store best model?", value=DEFAULT_IS_STORE_BEST_MODEL_ENABLED
    )

    results_path = None
    if is_store_best_model_enabled:
        results_path = st.text_input(
            "Store best model at", value=DEFAULT_MODEL_STORE_LOCATION
        )
    # ]

    is_start_training = False
    if not has_errors:
        is_start_training = st.button("Start training")

    if is_start_training:
        st.markdown("---")
        st.title("Output:")

        model_structure_placeholder = st.empty()
        last_epoch_placeholder = st.empty()
        all_epochs_placeholder = st.empty()

        def dataset_loaders_factory():
            return on_the_fly_loader_factory(
                selected_sample_generators,
                batch_size=batch_size,
                n_test_samples=n_test_samples,
                sample_size=SAMPLE_SIZE,
                n_jobs=n_jobs,
            )

        (
            device,
            loss,
            model,
            n_iter_per_epoch,
            n_train_data,
            optimizer,
            n_classes,
            train_loader,
            val_loader,
        ) = prepare_training(
            dataset_loaders_factory, lr, n_updates_per_epoch, seed, weight_decay
        )

        label_lookup = prettify_label_array(
            range(n_classes), selected_sample_generators
        )
        with open(r"data\labels.pickle", "wb") as handle:
            pickle.dump(label_lookup, handle)

        with model_structure_placeholder:
            n_model_params = sum(p.numel() for p in model.parameters())
            n_model_params = "{:,}".format(n_model_params).replace(",", " ")

            a = st.container()
            a.subheader("Model structure of convolutional neural network")
            a.write(f"The model has {n_model_params} trainable parameters.")
            a.code(str(model).replace("\n", "  \n"))

        st.subheader("Logging output")

        best_score_so_far = -1
        train_losses = np.zeros(n_epochs)
        val_scores = np.zeros(n_epochs)
        ppv_values = np.zeros((n_epochs, n_classes))
        tpr_values = np.zeros((n_epochs, n_classes))
        for epoch_idx in range(n_epochs):
            losses = np.zeros(n_train_data * n_iter_per_epoch)

            train_epoch(
                losses,
                n_iter_per_epoch,
                train_loader,
                device,
                optimizer,
                model,
                loss,
                n_train_data,
            )

            val_score, predicted, actual = evaluate_accuracy_on_validation_set(
                model, device, val_loader
            )
            train_losses[epoch_idx] = losses.mean()
            val_scores[epoch_idx] = val_score

            # assumed to be calculated s.t. this is an integer anyways
            n_test_samples_per_class = len(actual) // n_classes
            n_total_predicted_by_class = np.zeros(n_classes)
            tp_values = np.zeros(n_classes)
            for i, label in enumerate(actual):
                predicted_label = predicted[i]
                n_total_predicted_by_class[predicted_label] += 1
                if predicted_label == label:
                    tp_values[label] += 1

            tpr_values[epoch_idx] = tp_values / n_test_samples_per_class
            ppv_values[epoch_idx] = np.divide(
                tp_values,
                n_total_predicted_by_class,
                out=np.zeros_like(tp_values),
                where=n_total_predicted_by_class != 0,
            )

            if val_score > best_score_so_far:
                best_score_so_far = val_score
                if is_store_best_model_enabled:
                    torch.save(model, results_path)

            with last_epoch_placeholder:
                a = st.container()
                a.subheader(f"Statistics for last epoch ({epoch_idx + 1})")
                with a:
                    plot_confusion_matrix(
                        actual,
                        predicted,
                        selected_sample_generators,
                        val_score,
                        f"epoch {epoch_idx + 1}",
                    )

            with all_epochs_placeholder:
                a = st.container()
                a.subheader("Statistics for all executed epochs")
                with a:
                    _plot_score_charts(epoch_idx, train_losses, val_scores)
                    _plot_tpv_and_ppv_per_label(
                        epoch_idx, tpr_values, ppv_values, selected_sample_generators
                    )

            st.write(
                f"Epoch {epoch_idx + 1}: training_loss={losses.mean():.9f}"
                + f", accuracy_on_validation={val_score:.9f}"
            )


def _plot_tpv_and_ppv_per_label(
    epoch_idx, tpr_values, ppv_values, selected_sample_generators
):
    if epoch_idx == 0:
        return

    n_classes = tpr_values.shape[1]

    tpr_df = pd.DataFrame({"epochs": range(1, epoch_idx + 2)})

    ppv_df = pd.DataFrame({"epochs": range(1, epoch_idx + 2)})

    label_lookup = prettify_label_array(range(n_classes), selected_sample_generators)
    for label in range(n_classes):
        tpr_df[label_lookup[label]] = tpr_values[: epoch_idx + 1, label] * 100
        ppv_df[label_lookup[label]] = ppv_values[: epoch_idx + 1, label] * 100

    tpr_df = tpr_df.reset_index(drop=True).melt("epochs")
    ppv_df = ppv_df.reset_index(drop=True).melt("epochs")

    selection = alt.selection_multi(fields=["variable"], bind="legend")

    tpr_line_chart = (
        alt.Chart(tpr_df, title="True positive rate")
        .mark_line()
        .encode(
            x="epochs:Q",
            y="value",
            color="variable",
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        )
        .add_selection(selection)
    )

    ppv_line_chart = (
        alt.Chart(ppv_df, title="Positive predictive value")
        .mark_line()
        .encode(
            x="epochs:Q",
            y="value",
            color="variable",
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        )
        .add_selection(selection)
    )

    tpr_chart = _plot_line_chart_with_tooltip(
        tpr_line_chart, tpr_df, "epochs", "value", "left", " %"
    )
    ppv_chart = _plot_line_chart_with_tooltip(
        ppv_line_chart, ppv_df, "epochs", "value", "right", " %"
    )

    st.altair_chart(tpr_chart | ppv_chart, use_container_width=True)


def _plot_score_charts(epoch_idx, train_losses, val_scores):
    if epoch_idx == 0:
        return

    df = pd.DataFrame(
        {
            "epochs": range(1, epoch_idx + 2),
            "train_loss": train_losses[: epoch_idx + 1],
            "train_loss_log": np.log1p(train_losses[: epoch_idx + 1]),
            "val_score": val_scores[: epoch_idx + 1] * 100,
        }
    )

    train_line_chart = (
        alt.Chart(
            df,
            title="Log-transformed training loss averaged "
            + "over batches (CrossEntropyLoss)",
        )
        .mark_line()
        .encode(
            alt.X("epochs:Q"), alt.Y("train_loss_log:Q", scale=alt.Scale(type="log"))
        )
    )

    val_line_chart = (
        alt.Chart(df, title="Validation score (accuracy)")
        .mark_line()
        .encode(x="epochs:Q", y="val_score:Q")
    )

    train_chart = _plot_line_chart_with_tooltip(
        train_line_chart, df, "epochs", "train_loss_log", "left", " loss"
    )
    val_chart = _plot_line_chart_with_tooltip(
        val_line_chart, df, "epochs", "val_score", "right", " %"
    )

    st.altair_chart(train_chart | val_chart, use_container_width=True)


def _plot_line_chart_with_tooltip(
    line_chart, df, x_field, y_field, text_align, additional_fmt=""
):
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest_val = alt.selection(
        type="single", nearest=True, on="mouseover", fields=[x_field], empty="none"
    )

    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors_chart = (
        alt.Chart(df)
        .mark_point()
        .encode(x=x_field + ":Q", opacity=alt.value(0),)
        .add_selection(nearest_val)
    )

    # Draw points on the line, and highlight based on selection
    points_chart = (
        alt.Chart(df)
        .mark_point()
        .encode(
            x=x_field + ":Q",
            y=y_field + ":Q",
            opacity=alt.condition(nearest_val, alt.value(1), alt.value(0)),
        )
    )

    # Draw text labels near the points, and highlight based on selection
    text_chart = (
        points_chart.transform_calculate(
            fmt_y=f'format(datum.{y_field},".4f") + "{additional_fmt}"'
        )
        .mark_text(align=text_align, dx=5, dy=-5)
        .encode(text=alt.condition(nearest_val, "fmt_y:N", alt.value(" ")))
    )

    # Draw a rule at the location of the selection
    rules_chart = (
        alt.Chart(df)
        .mark_rule(color="gray")
        .encode(x=x_field + ":Q",)
        .transform_filter(nearest_val)
    )

    return alt.layer(line_chart, selectors_chart, points_chart, rules_chart, text_chart)
