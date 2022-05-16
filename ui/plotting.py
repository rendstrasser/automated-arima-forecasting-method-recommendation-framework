"""Module for plotting utilities."""
import altair as alt
import pandas as pd
import streamlit as st


def plot_confusion_matrix(
    actual, predicted, selected_sample_generators, val_score, context
):
    """Plots a confusion matrix."""

    predicted = prettify_label_array(predicted, selected_sample_generators)
    actual = prettify_label_array(actual, selected_sample_generators)

    df = pd.DataFrame({"predicted": predicted, "actual": actual})

    base_chart = (
        alt.Chart(
            df,
            title=f"Validation score (accuracy) - {context} = {val_score * 100:.4f} %",
        )
        .transform_aggregate(num_elems="count()", groupby=["actual", "predicted"])
        .encode(
            alt.X("predicted:O", scale=alt.Scale(paddingInner=0)),
            alt.Y("actual:O", scale=alt.Scale(paddingInner=0)),
        )
        .properties(width=640, height=480)
    )

    confusion_matrix_chart = base_chart.mark_rect().encode(
        color=alt.Color(
            "num_elems:Q",
            scale=alt.Scale(scheme="viridis"),
            legend=alt.Legend(direction="horizontal"),
        )
    )

    text_chart = base_chart.mark_text(
        baseline="middle", fontSize=16, fontWeight=200
    ).encode(text="num_elems:Q")

    st.altair_chart(confusion_matrix_chart + text_chart, use_container_width=True)


def prettify_label_array(labels, selected_sample_generators):
    """Prettifies an array of generators to pretty generator names."""
    return [
        _prettify_gen_name(type(selected_sample_generators[label])) for label in labels
    ]


def _prettify_gen_name(generator_clazz):
    return generator_clazz.__name__.replace("Generator", "")
