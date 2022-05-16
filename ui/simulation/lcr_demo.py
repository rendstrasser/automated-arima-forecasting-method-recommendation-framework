"""UI methods for LCR demo."""
import numpy as np
import streamlit as st

from feature_transformation.lcr import randomly_permutate, lcr


def print_lcr_demo(y):
    """Prints a demo for the LCR transformation (no longer maintained)."""
    lcr_demo_container = st.expander("LCR demo")
    lcr_demo_container.title("LCR demo")
    n_buckets = int(
        lcr_demo_container.number_input(
            "Number of uniformly distributed buckets", value=200
        )
    )
    lcr_x = randomly_permutate(y)
    lcr_y_1 = lcr(lcr_x, n_buckets=n_buckets)

    lcr_demo_container.text("Randomly permuted simulated y")
    lcr_demo_container.line_chart(lcr_x)
    lcr_demo_container.line_chart(lcr_y_1)

    lcr_y_2 = lcr(y, n_buckets=n_buckets)
    lcr_demo_container.text("Simulated y from above")
    lcr_demo_container.line_chart(y)
    lcr_demo_container.line_chart(lcr_y_2)

    lcr_demo_container.text("Difference of simulated y to random in LCR")
    lcr_demo_container.line_chart(np.abs(lcr_y_2 - lcr_y_1))
