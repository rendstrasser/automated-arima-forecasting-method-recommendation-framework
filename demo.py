"""Entry point for Streamlit demo."""
import importlib

import numpy as np
import streamlit as st
from numpy.random import SeedSequence

from ui.data.sample_generation_demo import print_sample_generation
from ui.models.baseline_demo import print_baseline_demo
from ui.models.dtw_demo import print_dtw_demo
from ui.models.training_demo import print_training_demo
from ui.simulation.simulation_demo import print_simulation_demo

PAGES = {
    "Simulation demo": print_simulation_demo,
    "Writing samples": print_sample_generation,
    "Baseline evaluation": print_baseline_demo,
    "DTW evaluation": print_dtw_demo,
    "Training a model": print_training_demo,
}


def main():
    """Entry point for Streamlit demo."""

    # seeding
    seed = int(st.sidebar.number_input("Seed", value=49))
    np.random.seed(seed)
    ss = SeedSequence(seed)

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()), index=3)

    page_generator = PAGES[selection]

    st.sidebar.title("Page-specific settings")

    # execute page generation
    page_generator(seed, ss)


if __name__ == "__main__":
    main()
