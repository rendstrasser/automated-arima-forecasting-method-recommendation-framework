from typing import List

import numpy as np
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

from sample_generation.generators.sample_generator import SampleGenerator


class DtwKnnModel:

    def __init__(
            self,
            generators: List[SampleGenerator],
            n_samples_per_generator: int,
            sample_size: int,
            n_jobs: int = 1,
            n_neighbors: int = 5
    ):
        n = n_samples_per_generator * len(generators)

        X = np.zeros(shape=(n, sample_size))
        y = np.zeros(n, dtype=np.int16)

        for gen_idx, generator in enumerate(generators):
            start_idx = int(gen_idx * n_samples_per_generator)
            end_idx = int((gen_idx + 1) * n_samples_per_generator)

            # not parallelized atm because of rng issues
            X[start_idx:end_idx] = np.asarray(
                [
                    np.array(generator.simulate_normalized())
                    for _ in range(n_samples_per_generator)
                ]
            )
            y[start_idx:end_idx] = np.repeat(
                gen_idx, repeats=n_samples_per_generator
            )

        self.clf = KNeighborsTimeSeriesClassifier(n_neighbors, n_jobs=n_jobs)
        self.clf.fit(X, y)

        print("Finished preparing DTW-KNN classifier")

    def predict(self, X):
        return self.clf.predict(X)