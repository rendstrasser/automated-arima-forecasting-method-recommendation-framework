"""Module for prepared dataset class."""
import numpy as np
from torch.utils.data import Dataset


class PreparedDataset(Dataset):
    """Dataset that contains previously generated data."""

    def __init__(self, X, y):
        self.X = np.expand_dims(np.array(X.values.tolist()), axis=1)
        self.y = y.to_numpy()

    def __len__(self):
        """Length of data."""
        return len(self.X)

    def __getitem__(self, index: int):
        """Retrieves time series at index.

        Parameters
        ----------
        index: int
            Index at which to retrieve the time series.

        Returns
        -------
        x: np.ndarray
            Time series at index
        y: int
            Label of time series at index.
        """
        return self.X[index], self.y[index]
