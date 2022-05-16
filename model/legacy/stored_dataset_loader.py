"""Module for loading a dataset that is stored on the disk."""
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .prepared_dataset import PreparedDataset


def stored_dataset_loaders_factory(
    path: str,
    batch_size: int = 32,
    test_size_in_perc: float = 0.2,
    random_state: int = None,
    debug_mode: bool = False,
):
    """Prepares a data loader factory that reads the data from a pickle file.

    Parameters
    ----------
    path
    batch_size
    test_size_in_perc
    random_state
    debug_mode
    """
    # prepare dataset
    df = pd.read_pickle(path)

    X = df["series"]
    y = df["labels"]
    n_classes = len(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_in_perc, random_state=random_state
    )

    training_set = PreparedDataset(X_train, y_train)
    train_loader = DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if debug_mode else 4,
    )

    validation_set = PreparedDataset(X_test, y_test)
    val_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if debug_mode else 4,
    )

    return n_classes, True, train_loader, val_loader
