"""Module for sample generator classification training."""
from math import ceil

import numpy as np
import torch.nn
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss

from .architectures import PrototypeFft


def train(
    dataloaders_factory,
    results_path: str = None,
    lr: float = 1e-5,
    weight_decay: float = 1e-5,
    n_epochs: int = 500,
    n_updates_per_epoch: int = 20_000,
    random_state: int = None,
):
    """Performs a complete training procedure.

    Parameters
    ----------
    dataloaders_factory
        Data loader factory to use to create data loaders.
    results_path: str
        Path to store the best-performing model parameters.
        If None, then no model parameters are stored.
    lr: float
        Learning rate.
    weight_decay: float
        Weight decay.
    n_epochs:
        Number of epochs:
    n_updates_per_epoch:
        Number of time series that should be generated and trained on per epoch.
    random_state:
        Seed for training procedure.
    """

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
        dataloaders_factory, lr, n_updates_per_epoch, random_state, weight_decay
    )

    best_score_so_far = -1
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

        val_score = evaluate_accuracy_on_validation_set(model, device, val_loader)
        if val_score > best_score_so_far and results_path:
            best_score_so_far = val_score
            torch.save(model, results_path)

        print(
            f"Epoch {epoch_idx}: training_loss={losses.mean():.9f}"
            + f", accuracy_on_validation={val_score:.9f}"
        )


def train_epoch(
    losses, n_iter_per_epoch, train_loader, device, optimizer, model, loss, n_train_data
):
    """Performs training for a single epoch.

    Parameters
    ----------
    losses
        Loss values that can be filled through training.
    n_iter_per_epoch
        Number of times we need to iterate through the data loader.
    train_loader
        Data loader for training data.
    device
        Device to use for training the model (cpu or cuda).
    optimizer
        Optimizer instance for training.
    model
        Model that we are training.
    loss
        Loss function for evaluating the loss
    n_train_data
        Number of training samples of the data loader
    """
    model.train()

    for i in range(n_iter_per_epoch):
        for batch_idx, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)

            # Reset gradients
            optimizer.zero_grad()

            predictions = model(inputs)

            loss_val = loss(predictions, labels)

            loss_val.backward()
            optimizer.step()

            losses[i * n_train_data + batch_idx] = loss_val


def prepare_training(
    dataloaders_factory, lr, n_updates_per_epoch, random_state, weight_decay
):
    """Performs training for a single epoch.

    Parameters
    ----------
    dataloaders_factory
        Data loader factory to use to create data loaders.
    lr: float
        Learning rate.
    weight_decay: float
        Weight decay.
    n_updates_per_epoch:
        Number of time series that should be generated and trained on per epoch.
    random_state:
        Seed for training procedure.

    Returns
    -------
    device
        Device on which model will be trained on (cpu or cuda)
    loss
        Loss function for the training of the model.
    model
        Model instance to train.
    n_iter_per_epoch
        Number of iterations through the data loader per epoch
    n_train_data
        Number of data in a single data loader run-through.
    optimizer
        Optimizer instance for training.
    n_classes
        Number of classes to predict.
    train_loader
        Training data loder.
    val_loader
        Validation data loader.
    """

    # preparations
    if random_state:
        np.random.seed(random_state)
        torch.random.manual_seed(random_state)
        torch.manual_seed(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        n_classes,
        is_data_loader_size_equal_length,
        train_loader,
        val_loader,
    ) = dataloaders_factory()

    n_train_data = len(train_loader.dataset)
    n_iter_per_epoch = (
        1
        if is_data_loader_size_equal_length
        else ceil(n_updates_per_epoch / n_train_data)
    )

    model = PrototypeFft(n_classes)
    model.to(device, dtype=torch.double)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss = CrossEntropyLoss()

    print(f"Starting training procedure - n_iter_per_epoch={n_iter_per_epoch}")

    return (
        device,
        loss,
        model,
        n_iter_per_epoch,
        n_train_data,
        optimizer,
        n_classes,
        train_loader,
        val_loader,
    )


def evaluate_accuracy_on_validation_set(model, device, validation_loader):
    """Evaluates accuracy of model on the validation set.

    Parameters
    ----------
    model
        Model to evaluate.
    device
        Device on which the model should be trained/evaluated on (cpu or cuda).
    validation_loader
        Validation data loader.

    Returns
    -------
    accuracy_score: float
        Accuracy of the model on the validation set
    predictions: List[int]
        Predictions of the model on the validation set
    labels: List[int]
        Labels of the validation set in the same order as predictions.
    """
    model.eval()

    with torch.no_grad():
        all_predictions = []
        all_labels = []

        for i, data in enumerate(validation_loader):
            inputs, labels = data
            all_labels.extend(labels.numpy().astype(int))

            inputs, labels = inputs.to(device), labels.to(device)

            predictions = model(inputs)
            _, predictions = torch.max(predictions.data, 1)
            predictions = predictions.cpu().numpy().astype(int)
            all_predictions.extend(predictions)

        return accuracy_score(all_labels, all_predictions), all_predictions, all_labels
