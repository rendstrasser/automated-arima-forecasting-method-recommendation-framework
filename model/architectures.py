"""Module for pyTorch neural network architectures for classifying sample generators."""
import torch


class PrototypeTemp(torch.nn.Module):
    """Convolutional + feed-forward neural network model.

    Structure is aligned to the style of a ResNet.
    """

    def __init__(self, nr_of_classes: int):
        super(PrototypeTemp, self).__init__()

        self._fft_conv_block = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(kernel_size=2, stride=2),
            torch.nn.Conv1d(in_channels=16, out_channels=8, kernel_size=7),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )

        self._linear_layers = torch.nn.Sequential(
            torch.nn.Linear(2848, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, nr_of_classes),
        )

    def forward(self, x):
        """Forward-step of network."""
        # y = x[:, :, :1460]
        fft = x[:, :, 1460:]

        fft_out = self._fft_conv_block(fft)

        return self._linear_layers(fft_out)


class PrototypeFft(torch.nn.Module):
    """Convolutional + feed-forward neural network model for FFT input.

    This is the current model that is used for training and testing.
    """

    def __init__(self, nr_of_classes: int):
        super(PrototypeFft, self).__init__()

        self._fft_conv_block = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(kernel_size=2, stride=2),
            torch.nn.Conv1d(in_channels=16, out_channels=8, kernel_size=7),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )

        self._linear_layers = torch.nn.Sequential(
            torch.nn.Linear(2848, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, nr_of_classes),
        )

    def forward(self, x):
        """Forward-step of network."""
        fft_out = self._fft_conv_block(x)

        return self._linear_layers(fft_out)


class Prototype(torch.nn.Module):
    """Convolutional + feed-forward neural network model.

    Structure is aligned to the style of a ResNet.

    This is the current model that is used for training and testing.
    """

    def __init__(self, nr_of_classes: int):
        super(Prototype, self).__init__()

        self._layers = torch.nn.Sequential(
            # ((W-F+2*P )/S)+1
            # W=1460
            # input size: 365 * 4 = 1460
            torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=31),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=31),
            torch.nn.ReLU(),
            # size: 1430
            torch.nn.AvgPool1d(kernel_size=2, stride=2),
            # size: 715
            torch.nn.Conv1d(in_channels=64, out_channels=32, kernel_size=14),
            torch.nn.ReLU(),
            # size: 702
            torch.nn.AvgPool1d(kernel_size=2, stride=2),
            # size: 351
            torch.nn.Conv1d(in_channels=32, out_channels=16, kernel_size=7),
            torch.nn.ReLU(),
            # size: 345
            torch.nn.Flatten(),
            torch.nn.Linear(5392, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, nr_of_classes),
        )

    def forward(self, x):
        """Forward-step of network."""
        return self._layers(x)


# noinspection PyTypeChecker
class PrototypeLegacy(torch.nn.Module):
    """Convolutional + feed-forward neural network model.

    Structure is aligned to the style of a ResNet.

    This was used for evaluations previous to 12.12.21.
    """

    def __init__(self, nr_of_classes: int):
        super(PrototypeLegacy, self).__init__()

        self._layers = torch.nn.Sequential(
            # ((W-F+2*P )/S)+1
            # W=1460
            # input size: 365 * 4 = 1460
            torch.nn.Conv1d(in_channels=1, out_channels=64, kernel_size=31),
            torch.nn.ReLU(),
            # size: 1430
            torch.nn.AvgPool1d(kernel_size=2, stride=2),
            # size: 715
            torch.nn.Conv1d(in_channels=64, out_channels=32, kernel_size=14),
            torch.nn.ReLU(),
            # size: 702
            torch.nn.AvgPool1d(kernel_size=2, stride=2),
            # size: 351
            torch.nn.Conv1d(in_channels=32, out_channels=16, kernel_size=7),
            torch.nn.ReLU(),
            # size: 345
            torch.nn.Flatten(),
            torch.nn.Linear(5520, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, nr_of_classes),
        )

    def forward(self, x):
        """Forward-step of network."""
        return self._layers(x)


class Prototype80Percent(torch.nn.Module):
    """Convolutional + feed-forward neural network model.

    This variant was used mid of October for a more thorough run where
    80 % accuracy was achieved on the validation set.
    """

    def __init__(self, nr_of_classes: int):
        super(Prototype80Percent, self).__init__()

        self._layers = torch.nn.Sequential(
            # ((W-F+2*P )/S)+1
            # W=1460
            # input size: 365 * 4 = 1460
            torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=31),
            torch.nn.ReLU(),
            # size: 1430
            torch.nn.AvgPool1d(kernel_size=2, stride=2),
            # size: 715
            torch.nn.Conv1d(in_channels=32, out_channels=16, kernel_size=14),
            torch.nn.ReLU(),
            # size: 702
            torch.nn.AvgPool1d(kernel_size=2, stride=2),
            # size: 351
            torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=7),
            torch.nn.ReLU(),
            # size: 345
            torch.nn.Flatten(),
            torch.nn.Linear(5520, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, nr_of_classes),
        )

    def forward(self, x):
        """Forward-step of network."""
        return self._layers(x)
