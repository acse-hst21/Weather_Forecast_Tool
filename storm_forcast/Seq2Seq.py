import torch.nn as nn

from .ConvLSTM import ConvLSTM


class Seq2Seq(nn.Module):
    """
    A model that generates sequence from sequence

    We finish off the network by stacking up a few ConvLSTM layers,
    followed by BatchNorm3d layers, and the finally followed by a Conv2d
    layer, which can take the hidden state at the last time step of the
    last layer and predict a frame. Finally, we pass this through a
    Sigmoid activation to get pixel values between 0 and 1.
    """
    def __init__(self, num_channels, num_kernels, kernel_size, padding,
                 activation, frame_size, num_layers):
        """
        Parameters
        ----------
        num_channels: int
            Number of channels of input image.
        num_kernels: int
            Number of feature map output of convolutional layers
        kernel_size: (int, int)
            Size of the convolutional kernel.
        padding: Union[str, _size_2_t]
            Padding added to the input tensor
        activation: str
            Activation functions type ('tanh' or 'relu')
        frame_size: (int, int)
            Size of the input image
        num_layers: int
            Number of internal hidden layers
        Notes
        ----------
        See https://github.com/sladewinter/ConvLSTM
        """
        super(Seq2Seq, self).__init__()

        self.sequential = nn.Sequential()

        # Add First layer (Different in_channels than the rest)
        self.sequential.add_module(
            "convlstm1", ConvLSTM(
                in_channels=num_channels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding,
                activation=activation, frame_size=frame_size)
        )

        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features=num_kernels)
        )

        # Add rest of the layers
        for i in range(2, num_layers + 1):
            self.sequential.add_module(
                f"convlstm{i}", ConvLSTM(
                    in_channels=num_kernels, out_channels=num_kernels,
                    kernel_size=kernel_size, padding=padding,
                    activation=activation, frame_size=frame_size)
            )

            self.sequential.add_module(
                f"batchnorm{i}", nn.BatchNorm3d(num_features=num_kernels)
            )

            # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv2d(
            in_channels=num_kernels, out_channels=num_channels,
            kernel_size=kernel_size, padding=padding)

    def forward(self, X):
        # Forward propagation through all the layers
        output = self.sequential(X)

        # Return only the last output frame
        output = self.conv(output[:, :, -1])

        return nn.Sigmoid()(output)
