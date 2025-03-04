import torch
import torch.nn as nn

from .ConvLSTMCell import ConvLSTMCell

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConvLSTM(nn.Module):
    """
    a single ConvLSTM layer in our network.
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size, padding, activation, frame_size):
        """
        Parameters
        ----------
        in_channels: int
            Number of channels of input tensor.
        out_channels: int
            Number of output feature maps
        kernel_size: (int, int)
            Size of the convolutional kernel.
        padding: Union[str, _size_2_t]
            Padding added to the input tensor
        activation: str
            Activation functions type ('tanh' or 'relu')
        frame_size: (int, int)
            Size of the input image

        Notes
        ----------
        See https://github.com/sladewinter/ConvLSTM
        """
        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels

        # We will unroll this over time steps
        self.convLSTMcell = ConvLSTMCell(
            in_channels,
            out_channels,
            kernel_size,
            padding,
            activation,
            frame_size)

    def forward(self, X):
        # X is a frame sequence (batch_size, num_channels, seq_len, height,
        # width)

        # Get the dimensions
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len,
                             height, width, device=device)

        # Initialize Hidden State
        H = torch.zeros(batch_size, self.out_channels,
                        height, width, device=device)

        # Initialize Cell Input
        C = torch.zeros(batch_size, self.out_channels,
                        height, width, device=device)

        # Unroll over time steps
        for time_step in range(seq_len):
            H, C = self.convLSTMcell(X[:, :, time_step], H, C)

            output[:, :, time_step] = H

        return output
