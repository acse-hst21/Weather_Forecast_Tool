import pytest
import torch

import storm_forcast as sf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.mark.parametrize('input_tensor', [
    torch.rand((5, 1, 1, 28, 28)).to(device)
])
def test_model_output_shape(input_tensor):
    model = sf.Seq2Seq(num_channels=1, num_kernels=1,
                       kernel_size=(3, 3), padding=(1, 1), activation='relu',
                       frame_size=(28, 28), num_layers=3).to(device)
    output_tensor = model(input_tensor)
    assert output_tensor.shape == torch.Size([5, 1, 28, 28])
