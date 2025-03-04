import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import wandb
from torch.utils.data import DataLoader

import pytorch_ssim
from .Seq2Seq import Seq2Seq
from .StormDataset import StormDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value
    and take out any randomness from cuda kernels

    Parameters
    ----------
    seed: int
        Random seed to set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # uses the inbuilt cudnn auto-tuner to find the fastest convolution
    # algorithms. -
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True


def set_device():
    """
    Parameters
    ----------
    If a Nvidia GPU exists, set the computing device to cuda
    """
    global device
    device = 'cpu'
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        device = 'cuda'
        print(
            "Cuda installed! Running on GPU %s!" %
            torch.cuda.get_device_name())

    else:
        print("No GPU available! Running on CPU")
    return True


def make_storm_dataloader(config, test):
    """
    Split data set (training set, validation set, test set)
    and load it in dataloader.

    Parameters
    ----------
    config: class
        hyperparameters to training. Contains `root_dir`,
        `storm_list`, `surprise`, `seq_size`, `download` Etc.
    test: bool
        Whether to load the test set
    """
    storm_dataset = StormDataset(root_dir=config.root_dir,
                                 storm_list=config.storm_list,
                                 surprise=config.surprise,
                                 test=test,
                                 transform=transforms.ToTensor(),
                                 seq_size=config.seq_size,
                                 download=config.download)
    if test:
        def collate_test(batch):
            batch = torch.stack(batch).unsqueeze(1)
            batch = batch.to(device)
            return batch, batch[:, :, -5:]

        # Capture all test data and divide sequence into
        # two parts, data and target (length=5)
        testds = storm_dataset[:]
        test_loader = DataLoader(testds, shuffle=True,
                                 batch_size=config.batch_size,
                                 collate_fn=collate_test)
        return test_loader
    else:
        def collate(batch):
            batch = torch.stack(batch).unsqueeze(1)
            batch = batch.to(device)
            return batch[:, :, 0:config.seq_size - 1], batch[:, :, -1]

        # Use 90% of all sequences as training sets and
        # 10% as verification sets.
        trainds = storm_dataset[:int(len(storm_dataset) * 0.9)]
        validds = storm_dataset[int(len(storm_dataset) * 0.9):]

        # The last image of the sequence is target
        # and the remaining images are data
        train_loader = DataLoader(trainds, batch_size=config.batch_size,
                                  shuffle=False, collate_fn=collate)
        val_loader = DataLoader(validds, batch_size=config.batch_size,
                                shuffle=False, collate_fn=collate)
        return train_loader, val_loader


def set_up(config):
    """
    Prepare the parameters for wandb
    """
    # Get data, make datasets and get data loaders
    train_loader, valid_loader = make_storm_dataloader(
        test=False, config=config)

    # Make model
    model = Seq2Seq(num_channels=1,
                    num_kernels=config.num_kernels,
                    kernel_size=config.kernel_size,
                    padding=config.padding,
                    activation=config.activation,
                    frame_size=config.frame_size,
                    num_layers=config.num_layers).to(device)
    config.model = model

    # Make optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    config.optimizer = optimizer

    # Make loss
    criterion = nn.MSELoss()
    config.criterion = criterion

    # Make ssim loss
    ssim_loss = pytorch_ssim.SSIM()
    config.ssim_loss = ssim_loss

    return model, criterion, optimizer, ssim_loss, train_loader, valid_loader


def train(model, optimizer, criterion, ssim_loss, data_loader):
    """
    Perform a single epoch training on the model

    Parameters
    ----------
    model:
        The model to train
    optimizer:
        The optimizer to use
    criterion:
        Define the method for calculating loss
    ssim_loss:
        Define the method for calculating SSIM Loss
    data_loader:
        A Dataloader containing training data
    """
    model.train()
    train_loss, train_ssim = 0, 0
    for batch_num, (input, target) in enumerate(data_loader, 1):
        input, target = input.to(device), target.to(device)
        output = model(input)
        # The loss is calculated separately and the sum is obtained
        # Since the bigger the SSIM, the better, it is easier
        # to optimize by taking a negative number.
        ssim_out = -ssim_loss(output.cuda(), target.cuda())
        loss = criterion(output.flatten(), target.flatten())
        total_loss = loss + ssim_out
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
        train_ssim += -ssim_out.item()

    return train_loss / \
        len(data_loader.dataset), train_ssim / len(data_loader.dataset)


def validate(model, criterion, ssim_loss, data_loader):
    """
    Perform a single epoch training on the model

    Parameters
    ----------
    model:
        The model to train
    criterion:
        Define the method for calculating loss
    ssim_loss:
        Define the method for calculating SSIM Loss
    data_loader:
        A Dataloader containing training data
    """
    model.eval()
    val_loss, val_ssim = 0, 0
    with torch.no_grad():
        for input, target in data_loader:
            input, target = input.to(device), target.to(device)
            output = model(input)
            ssim_out = -ssim_loss(output.cuda(), target.cuda())
            loss = criterion(output.flatten(), target.flatten())
            val_loss += loss.item()
            val_ssim += -ssim_out.item()

    return val_loss / len(data_loader), val_ssim / len(data_loader)


def train_model(hyperparameters):
    """
    According to the input of the super-parameter training model,
    and save the model. The run progress metrics will be displayed on
    https://wandb.ai/katrina-2022
    """
    # tell wandb to get started
    with wandb.init(config=hyperparameters,
                    project="ConvLSTM_storm",
                    entity="katrina-2022"):
        # Set seed and devices
        set_seed(42)
        set_device()

        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # Set up run with hypeparameters
        model, criterion, optimizer, ssim_loss, \
            train_loader, valid_loader = set_up(config)

        # Let wandb watch the model and the criterion
        wandb.watch(model, criterion)

        for epoch in range(config.epochs):
            train_loss, train_ssim = train(
                model, optimizer, criterion, ssim_loss, train_loader)
            validation_loss, validation_ssim = validate(
                model, criterion, ssim_loss, valid_loader)
            log = {"epoch": epoch + 1,
                   "train_loss": train_loss,
                   "train_ssim": train_ssim,
                   "valid_loss": validation_loss,
                   "valid_ssim": validation_ssim}
            wandb.log(log)

        # save the model to file
        model_save_name = f'ConvLSTM_storm_{config.num_kernels}' + \
                          f'_{config.epochs}.pt'
        path = F"./models/{model_save_name}"
        torch.save(model.state_dict(), path)
        return path
