import os
import tarfile
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import radiant_mlhub
import torch
from PIL import Image
from torch.utils.data import Dataset


class StormDataset(Dataset):
    """
    A dataset class that stores storm data

    Collection of data download, loading, pre-processing functions
    """

    def __init__(
            self,
            root_dir,
            storm_list,
            test=False,
            transform=None,
            seq_size=10,
            download=False,
            surprise=False):
        """
        Parameters
        ----------
        root_dir: str
            The path to the data set (can be a relative path)
        storm_list: list
            Storm ID to load
        test: bool
            Whether to read the test set. Default is False
        transform: class
            The transformation to be made to the image
        seq_size: int
            The length of each sequence. Default is 10
        download: bool
            Whether to download and decompress the dataset before
            loading. Default is False
        surprise: bool
            Compatible for Surprise Storm, whether to read special types of
            folders. Default is False
        """
        self.root_dir = root_dir  # Dataset path
        self.storm_list = storm_list
        self.test = test
        self.transform = transform  # Image transform function
        self.seq_size = seq_size
        self.surprise = surprise
        if download:
            self._download()
        self.data = self._load_data()  # X, y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def _load_data(self):
        """
        Read the file structure from the folder, filter out the storm
        ID you want, and read the images that will be associated.
        Re-structure the data according to seq_size.

        e.g.
        If we have 10 data [1,2,3,4,5,6,7,8,9,10], seq_size is 6,
        the re-structured data will be
        [[1,2,3,4,5,6], [2,3,4,5,6,7], [3,4,5,6,7,8],
        [4,5,6,7,8,9], [5,6,7,8,9,10]
        """
        # Gets the dataframe containing the storm ID, order, and file path
        dataset_structure = self._get_structure(test=self.test)
        # Filter specified storm
        result = dataset_structure.loc[dataset_structure['storm'].isin(
            self.storm_list)]
        # Save the required file path
        self.paths = []
        for _, row in result.iterrows():
            self.paths.append(row['path'])
        #  Read picture data and transform it
        img_data = []
        for path in self.paths:
            img = Image.open(path)
            if self.transform is not None:
                img = self.transform(img)
            img_data.append(img)
        # Restructure data to sequence
        img_sub_list = []
        for i in range(len(img_data)):
            j = i + self.seq_size
            if j > len(img_data):
                break
            img_sub_list.append(torch.cat(img_data[i: j]))
        return torch.stack(img_sub_list)

    def _download(self):
        """
        Download the raw data set from the Radiant MLHub and unzip it.
        """
        os.environ['MLHUB_API_KEY'] = \
            '7a080b475bd095422cf2709aaf34cd4e97e355c906e33d5094eda7eba6b0b321'
        # Download
        download_dir = Path(self.root_dir).expanduser().resolve()
        dataset = radiant_mlhub.Dataset.fetch(
            'nasa_tropical_storm_competition')
        # Extract
        archive_paths = dataset.download(output_dir=download_dir)
        for archive_path in archive_paths:
            print(f'Extracting {archive_path}...')
            with tarfile.open(archive_path) as tfile:
                tfile.extractall(path=download_dir)
        print('Done')

    def _get_structure(self, test=False):
        """
        Parameters
        ----------
        test: bool
            Whether to load the test data set
        """
        # Determine the folder name to read based on the parameters
        train_data = []
        if test:
            train_source = 'nasa_tropical_storm_competition_test_source'
        else:
            train_source = 'nasa_tropical_storm_competition_train_source'

        if self.surprise and test:
            train_source = \
                'nasa_tropical_storm_competition_surprise_storm_test_source'
        if self.surprise and not test:
            train_source = \
                'nasa_tropical_storm_competition_surprise_storm_train_source'

        # Find all JPG and store it in pandas
        data_dir = Path(self.root_dir).expanduser().resolve()
        jpg_names = glob(str(data_dir / train_source / '**' / '*.jpg'))
        for jpg_path in jpg_names:
            jpg_path = Path(jpg_path)
            image_id = '_'.join(jpg_path.parent.stem.rsplit('_', 3)[-2:])
            storm_id, step_id = image_id.split('_')
            train_data.append([
                storm_id,
                step_id,
                jpg_path
            ])
        # Sort by Storm and step
        train_df = pd.DataFrame(
            np.array(train_data),
            columns=['storm', 'step', 'path']
        ).sort_values(by=['storm', 'step']).reset_index(drop=True)
        return train_df
