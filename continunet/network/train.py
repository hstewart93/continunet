"""Train the ContinUNet model on SDC1 dataset publically available by calling Kaggle API."""

import os
from kaggle.api.kaggle_api_extended import KaggleApi

from config import KAGGLE_DATASET, TRAIN_DATASET_PATH


class ApiCaller:
    def __init__(self):
        self.api = KaggleApi()

    def download_dataset(self, dataset_name: str, save_path: str):
        """Download the dataset from Kaggle and save it to the specified path."""
        os.makedirs(save_path, exist_ok=True)
        self.api.dataset_download_files(dataset_name, path=save_path, unzip=True)
        return self


class UnetTrainer:
    def __init__(self):
        self.data_api = ApiCaller()

    def get_data(self):
        # check if data is already downloaded
        if not os.path.exists(KAGGLE_DATASET):
            self.data_api.download_dataset(TRAIN_DATASET_PATH, KAGGLE_DATASET)
        return self

    def load_data(self):
        pass
