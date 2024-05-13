"""Train the ContinUNet model on SDC1 dataset publically available by calling Kaggle API."""

import os
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()

DATASET_NAME = "harrietstewart/continunet"
SAVE_PATH = "continunet/network/data"

os.makedirs(SAVE_PATH, exist_ok=True)

api.dataset_download_files(DATASET_NAME, path=SAVE_PATH, unzip=True)
