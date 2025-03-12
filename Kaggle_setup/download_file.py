from kaggle.api.kaggle_api_extended import KaggleApi

# Kaggle API stuff
api = KaggleApi()
api.authenticate()

# The dataset loaction on kaggle
dataset = 'dataforlyf/mit-af-3seconds'  

download_path = '/home/s2742733/mlp/MLP_CW'  # Change to your location on the drive

# Download and unzip the dataset
api.dataset_download_files(dataset, path=download_path, unzip=True)