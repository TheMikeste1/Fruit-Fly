import torch
import torchvision
import kaggle


def download_data():
    api = kaggle.KaggleApi()
    api.authenticate()
    api.dataset_download_files("whichtree", path="data", unzip=False)


def main():
    download_data()


if __name__ == "__main__":
    main()
