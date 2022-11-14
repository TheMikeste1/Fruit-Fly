import os
import threading

import kaggle
import torch
import torchvision
import zipfile

import frame_extraction
from utils import *


def extract_frames(verbose=True):
    video_path = "data/ml-apples-1.mp4"
    if not os.path.exists(video_path):
        video_path = "data/tmp/ml-apples-1.mp4"
    if not os.path.exists(video_path):
        raise FileNotFoundError(
            "Video data not found. Please download from "
            "https://drive.google.com/file/d/1rUllqH4ub0sCb3yRRxXu7i9uTqhxln_W/view?usp=share_link"
            f" and place in {video_path}"
            "\nNote: I was considering implementing this through the gdrive package, "
            "but it seems like effort. If you want to do it, feel free to implement it."
        )
    verbose_print = get_verbose_print(verbose)

    verbose_print("Extracting frames from video. . .")
    frame_extraction.extract_frames(
        video_path, output_path="data/extracted_frames", verbose=verbose
    )
    verbose_print("Frame extraction complete.")


def download_kaggle_dataset(verbose=True):
    verbose_print = get_verbose_print(verbose)

    if not os.path.exists("data/tmp/whichtree.zip"):
        api = kaggle.KaggleApi()
        api.authenticate()
        verbose_print(
            "Downloading data from "
            "https://www.kaggle.com/competitions/whichtree/data. . ."
        )
        api.competition_download_files("whichtree", path="data/tmp")
        verbose_print("Kaggle download complete.", end=" ")
    else:
        verbose_print("Kaggle data already downloaded.", end=" ")
    verbose_print("Unzipping data. . .")
    with zipfile.ZipFile("data/tmp/whichtree.zip", "r") as zip_ref:
        zip_ref.extractall("data/whichtree")
    verbose_print("Unzip complete.")


def download_data(verbose=True):
    threads = []
    if not os.path.exists("data/extracted_frames"):
        t = threading.Thread(target=extract_frames, args=(verbose,))
        threads.append(t)

    if not os.path.exists("data/whichtree"):
        t = threading.Thread(target=download_kaggle_dataset, args=(verbose,))
        threads.append(t)

    verbose_print = get_verbose_print(verbose)
    if len(threads) > 0:
        verbose_print(f"Starting data download using {len(threads)} threads. . .")
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        verbose_print("Data download complete.")
    else:
        verbose_print("Data already downloaded.")


def main():
    download_data()


if __name__ == "__main__":
    main()
