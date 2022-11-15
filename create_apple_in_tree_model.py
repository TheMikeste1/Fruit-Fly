import enum
import os
import threading

import cv2
import kaggle
import pandas as pd
import torch
import torchvision
import zipfile

import frame_extraction
from utils import *


class Categories(enum.Enum):
    NO_APPLES = 0
    HAS_APPLES = 1


def download_kaggle_dataset(verbose=True):
    user_home = os.path.expanduser(f"~{os.path.sep}.kaggle{os.path.sep}kaggle.json")
    if not os.path.exists(user_home) and "KAGGLE_CONFIG_DIR" not in os.environ.values():
        # Since kaggle.json isn't in the home directory and
        # the environment variable isn't set, it needs to be here.
        assert os.path.exists(os.getcwd() + os.path.sep + "kaggle.json"), (
            f"Please put your kaggle.json file either in {os.getcwd()}"
            f" or {user_home} and try again"
        )
        os.environ["KAGGLE_CONFIG_DIR"] = os.getcwd()

    verbose_print = get_verbose_print(verbose)

    if not os.path.exists("data/tmp/whichtree.zip"):
        api = kaggle.KaggleApi()
        api.authenticate()
        verbose_print(
            "KAGGLE: Downloading data from "
            "https://www.kaggle.com/competitions/whichtree/data. . ."
        )
        api.competition_download_files("whichtree", path="data/tmp")
        verbose_print("KAGGLE: Kaggle download complete.", end=" ")
    else:
        verbose_print("KAGGLE: Kaggle data already downloaded.", end=" ")
    verbose_print("KAGGLE: Unzipping data. . .")
    with zipfile.ZipFile("data/tmp/whichtree.zip", "r") as zip_ref:
        zip_ref.extractall("data/whichtree")
    verbose_print("KAGGLE: Unzip complete.")


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

    verbose_print("FRAMES: Extracting frames from video. . .")
    frame_extraction.extract_frames(
        video_path, output_path="data/extracted_frames", verbose=verbose
    )
    verbose_print("FRAMES: Frame extraction complete.")

    # Let's chop off the top and bottom, since most apples are in the center.
    # That way we can convert them into the desired size more easily.
    verbose_print("FRAMES: Processing frames. . .")
    for root, dirs, files in os.walk("data/extracted_frames"):
        for file_path in files:
            if not file_path.lower().endswith(".jpg"):
                continue
            full_path = os.path.join(root, file_path)
            img = cv2.imread(full_path)
            min_axis = min(img.shape[:2])
            img = img[
                (img.shape[0] - min_axis) // 2 : (img.shape[0] + min_axis) // 2,
                (img.shape[1] - min_axis) // 2 : (img.shape[1] + min_axis) // 2,
            ]
            cv2.imwrite(full_path, img)
    verbose_print("FRAMES: Done processing frames.")


def get_df_files():
    df_frame_files = pd.DataFrame(
        [
            os.path.join(root, f)
            for root, dirs, files in os.walk("data/extracted_frames")
            for f in files
            if f.lower().endswith(".jpg")
        ],
        columns=["img_path"],
    )

    df_frame_files["class"] = Categories.HAS_APPLES.value

    df_whichtree = pd.DataFrame(
        [
            os.path.join(root, f)
            for root, dirs, files in os.walk("data/whichtree")
            for f in files
            if f.lower().endswith(".jpg")
        ],
        columns=["img_path"],
    )

    df_whichtree["class"] = Categories.NO_APPLES.value

    df_files = pd.concat([df_frame_files, df_whichtree], ignore_index=True).reset_index(
        drop=True
    )

    return df_files


def main():
    download_data()
    df_files = get_df_files()
    print(f"Total number of images: {len(df_files)}")


if __name__ == "__main__":
    main()
