import enum
import os
import zipfile

import numpy as np
import requests
import threading
from datetime import datetime

import cv2
import kaggle
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
import seaborn as sns
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchinfo
import torchvision
from tqdm.auto import tqdm

from utils import *


class Categories(enum.Enum):
    ROTTEN = 0
    RIPE = 1


class CropToSquare:
    def __call__(self, img: torch.Tensor):
        h, w = img.shape[-2:]
        if h > w:
            diff = h - w
            img = img[..., diff // 2 : -diff // 2, :]
        elif w > h:
            diff = w - h
            img = img[..., :, diff // 2 : -diff // 2]
        return img


class RandomPadding:
    def __init__(self, max_padding: int, min_padding: int = 0):
        self.max_padding = max_padding
        self.min_padding = min_padding

    def __call__(self, img: torch.Tensor):
        h, w = img.shape[-2:]
        padding = np.random.randint(self.min_padding, self.max_padding)
        img = torch.nn.functional.pad(img, (padding, padding, padding, padding))
        return img


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

    if not os.path.exists("data/tmp/fruits.zip"):
        api = kaggle.KaggleApi()
        api.authenticate()
        verbose_print(
            "KAGGLE: Downloading data from "
            "https://www.kaggle.com/datasets/moltean/fruits. . ."
        )
        api.dataset_download_files("moltean/fruits", path="data/fruits360", unzip=True)
        verbose_print("KAGGLE: Kaggle download complete.")
    else:
        verbose_print("KAGGLE: Kaggle data already downloaded.")


def download_mendeley_dataset(verbose=True):
    verbose_print = get_verbose_print(verbose)

    if not os.path.exists("data/tmp/fruits.zip"):
        verbose_print(
            "MENDELEY: Downloading data from "
            "https://data.mendeley.com/datasets/bdd69gyhv8/1. . ."
        )
        url = (
            "https://data.mendeley.com/public-files/datasets/bdd69gyhv8/files"
            "/de93ba06-6a58-45e3-913d-837b2ae52acb/file_downloaded"
        )
        # https://stackoverflow.com/a/37573701/10078500
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        bar = tqdm(
            total=total_size_in_bytes, unit="iB", unit_scale=True, desc="MENDELEY"
        )
        with open("data/tmp/mendeley.zip", "wb") as file:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                file.write(data)
        bar.close()
        if total_size_in_bytes != 0 and bar.n != total_size_in_bytes:
            raise Exception("Unable to download Mendeley dataset.")
        verbose_print("MENDELEY: Download complete.", end=" ")
    else:
        verbose_print("MENDELEY: Data already downloaded.", end=" ")
    verbose_print("Unzipping data. . .")
    with zipfile.ZipFile("data/tmp/mendeley.zip", "r") as zip_ref:
        zip_ref.extractall("data/mendeley")
    verbose_print("MENDELEY: Unzip complete.")


def download_data(verbose=True):
    threads = []
    if not os.path.exists("data/mendeley"):
        # https://data.mendeley.com/public-files/datasets/bdd69gyhv8/files/de93ba06
        # -6a58-45e3-913d-837b2ae52acb/file_downloaded
        t = threading.Thread(target=download_mendeley_dataset, args=(verbose,))
        threads.append(t)

    if not os.path.exists("data/fruits360"):
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


def get_df_mendeley():
    df_mendeley = pd.DataFrame(
        [
            os.path.join(root, f)
            for root, dirs, files in os.walk("data/mendeley")
            if root.endswith("Apple")
            for f in files
            if f.lower().endswith(".jpg")
        ],
        columns=["img_path"],
    )

    df_mendeley["class"] = Categories.ROTTEN.value
    df_mendeley.loc[
        df_mendeley["img_path"].str.contains("Fresh"), "class"
    ] = Categories.RIPE.value
    return df_mendeley


def get_df_fruits360():
    df_fruits360 = pd.DataFrame(
        [
            os.path.join(root, f)
            for root, dirs, files in os.walk("data/fruits360/fruits-360-original-size")
            if "apple" in root
            for f in files
            if f.lower().endswith(".jpg")
        ],
        columns=["img_path"],
    )

    df_fruits360["class"] = Categories.RIPE.value
    df_fruits360.loc[
        (
            ~df_fruits360["img_path"].str.contains("rotten")
            & ~df_fruits360["img_path"].str.contains("hit")
        ),
        "class",
    ] = Categories.ROTTEN.value
    return df_fruits360


def get_df_files():
    df_mendeley = get_df_mendeley()
    df_fruits360 = get_df_fruits360()

    df_files = pd.concat([df_mendeley, df_fruits360], ignore_index=True).reset_index(
        drop=True
    )

    return df_files


def load_model(
    image_size: [int],
    num_outputs: int,
    summarize: bool = True,
    device: str | torch.device = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model. . .", end="" if summarize else "\n")
    model = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", verbose=False)
    # Replace the last layer with the number of classes we want
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_outputs)
    model.to(device)

    if summarize:
        print("\rAnalysing model. . .")
        torchinfo.summary(model, input_size=(1, 3, *image_size), device=device)

    return model


def train(
    model,
    df_files: pd.DataFrame,
    image_size: [int],
    device: str | torch.device,
    name: str = "",
):
    NUM_EPOCHS = 16
    BATCH_SIZE = 32
    INITIAL_LEARNING_RATE = 0.01
    LR_SCHEDULER_KWARGS = {"gamma": 0.9}

    train_files, valid_files = train_test_split(df_files, test_size=0.2)

    print(f"Train Files: {len(train_files)}")
    print(f"Validation Files: {len(valid_files)}")
    assert (
        train_files["class"].nunique() == 2
    ), "Train files should at least one of each class"
    assert (
        valid_files["class"].nunique() == 2
    ), "Validation files should at least one of each class"

    # Print out the number of each class in the training and validation sets
    print("Train Files:")
    print(train_files["class"].value_counts())
    print("Validation Files:")
    print(valid_files["class"].value_counts())

    transformer = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomHorizontalFlip(),
            CropToSquare(),
            torchvision.transforms.Resize((image_size[0] * 3, image_size[1] * 3)),
            torchvision.transforms.RandomApply(
                [
                    RandomPadding(256),
                ]
            ),
            torchvision.transforms.RandomRotation((-45, 45)),
            torchvision.transforms.RandomApply(
                [
                    torchvision.transforms.RandomPerspective(),
                ],
            ),
            torchvision.transforms.RandomApply(
                [
                    torchvision.transforms.ColorJitter(),
                ],
            ),
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ConvertImageDtype(torch.float32),
        ]
    )
    train_data = ImageDataset(train_files, transform=transformer, device=device)
    valid_data = ImageDataset(valid_files, transform=transformer, device=device)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=BATCH_SIZE, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_data, batch_size=BATCH_SIZE, shuffle=False
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, **LR_SCHEDULER_KWARGS
    )
    criterion = nn.CrossEntropyLoss()

    print(f"Running on {device}")

    # keeping track of losses
    train_losses = []
    valid_losses = []

    if not os.path.exists("models"):
        os.mkdir("models")
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_writer = SummaryWriter(f"runs/detect_rotten/{name}{date_str}")
    for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        # training the model
        model.train()
        for data, target in train_loader:
            # move tensors to GPU
            summary_writer.add_images("train_images", data, epoch)
            data = data.to(device)
            target = target.to(device)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss wrt model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)

        train_loss = train_loss / len(train_loader.sampler)
        summary_writer.add_scalar("train_loss", train_loss, epoch)
        lr_scheduler.step()
        torch.save(
            model.state_dict(),
            f"models/detect_rotten/{name}{epoch}its_{date_str}.pt",
        )

        # validate the model
        model.eval()
        for data, target in valid_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)

            # update average validation loss
            valid_loss += loss.item() * data.size(0)

        # calculate average losses
        valid_loss = valid_loss / len(valid_loader.sampler)
        summary_writer.add_scalar("valid_loss", valid_loss, epoch)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

    plt.plot(train_losses, label="Training loss")
    plt.plot(valid_losses, label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(frameon=False)

    plt.show(block=False)


def test(
    model,
    df_files: pd.DataFrame,
    image_size: [int],
    device: str | torch.device,
    name: str = "",
):
    BATCH_SIZE = 32

    transformer = torchvision.transforms.Compose(
        [
            CropToSquare(),
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ConvertImageDtype(torch.float32),
        ]
    )
    test_data = ImageDataset(df_files, transform=transformer, device=device)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=BATCH_SIZE, shuffle=False
    )

    model.eval()
    targets = []
    predictions = []
    print("Beginning Testing. . .")
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            targets.extend(target.cpu().numpy())
            data = data.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            predictions.extend(predicted.cpu().numpy())
    p, r, f, s = precision_recall_fscore_support(targets, predictions, zero_division=0)
    print(f"Precision: {p}")
    print(f"Recall: {r}")
    print(f"F1: {f}")
    print(f"Support: {s}")
    print(f"Accuracy: {accuracy_score(targets, predictions)}")
    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    class_names = ["Not Apple", "Apple"]
    df_confusion_matrix = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(df_confusion_matrix)
    plot = sns.heatmap(
        df_confusion_matrix,
        annot=True,
        fmt="d",
        cmap=sns.color_palette("mako", as_cmap=True),
    )
    plot.set_title("Confusion Matrix: " + name)
    plot.set_xlabel("Predicted Label")
    plot.set_ylabel("True Label")
    plt.savefig(f"confusion_matrix_{name}.eps", format="eps")
    plt.show(block=False)


def main():
    download_data()
    name = "all_files_"
    df_files = get_df_files()
    # df_files = get_df_mendeley()
    # df_files = get_df_fruits360()
    print(f"Total number of images: {len(df_files)}")

    image_size = (224, 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(image_size=image_size, num_outputs=2, device=device)
    # model.load_state_dict(torch.load("models/mobile_model_apple_trees_16its.pt"))
    df_train, df_test = train_test_split(df_files, test_size=0.1)
    print(f"Test Files: {len(df_test)}")
    assert (
        df_test["class"].nunique() == 2
    ), "Test files should at least one of each class"

    train(model, df_train, image_size, device, name=name)
    test(model, df_test, image_size, device)

    if plt.get_fignums():
        print(f"Waiting on figures {plt.get_fignums()}; Press any key to continue.")
        plt.waitforbuttonpress()


def load_and_predict():
    image_size = (224, 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        image_size=image_size, num_outputs=2, device=device, summarize=False
    )
    model.load_state_dict(
        torch.load("models/detect_rotten/all_files_16its_2022-11-22_18-33-45.pt")
    )
    df_files = get_df_files()
    # df_files = get_df_mendeley()
    # df_files = get_df_fruits360()
    # df_files = pd.DataFrame(
    #     [
    #         os.path.join(root, f)
    #         for root, dirs, files in os.walk("data/hands_holding_apples")
    #         for f in files
    #         if f.lower().endswith(".jpg")
    #     ],
    #     columns=["img_path"],
    # )
    df_files["class"] = Categories.RIPE.value

    test(model, df_files, image_size, device, name="All Files on Dataset")

    if plt.get_fignums():
        print(f"Waiting on figures {plt.get_fignums()}; Press any key to continue.")
        plt.waitforbuttonpress()


if __name__ == "__main__":
    main()
    # load_and_predict()
