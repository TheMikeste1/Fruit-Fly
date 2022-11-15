import enum
import os
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
import torch
import torch.nn as nn
import torch.utils.data
import torchinfo
import torchvision
from tqdm import tqdm
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


def train(model, df_files: pd.DataFrame, image_size: [int], device: str | torch.device):
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
            torchvision.transforms.RandomApply(
                [
                    torchvision.transforms.RandomErasing(),
                ],
                p=0.8,
            ),
            torchvision.transforms.RandomRotation((-45, 45)),
            torchvision.transforms.RandomApply(
                [
                    torchvision.transforms.RandomCrop(image_size),
                ]
            ),
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
    for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        # training the model
        model.train()
        for data, target in train_loader:
            # move tensors to GPU
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
        lr_scheduler.step()
        torch.save(
            model.state_dict(),
            f"models/mobile_model_apple_trees_{epoch}its_{date_str}.pt",
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
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # print training/validation statistics
        print(
            f"Epoch: {epoch} "
            f"\tTraining Loss: {train_loss:.6f} "
            f"\tValidation Loss: {valid_loss:.6f}"
        )

    plt.plot(train_losses, label="Training loss")
    plt.plot(valid_losses, label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(frameon=False)

    plt.show(block=False)


def test(model, df_files: pd.DataFrame, image_size: [int], device: str | torch.device):
    BATCH_SIZE = 16

    transformer = torchvision.transforms.Compose(
        [
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
        for data, target in test_loader:
            targets.extend(target.cpu().numpy())
            data = data.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            predictions.extend(predicted.cpu().numpy())
    p, r, f, s = precision_recall_fscore_support(targets, predictions)
    print(f"Precision: {p}")
    print(f"Recall: {r}")
    print(f"F1: {f}")
    print(f"Support: {s}")
    print(f"Accuracy: {accuracy_score(targets, predictions)}")
    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    print(cm)


def main():
    download_data()
    df_files = get_df_files()
    print(f"Total number of images: {len(df_files)}")

    image_size = (2592, 1936)  # Original size of the images
    scale_factor = min(*image_size) / 224
    image_size = (image_size[0] // scale_factor, image_size[1] // scale_factor)
    image_size = (int(image_size[0]), int(image_size[1]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(image_size=image_size, num_outputs=2, device=device)
    # model.load_state_dict(torch.load("models/mobile_model_apple_trees_16its.pt"))
    df_train, df_test = train_test_split(df_files, test_size=0.1)
    print(f"Test Files: {len(df_test)}")
    assert (
        df_test["class"].nunique() == 2
    ), "Test files should at least one of each class"

    train(model, df_train, image_size, device)
    test(model, df_test, image_size, device)

    if plt.get_fignums():
        plt.waitforbuttonpress()


if __name__ == "__main__":
    main()
