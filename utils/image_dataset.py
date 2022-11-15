import pandas as pd
import torch.utils.data
import torchvision


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, files: pd.DataFrame, transform=None, device=None):
        super().__init__()
        self.files = files.reset_index()
        assert {"img_path", "class"}.issubset(
            self.files.columns
        ), "files must have columns 'img_path' and 'class'"
        self.transform = transform
        if device is None:
            device = torch.device("cpu")
            print("No device specified, using CPU")
        self.device = device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path, label = self.files.loc[index, ["img_path", "class"]]
        image = torchvision.io.read_image(img_path)
        image = image.to(self.device)
        if self.transform is not None:
            image = self.transform(image)
        return image, label.to(self.device)
