import os

import torchvision
from tqdm.auto import tqdm

import create_apple_model as cam

path = "D:\Downloads\Apples\p"
cropper = cam.CropToSquare()
for root, dirs, files in os.walk(path):
    for file in tqdm(files):
        img_path = os.path.join(root, file)
        image = torchvision.io.read_image(img_path)
        image = cropper(image)
        torchvision.io.write_jpeg(image, img_path)





