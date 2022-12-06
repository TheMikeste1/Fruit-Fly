import torchvision
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map

import create_apple_model as cam

new_path = "D:\\temp\\haar_cascade\\"

def save_image(row):
    index = row[0]
    row = row[1]
    cropper = cam.CropToSquare()
    img_path = row["img_path"]
    image = torchvision.io.read_image(img_path)
    class_ = "p" if row["class"] == cam.Categories.APPLE.value else "n"
    if class_ == "p":
        image = cropper(image)
    torchvision.io.write_jpeg(image,
                              new_path + class_ + "\\" + str(index) + ".jpg")

df_files = cam.get_df_files()
thread_map(save_image, list(df_files.iterrows()), max_workers=8)
