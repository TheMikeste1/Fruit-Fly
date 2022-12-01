import torchvision
from tqdm.auto import tqdm

import create_apple_model as cam

new_path = "D:\\temp\\haar_cascade\\"

df_files = cam.get_df_files()
df_files = df_files[df_files["class"] != cam.Categories.APPLE.value].reset_index(
    drop=True
)
cropper = cam.CropToSquare()
for index in tqdm(range(len(df_files))):
    row = df_files.loc[index]
    img_path = row["img_path"]
    image = torchvision.io.read_image(img_path)
    class_ = "p" if row["class"] == cam.Categories.APPLE.value else "n"
    if class_ == "p":
        image = cropper(image)
    torchvision.io.write_jpeg(image, new_path + class_ + "\\" + str(index) + ".jpg")
