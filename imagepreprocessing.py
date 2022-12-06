import cv2
import numpy as np


def process_img(img_path):
    """Feature Extraction for a single image

    Args:
        img_path (str): string containing the path to the desired image

    Returns:
        list: one dimensional list containg the counts of occurances of different h, s, and v values in the image
    """
    img = cv2.imread(img_path)
    if img.shape != (100, 100, 3):
        img = cv2.resize(img, (100, 100))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    counts = [[], [], []]
    # Count the occurances of each h, s, and v values
    counts[0] = cv2.calcHist(hsv, [0], None, [179], [0, 179]).flatten("C")
    counts[1] = cv2.calcHist(hsv, [1], None, [255], [0, 255]).flatten("C")
    counts[2] = cv2.calcHist(hsv, [2], None, [255], [0, 255]).flatten("C")

    # Flatten the overall array to be one dimension
    return np.array([item for channel in counts for item in channel])


# Run once before training
def resize_data():
    IMG_SHAPE = (100, 100)
    """
    Resizes all data images to be (100x100)
    Expects a file structure like
    data/
      train/
        ripe/
          img.jpg
        unripe/
          img.jpg
      test/
        ripe/
          img.jpg
        unripe/
          img.jpg
      val/
        ripe/
          img.jpg
        unripe/
          img.jpg
    """
    data_path = "data/"
    for tvt in os.listdir(data_path):  # iterate train val test split
        ttest_path = os.path.join(data_path, tvt)

        for rur in os.listdir(ttest_path):  # iterate ripe unripe dirs
            ripe_path = os.path.join(ttest_path, rur)

            for name in os.listdir(ripe_path):  # iterate image names
                img_path = os.path.join(ripe_path, name)

                img = cv2.imread(img_path)
                if img.shape != (IMG_SHAPE[0], IMG_SHAPE[1], 3):
                    print(img.shape)
                    assert False
                # img = cv2.resize(img, IMG_SHAPE)
                # cv2.imwrite(img_path, img)
