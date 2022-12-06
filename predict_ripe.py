import numpy as np
import numpy.typing as npt
import cv2
import pickle


class RipenessPredictor:
    def __init__(self, model_path):
        with open(model_path, "rb") as model_file:
            self.__model = pickle.load(model_file)

    def predict_ripe(self, img: npt.NDArray) -> bool:
        """Predict the ripeness of an apple based on a BGR image

        Args:
            img (ndarray): nd array containing the BGR representation of an image Shape: (x, y, 3)

        Returns:
            isRipe: True if the apple is predicted to be ripe
        """
        if img.shape != (100, 100, 3):
            img = cv2.resize(img, (100, 100))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        counts = [[], [], []]
        # Count the occurances of each h, s, and v values
        counts[0] = cv2.calcHist(hsv, [0], None, [179], [0, 179]).flatten("C")
        counts[1] = cv2.calcHist(hsv, [1], None, [255], [0, 255]).flatten("C")
        counts[2] = cv2.calcHist(hsv, [2], None, [255], [0, 255]).flatten("C")

        # Flatten the overall array to be one dimension
        X = np.array([item for channel in counts for item in channel]).reshape(1, -1)
        return bool(self.__model.predict(X)[0])
