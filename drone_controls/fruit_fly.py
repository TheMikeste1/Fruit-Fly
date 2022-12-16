import cv2
from djitellopy import Tello
import numpy as np
import torch
from torch import nn
import torchvision

from create_apple_model import Categories as CAP_Categories
from create_rotten_model import Categories as CRM_Categories
from predict_ripe import RipenessPredictor


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


def detect_apple(model_apple_in_view: nn.Module, image: torch.Tensor, device) -> bool:
    image = image.to(device).unsqueeze(0)
    image = CropToSquare()(image)
    out = model_apple_in_view(image)
    return torch.argmax(out).item() == CAP_Categories.APPLE.value


def detect_rotten(model_detect_rotten: nn.Module, image, device):
    image = image.to(device).unsqueeze(0)
    image = CropToSquare()(image)
    out = model_detect_rotten(image)
    return torch.argmax(out).item() == CRM_Categories.ROTTEN.value


def detect_ripe(model_ripeness_predictor: RipenessPredictor, image):
    return model_ripeness_predictor.predict_ripe(image)


def find_apple_center(drone, model_haar_cascade, image):
    apples = model_haar_cascade.detectMultiScale(image)
    marked_image = image.copy()
    apple_centers = []
    apple_areas = []

    for (x, y, w, h) in apples:
        cv2.rectangle(marked_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # get centerx and centery values
        cx = x + w // 2
        cy = y + h // 2
        # calculate the total area
        area = w * h
        # draw a circle where the center is
        cv2.circle(marked_image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        # append these values to the lists we created above
        apple_centers.append([cx, cy])
        apple_areas.append(area)
    # get the index of the max area detected over coming frame
    # so that focus on the bigger face if there are multiple faces
    if apple_areas:
        i = apple_areas.index(max(apple_areas))
        return marked_image, [apple_centers[i], apple_areas[i]]
    else:
        return marked_image, [[0, 0], 0]


def track_apple(drone, info, w, pid, p_error, area_range):
    forward_vel = 0
    area = info[1]  # area value
    x, y = info[0]  # centerx and centery values

    error = x - w // 2  # what is the difference between our object and the center
    yaw_vel = pid[0] * error + pid[1] * (
        error - p_error
    )  # calculate yaw value with using PID method
    yaw_vel = int(np.clip(yaw_vel, -100, 100))

    if (
        area_range[0] < area < area_range[1]
    ):  # if the object is in the range we indicated, do not move
        forward_vel = 0
    elif area > area_range[1]:  # if it is too close, move backward
        forward_vel = -20
    elif (
        area < area_range[0] and area != 0
    ):  # if it is too far and if it detects an apple, move forward
        forward_vel = 20

    if (
        x == 0
    ):  # if it does not detect an apple, both speed and error should be 0 so the drone
        # will not rotate
        yaw_vel = 0
        error = 0

    drone.send_rc_control(0, forward_vel, 0, yaw_vel)
    return error


def fruit_fly(drone: Tello):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        model_apple_in_view,
        model_haar_cascade,
        model_detect_rotten,
        model_ripeness_predictor,
    ) = load_models(device)

    drone.connect()
    drone.LOGGER.info(f"Battery: {drone.get_battery()}%")

    img_to_tensor = torchvision.transforms.ToTensor()
    error = 0
    rdi_size = (100, 100)
    try:
        while True:
            img = drone.get_frame_read().frame

            # Convert the image to PyTorch tensor with the appropriate coloring
            torch_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            torch_img = img_to_tensor(torch_img)
            apple_detected = (
                detect_apple(model_apple_in_view, torch_img, device)
            )
            if apple_detected:
                # Move towards apple
                marked_image, info = find_apple_center(drone, model_haar_cascade, img)
                error = track_apple(
                    drone, info, img.shape[1], [0.4, 0.4, 0], error, [6200, 6800]
                )

                # Trim image to only include the detected apple
                center_x, center_y = info[0]
                area = info[1]
                if area != 0:
                    side_length = np.sqrt(area)
                    x1 = int(center_x - side_length // 2)
                    x2 = int(center_x + side_length // 2)
                    y1 = int(center_y - side_length // 2)
                    y2 = int(center_y + side_length // 2)
                    torch_img = torch_img[..., y1:y2, x1:x2]
                    trimmed_image = cv2.resize(img[y1:y2, x1:x2], rdi_size)

                    # Detect ripeness/rotten
                    is_rotten = detect_rotten(model_detect_rotten, torch_img, device)
                    is_ripe = detect_ripe(model_ripeness_predictor, trimmed_image)

                    print(f"Rotten: {is_rotten}, Ripe: {is_ripe}")
                    cv2.imshow("Ripeness Detector Image", trimmed_image)

                img = marked_image

            cv2.imshow("Output", img)
            # if we click 'q' then land the drone
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        drone.land()
        drone.streamoff()


def load_mobilenet(
    num_outputs: int,
    device: str | torch.device = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", verbose=False)
    # Replace the last layer with the number of classes we want
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_outputs)
    model.to(device)
    return model


def load_models(
    device,
) -> tuple[nn.Module, cv2.CascadeClassifier, nn.Module, RipenessPredictor]:
    apple_in_view = load_mobilenet(num_outputs=2).to(device)
    apple_in_view.load_state_dict(
        torch.load("models/detect_apple/all_files_16its_2022-11-22_18-33-45.pt")
    )
    apple_in_view.eval()

    haar_cascade = cv2.CascadeClassifier("models/cascade/curated_set_cascade.xml")
    detect_rotten = load_mobilenet(num_outputs=2).to(device)
    detect_rotten.load_state_dict(
        torch.load("models/detect_rotten/all_files_16its_2022-11-23_13-55-28.pt")
    )
    detect_rotten.eval()

    ripeness_predictor = RipenessPredictor("models/ripeness/ripeness_model.sav")

    return (
        apple_in_view,
        haar_cascade,
        detect_rotten,
        ripeness_predictor,
    )


class WebcamTello(Tello):
    def __init__(self):
        import djitellopy

        Tello.threads_initialized = True
        host = "192.168.0.1.WEBCAM"
        super().__init__(host)
        self.responses = djitellopy.tello.drones[host]["responses"]
        self.state = djitellopy.tello.drones[host]["state"]
        self.responses.append("DUMMY")
        self.state["DUMMY"] = "DUMMY"

    def get_udp_video_address(self):
        return 0

    def send_command_with_return(
        self, command: str, timeout: int = Tello.RESPONSE_TIMEOUT
    ) -> str:
        return "okay"

    def send_command_without_return(self, command: str) -> None:
        pass

    def get_state_field(self, key: str):
        self.state[key] = -1
        return super().get_state_field(key)


def main():
    # Swap this line to use the real drone
    # drone = Tello()
    drone = WebcamTello()
    fruit_fly(drone)


if __name__ == "__main__":
    main()
