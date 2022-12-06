import cv2
import numpy as np
from djitellopy import tello
import time

from utils import VideoWriter

fbRange = [6200, 6800]
pid = [0.4, 0.4, 0]
pError = 0
w, h = 360, 240

# connect to the drone first
drone_obj = tello.Tello()
drone_obj.connect()
time.sleep(0.25)
print(f"Battery: {drone_obj.get_battery()}%")
time.sleep(0.25)
# start streaming the camera feed
drone_obj.streamon()
time.sleep(0.25)
# take off the drone for a while(2.5 seconds) with speed of 25
drone_obj.takeoff()
time.sleep(0.25)
drone_obj.send_rc_control(0, 0, 2, 0)
time.sleep(2.5)


def find_apple(img):
    apple_cascade = cv2.CascadeClassifier("./models/cascade/curated_set_cascade.xml")
    apples = apple_cascade.detectMultiScale(img)
    apple_centers = []
    apple_areas = []

    for (x, y, w, h) in apples:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # get centerx and centery values
        cx = x + w // 2
        cy = y + h // 2
        # calculate the total area
        area = w * h
        # draw a circle where the center is
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        # append these values to the lists we created above
        apple_centers.append([cx, cy])
        apple_areas.append(area)
    # get the index of the max area detected over coming frame
    # so that focus on the bigger face if there are multiple faces
    if apple_areas:
        i = apple_areas.index(max(apple_areas))
        return img, [apple_centers[i], apple_areas[i]]
    else:
        return img, [[0, 0], 0]


def track_apple(info, w, pid, p_error, area_range):
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

    drone_obj.send_rc_control(0, forward_vel, 0, yaw_vel)
    return error


# cap = cv2.VideoCapture(0)
writer = VideoWriter(
    30,
    w,
    h,
    "demo.mp4",
)

try:
    while True:
        img = drone_obj.get_frame_read().frame
        # _, img = cap.read()
        img = cv2.resize(img, (w, h))
        img, info = find_apple(img)
        image_to_write = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        writer.write(image_to_write)
        pError = track_apple(info, w, pid, pError, fbRange)
        print("Center:", info[0], "Area:", info[1])
        cv2.imshow("Output", img)
        # if we click 'q' then land the drone
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    writer.close()
    drone_obj.land()
    drone_obj.streamoff()
