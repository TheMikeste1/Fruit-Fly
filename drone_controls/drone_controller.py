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


def findFace(img):
    face_cascade = cv2.CascadeClassifier(
        "./models/cascade/curated_set_cascade.xml"
    )
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(imgGray, 1.2, 8)

    myFaceListC = []
    myFaceListArea = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # get centerx and centery values
        cx = x + w // 2
        cy = y + h // 2
        # calculate the total area
        area = w * h
        # draw a circle where the center is
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        # append these values to the lists we created above
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)
    # get the index of the max area detected over coming frame
    # so that focus on the bigger face if there are multiple faces
    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]


def trackFace(info, w, pid, pError):
    fb = 0  # forward backward velocity
    area = info[1]  # area value
    x, y = info[0]  # centerx and centery values

    error = x - w // 2  # what is the difference between our object and the center
    speed = pid[0] * error + pid[1] * (
        error - pError
    )  # calculate yaw value with using PID method
    speed = int(np.clip(speed, -100, 100))

    if (
        area > fbRange[0] and area < fbRange[1]
    ):  # if the object is in the range we indicated [6200,6800], do not move
        fb = 0
    elif area > fbRange[1]:  # if it is too close, move backward
        fb = -20
    elif (
        area < fbRange[0] and area != 0
    ):  # if it is too far and if it detects a face, move forward
        fb = 20

    if x == 0:  # if it detects no face, both speed and error will be 0, so the drone
        # would not rotate
        speed = 0
        error = 0

    print(speed, fb)
    drone_obj.send_rc_control(0, fb, 0, speed)
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
        img, info = findFace(img)
        image_to_write = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        writer.write(image_to_write)
        pError = trackFace(info, w, pid, pError)
        print("Center:", info[0], "Area:", info[1])
        cv2.imshow("Output", img)
        # if we click 'q' then land the drone
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    writer.close()
    drone_obj.land()
    drone_obj.streamoff()
