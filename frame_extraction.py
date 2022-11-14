import os

import cv2

OUTPUT_PATH = "data/"  # Path to directory of output images
OUTPUT_FILE_NAME = "IMG_"  # Name of resulting images before frame number
OUTPUT_EXT = ".jpg"  # File extension of saved image
SAVE_NTH = 1  # Save every nth image, increase for less images
OUTPUT_NUM_DIGITS = 4  # Number of numberical digits for file name formatting

# Get the input video
video_path = input("Enter the path to the video to extract frames from: ")
video = cv2.VideoCapture(video_path)
if not video.isOpened():
    print("Unable to open video.")
    exit()

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Read frames and write it to a file
frame_num = 0
while video.isOpened():
    ret, frame = video.read()

    # Frames were being read flipped so flip it back
    frame = cv2.flip(frame, 0)

    # break if video ends
    if not ret:
        break

    if frame_num % SAVE_NTH == 0:
        cv2.imwrite(
            f"{OUTPUT_PATH}{OUTPUT_FILE_NAME}{frame_num:>0{OUTPUT_NUM_DIGITS}d}{OUTPUT_EXT}",
            frame,
        )
    frame_num += 1
print(f"Final frame number {frame_num - 1}")
