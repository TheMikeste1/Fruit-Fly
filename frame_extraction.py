__all__ = ["extract_frames"]

import os

import cv2

from utils import get_verbose_print

OUTPUT_PATH = "data/"  # Path to directory of output images
OUTPUT_FILE_NAME = "IMG_"  # Name of resulting images before frame number
OUTPUT_EXT = ".jpg"  # File extension of saved image
SAVE_NTH = 1  # Save every nth image, increase for less images
OUTPUT_NUM_DIGITS = 4  # Number of numberical digits for file name formatting


def extract_frames(
    video_path: str,
    output_path: str = OUTPUT_PATH,
    output_file_name: str = OUTPUT_FILE_NAME,
    output_ext: str = OUTPUT_EXT,
    save_nth: int = SAVE_NTH,
    output_num_digits: int = OUTPUT_NUM_DIGITS,
    verbose: bool = True,
):
    verbose_print = get_verbose_print(verbose)

    # Get the input video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise FileNotFoundError(f"Unable to open video at {video_path}.")

    if not output_path.endswith(os.path.sep):
        output_path += os.path.sep

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Read frames and write it to a file
    frame_num = 0
    while video.isOpened():
        ret, frame = video.read()

        # Frames were being read flipped so flip it back
        frame = cv2.flip(frame, 0)

        # break if video ends
        if not ret:
            break

        if frame_num % save_nth == 0:
            cv2.imwrite(
                f"{output_path}{output_file_name}{frame_num:>0{output_num_digits}d}{output_ext}",
                frame,
            )
        frame_num += 1
    video.release()
    verbose_print(f"Final frame number {frame_num - 1}")


if __name__ == "__main__":
    video_path = input("Enter the path to the video to extract frames from: ")
    extract_frames(video_path)
