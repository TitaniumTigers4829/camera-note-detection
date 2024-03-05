import cv2
import numpy as np
from networktables import NetworkTablesInstance
from cscore import CameraServer


# Set your team number
TEAM_NUMBER = 4829
# Specify the camera index (usually 0 for built-in webcam)
CAMERA_INDEX = 1
# Define lower and upper bounds for orange color in HSV
LOWER_ORANGE_HSV = np.array([4, 180, 80])
UPPER_ORANGE_HSV = np.array([6, 255, 255])
# The minimum contour area to detect a note
MINIMUM_CONTOUR_AREA = 400
# The threshold for a contour to be considered a disk
CONTOUR_DISK_THRESHOLD = 1


def find_largest_orange_contour(hsv_image: np.ndarray) -> np.ndarray:
    """
    Finds the largest orange contour in an HSV image
    :param hsv_image: the image to find the contour in
    :return: the largest orange contour
    """
    # Threshold the HSV image to get only orange colors
    mask = cv2.inRange(hsv_image, LOWER_ORANGE_HSV, UPPER_ORANGE_HSV)
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        return max(contours, key=cv2.contourArea)


def contour_is_note(contour: np.ndarray) -> bool:
    """
    Checks if the contour is shaped like a note
    :param contour: the contour to check
    :return: True if the contour is shaped like a note
    """
    # Makes sure the contour isn't some random small spec of noise
    contour_is_big_enough = cv2.contourArea(contour) >= MINIMUM_CONTOUR_AREA
    # Basically checks that the contour has a hole in the middle, and thus also checks that it is mostly on screen
    contour_is_disk = cv2.contourArea(cv2.convexHull(contour)) / cv2.contourArea(contour) > CONTOUR_DISK_THRESHOLD

    return contour_is_big_enough and contour_is_disk


def main():
    # Connects to the robot
    network_table_instance = NetworkTablesInstance.getDefault()
    network_table_instance.startClientTeam(TEAM_NUMBER)
    network_table_instance.startDSClient()
    # Gets the camera
    camera_instance = CameraServer.getInstance()
    camera = camera_instance.startAutomaticCapture()
    camera.setResolution(1920, 1080)
    sink = camera_instance.getVideo()
    img = np.zeros(shape=(1920, 1080, 3), dtype=np.uint8)

    while True:
        # Capture frame-by-frame
        time, frame = sink.grabFrame(img)
        if time == 0:
            print("Error: Unable to capture frame")

        # Converts from BGR to HSV
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        contour = find_largest_orange_contour(frame_hsv)
        if contour is not None and contour_is_note(contour):
            # Gets the bounding box for the note, and draws it
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
