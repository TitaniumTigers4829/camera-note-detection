import cv2
import numpy as np
from networktables import NetworkTablesInstance
from cscore import CameraServer


# Set your team number
TEAM_NUMBER = 4829
# Specify the camera index (usually 0 for built-in webcam)
CAMERA_INDEX = 0
# Define lower and upper bounds for orange color in HSV
LOWER_ORANGE_HSV = np.array([3, 80, 80])
UPPER_ORANGE_HSV = np.array([6, 255, 255])
# The minimum contour area to detect a note
MINIMUM_CONTOUR_AREA = 400
# The threshold for a contour to be considered a disk
CONTOUR_DISK_THRESHOLD = 0.9


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
    if cv2.contourArea(contour) < MINIMUM_CONTOUR_AREA:
        return False

    # Gets the smallest convex polygon that can fit around the contour
    contour_hull = cv2.convexHull(contour)
    # Fits an ellipse to the hull, and gets its area
    ellipse = cv2.fitEllipse(contour_hull)
    best_fit_ellipse_area = np.pi * (ellipse[1][0] / 2) * (ellipse[1][1] / 2)
    # Returns True if the hull is almost as big as the ellipse
    return cv2.contourArea(contour_hull) / best_fit_ellipse_area > CONTOUR_DISK_THRESHOLD


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
            break

        # Converts from BGR to HSV
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        contour = find_largest_orange_contour(frame_hsv)
        if contour is not None and contour_is_note(contour):
            cv2.ellipse(frame, cv2.fitEllipse(contour), (255, 0, 255), 2)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
