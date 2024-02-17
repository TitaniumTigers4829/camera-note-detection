import cv2
import numpy as np
from networktables import NetworkTables


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
    # Threshold the HSV image to get only orange colors
    mask = cv2.inRange(hsv_image, LOWER_ORANGE_HSV, UPPER_ORANGE_HSV)
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        return max(contours, key=cv2.contourArea)


def contour_is_note(contour: np.ndarray) -> bool:
    # Makes sure the contour isn't some random small spec of noise
    contour_is_big_enough = cv2.contourArea(contour) >= MINIMUM_CONTOUR_AREA
    # Basically checks that the contour has a hole in the middle, and thus also checks that it is mostly on screen
    contour_is_disk = cv2.contourArea(cv2.convexHull(contour)) / cv2.contourArea(contour) > CONTOUR_DISK_THRESHOLD

    return contour_is_big_enough and contour_is_disk


def main():
    # Open the camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    # Connects to the robot
    # TODO: Figure out what this address should be, I think it's this or maybe roborio-4829-frc.local
    NetworkTables.initialize(server="127.0.0.1")
    smart_dashboard = NetworkTables.getTable("SmartDashboard")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            # Convert frame from BGR to HSV color space
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Finds the largest orange contour and checks if it looks like a note
            contour = find_largest_orange_contour(frame)
            if contour_is_note(contour):
                x, y, w, h = cv2.boundingRect(contour)
                # Draw rectangle around the largest clump
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                smart_dashboard.putBoolean("Can See Note", True)
            else:
                smart_dashboard.putBoolean("Can See Note", False)

            # Display the resulting frame
            cv2.imshow("Frame", frame)

            # Closes the program if q is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print("Error: Unable to capture frame")
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
