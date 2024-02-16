import cv2
import numpy as np
from networktables import NetworkTables


# Specify the camera index (usually 0 for built-in webcam)
camera_index = 1

# Open the camera
cap = cv2.VideoCapture(camera_index)

# TODO We need to tune these values as right now for the specific camera, these are tuned for the laptop camera
# Define lower and upper bounds for orange color in HSV
lower_orange = np.array([4, 180, 80])
upper_orange = np.array([6, 255, 255])

# Define the minimum contour area to detect a note
minimum_area_pixels = 400

# The number of pixels the bounding box should be away from the edge of the frame
detection_padding = 5


def find_largest_orange_clump(image):
    # Convert frame from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only orange colors
    mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (clump) of orange pixels
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) >= minimum_area_pixels:
            print(cv2.contourArea(largest_contour))
            # Get the bounding box of the largest contour
            x_coord, y_coord, width, height = cv2.boundingRect(largest_contour)
            # Return the position (x, y) and size (w, h) of the largest clump
            return (x_coord, y_coord), (width, height)
    return None, None


# Connects to the robot
# TODO: Figure out what this address should be, I think it's this or maybe roborio-4829-frc.local
NetworkTables.initialize(server="127.0.0.1")
smart_dashboard = NetworkTables.getTable("SmartDashboard")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Find the largest clump of orange pixels
        position, size = find_largest_orange_clump(frame)

        if position is not None and size is not None:
            x, y = position
            w, h = size
            # Stops it from detecting disks partially offscreen
            # TODO: Once we have a fixed camera angle, we could probably find a ratio between disk width and height
            if (x > detection_padding and y > detection_padding and
                    x + w < frame.shape[1] - detection_padding and y + h < frame.shape[0] - detection_padding):
                # Draw rectangle around the largest clump
                cv2.rectangle(frame, position, (x + w, y + h), (0, 255, 0), 2)
            # TODO: Add in code to send values about the notes it sees here

            smart_dashboard.putBoolean("Can See Note", True)
        else:
            smart_dashboard.putBoolean("Can See Note", False)

        # Display the resulting frame
        cv2.imshow("Frame", frame)

        # Break the loop if "q" is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        print("Error: Unable to capture frame")
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
