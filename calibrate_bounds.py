# This file is just used for tuning for different lighting conditions
import cv2
import numpy as np


def nothing(x):
    pass


# Create a window to adjust the lower and upper bounds
cv2.namedWindow("Trackbars")
cv2.createTrackbar("Hue Lower", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("Saturation Lower", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Value Lower", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Hue Upper", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("Saturation Upper", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Value Upper", "Trackbars", 0, 255, nothing)

# Initialize with your initial estimate or default values
cv2.setTrackbarPos("Hue Lower", "Trackbars", 0)
cv2.setTrackbarPos("Saturation Lower", "Trackbars", 160)
cv2.setTrackbarPos("Value Lower", "Trackbars", 170)
cv2.setTrackbarPos("Hue Upper", "Trackbars", 15)
cv2.setTrackbarPos("Saturation Upper", "Trackbars", 255)
cv2.setTrackbarPos("Value Upper", "Trackbars", 255)

# Specify the camera index (usually 0 for built-in webcam)
camera_index = 0

# Open the camera
cap = cv2.VideoCapture(camera_index)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Get current trackbar positions
        hue_lower = cv2.getTrackbarPos("Hue Lower", "Trackbars")
        saturation_lower = cv2.getTrackbarPos("Saturation Lower", "Trackbars")
        value_lower = cv2.getTrackbarPos("Value Lower", "Trackbars")
        hue_upper = cv2.getTrackbarPos("Hue Upper", "Trackbars")
        saturation_upper = cv2.getTrackbarPos("Saturation Upper", "Trackbars")
        value_upper = cv2.getTrackbarPos("Value Upper", "Trackbars")

        # Define lower and upper bounds for orange color in HSV
        lower_orange = np.array([hue_lower, saturation_lower, value_lower])
        upper_orange = np.array([hue_upper, saturation_upper, value_upper])

        # Convert frame from BGR to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image to get only orange colors
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour (clump) of orange pixels
        if contours:
            # Gets the largest contour and draws it on
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(frame, [largest_contour], 0, [0, 255, 0], 2)

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
