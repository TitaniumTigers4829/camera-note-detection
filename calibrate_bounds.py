import cv2
import numpy as np

# Create a window to adjust the lower and upper bounds
cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)  # Use WINDOW_NORMAL to allow resizing
cv2.resizeWindow("Trackbars", 600, 300)  # Set the size of the window (width, height)

# Create trackbars
cv2.createTrackbar("Hue Lower", "Trackbars", 1, 179, lambda x: None)
cv2.createTrackbar("Saturation Lower", "Trackbars", 80, 255, lambda x: None)
cv2.createTrackbar("Value Lower", "Trackbars", 130, 255, lambda x: None)
cv2.createTrackbar("Hue Upper", "Trackbars", 6, 179, lambda x: None)
cv2.createTrackbar("Saturation Upper", "Trackbars", 255, 255, lambda x: None)
cv2.createTrackbar("Value Upper", "Trackbars", 255, 255, lambda x: None)

# Specify the camera index (usually 0 for built-in webcam)
camera_index = 1

# Open the camera
cap = cv2.VideoCapture(camera_index)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Convert frame from BGR to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

        # Threshold the HSV image to get only orange colors
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour (clump) of orange pixels
        if contours:
            # Draws Everything else it's detecting
            cv2.drawContours(frame, contours, -1, [0, 255, 0], 1)
            # Gets the largest contour and draws it on
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(frame, [largest_contour], 0, [255, 0, 0], 2)

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
