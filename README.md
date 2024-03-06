This aims to detect notes with color filtering and contour finding.

This *should* be faster than neural networks, and it allows us to use a camera and coprocessor rather than a limelight.

This repo is just for testing and tuning on a laptop with a camera. Please see the example on WPILib for setting up the rPi: https://docs.wpilib.org/en/stable/docs/software/vision-processing/wpilibpi/using-the-raspberry-pi-for-frc.html Also, here is an example for setting up a USB camera for the rPi: https://docs.wpilib.org/en/stable/docs/software/vision-processing/wpilibpi/basic-vision-example.html

To tune your upper and lower hsv thresholds, use calibrate_bounds and try to minimize the green outside of notes.
