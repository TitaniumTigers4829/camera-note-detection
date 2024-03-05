This aims to detect notes with color filtering and contour finding.

This *should* be faster than neural networks, and it allows us to use a camera and coprocessor rather than a limelight.

For the actual code being used to send network table values, use main. To tune your upper and lower hsv thresholds, use calibrate_bounds and try to minimize the green outside of notes.
