import cv2
import numpy as np
images = ["nogoal.jpg", "goal.jpg"]
for image in images:
    # Read in the image
    img = cv2.imread(image)
    if img is None:
        print("image not found.")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply threshold to create binary image
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Set up the detector with parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 100
    params.filterByCircularity = True
    params.minCircularity = 0.5

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(thresh)

    # Draw blobs on the image
    img_with_blobs = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    # Define the size of the output image
    width = 600
    height = 600
    dim = (width, height)

    # Resize the image
    img_with_blobs_small = cv2.resize(img_with_blobs, dim, interpolation = cv2.INTER_LINEAR)

    # Show the image with blobs
    cv2.imshow("Blobs", img_with_blobs_small)
    cv2.waitKey(0)
