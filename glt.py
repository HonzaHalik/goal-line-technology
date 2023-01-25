import cv2
import numpy as np

images = ["nogoal.jpg", "goal.jpg",]
for image in images:
    # Read in the image
    img = cv2.mimread(image)
    # Define the range of colors to detect (white color)
    lower = np.array([180,180,180])
    upper = np.array([255,255,255])

    # Create a mask for the white color
    mask = cv2.inRange(img, lower, upper)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the area of the white object's contour
    for contour in contours:
        area = cv2.contourArea(contour)
        # Fill the black line with white pixels
        mask2 = np.zeros(img.shape, dtype=np.uint8)
        cv2.fillPoly(mask2, [contour], (255, 255, 255))

        # Check if the white object covers the black line
        result = cv2.bitwise_and(img, mask2)
        if np.count_nonzero(result) > 0:
            print("GOAL")
            break
        else:
            print("NO GOAL")
            
    print("image scanned")


