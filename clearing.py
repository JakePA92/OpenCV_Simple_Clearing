import cv2 as cv
import numpy as np

# Load the image
image = cv.imread('clearing.png')

# Convert the image to grayscale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Apply a binary threshold to the image
_, threshold = cv.threshold(gray, 135, 255, cv.THRESH_BINARY)

# Find contours in the threshold image
contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# Filter contours based on area
min_contour_area = 1000  # Minimum contour area to consider a space "open"
open_spaces = [cnt for cnt in contours if cv.contourArea(cnt) > min_contour_area]

# Draw the open spaces on the image
for contour in open_spaces:
    cv.drawContours(image, [contour], -1, (0, 255, 0), 2)

# Display the image
cv.imshow('Open Spaces', image)
cv.imwrite( 'Open_Spaces.png', image)
cv.waitKey(0)
cv.destroyAllWindows()
