import numpy as np
import imutils
import cv2

def sort_contours(cnts, method = "left-to-right"):
    
    reverse = False
    i = 0
    
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), 
                                key = lambda b:b[1][i], reverse = reverse))
    
    return (cnts, boundingBoxes)

def draw_contour(image, c, i):
    
    m = cv2.moments(c)
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    
    cv2.putText(image, "#{}".format(i + 1), (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    return image

image = cv2.imread("2.png")

accumEdged = np.zeros(image.shape[:2], dtype = "uint8")

for channel in cv2.split(image):
    channel = cv2.medianBlur(channel,11)
    edged = cv2.Canny(channel, 50, 200)
    accumEdged = cv2.bitwise_or(accumEdged, edged)

cv2.imshow("Edge Map", accumEdged)

cnts = cv2.findContours(accumEdged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)


cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:4]

orig = image.copy()

for (i, c) in enumerate(cnts):
    orig = draw_contour(orig, c, i)

cv2.imshow("Unsorted", orig)
cv2.waitKey(0)

(cnts, boundingBoxes) = sort_contours(cnts)

for (i, c) in enumerate(cnts):
    draw_contour(image, c, i)
    
cv2.imshow("Sorted", image)
cv2.waitKey(0)