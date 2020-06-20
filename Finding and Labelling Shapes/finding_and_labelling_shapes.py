import cv2
import numpy as np
import argparse
import imutils

image = cv2.imread("1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

c = max(cnts, key = cv2.contourArea)

print(c[c[:,:,0].argmin()][0])

for c in cnts:
    
    m = cv2.moments(c)
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.circle(image, (cx, cy), 7, (255, 255, 255), -1)
    cv2.putText(image, "center", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
cv2.imshow("", image)
cv2.waitKey(0)