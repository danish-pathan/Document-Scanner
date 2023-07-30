import numpy as np
import cv2
import imutils
from skimage.filters import threshold_local
from four_point import four_point_transform

# image = cv2.imread("D:\python\project\doc_scanner\img.jpg")
# image = cv2.imread("D:\python\project\doc-2.png")
image = cv2.imread("input.jpg")
ratio = image.shape[0]/500.0
orig = image.copy()

image = imutils.resize(image, height = 500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
smooth = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(gray, 75, 200)
# cv2.imshow("edged", edged)
# cv2.waitKey()

cnts = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True) [:5]

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)
    if (len(approx) == 4):
        screenCnt = approx
        break
        
wraped = four_point_transform(orig, screenCnt.reshape(4, 2)*ratio)

# cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Image", wraped)
cv2.imwrite("output.jpg", wraped) # saving the output image in JPG format
cv2.waitKey(0)

# cv2.drawContours(image, cnts, -1, (0, 255, 0), 2)
# cv2.imshow("draw", image)
# cv2.waitKey(0)
