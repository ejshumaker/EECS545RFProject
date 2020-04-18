import cv2
import numpy as np

orig = cv2.imread('data/cheetah_results/cheetah_frm00066_orig.png', cv2.IMREAD_ANYCOLOR)
img = cv2.imread('data/cheetah_results/cheetah_frm00066.png', cv2.IMREAD_GRAYSCALE)

cv2.imshow('frame', img)
cv2.waitKey(0)

# find contours
contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    x, y, dx, dy = cv2.boundingRect(contour)
    if dx > 10 and dy > 10:
        cv2.rectangle(orig, (x, y), (x + dx, y + dy), (0, 255, 0))

# create an empty image for contours
img_contours = orig.copy()
# draw the contours on the empty image
# cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)
# save image
# cv2.imwrite('data/contours.png', img_contours) 

cv2.imshow('frame', img_contours)
cv2.waitKey(0)
