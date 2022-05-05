import easyocr
import argparse
import cv2
import numpy as np

img = cv2.imread('a01-000u-01.tif')
cv2.imwrite('a01-000u-01.jpg', img)
