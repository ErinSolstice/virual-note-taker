# from matplotlib import pyplot as plt
import easyocr
# import argparse
import numpy as np
from fpdf import FPDF
from autocorrect import Speller
import cv2


img_name = "sampleImages/slide"
img_type = ".png"
img = cv2.imread(img_name+img_type)
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_grey, (3, 3), 0)
img_canny = cv2.Canny(img_blur, 100, 200)
img_dil = cv2.dilate(img_canny, (3, 3))
contours, hierarchy = cv2.findContours(img_dil, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
img_copy = img.copy()
# cv2.imshow("img", img)
# cv2.imshow("img_grey", img_grey)
# cv2.imshow("img_blur", img_blur)
cv2.imshow("img_canny", img_canny)
cv2.imshow("img_dil", img_dil)
cv2.drawContours(image=img_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
cv2.imshow("img_contour", img_copy)
cv2.waitKey(0)

reader = easyocr.Reader(['en'])  # this needs to run only once to load the model into memory
# Patrick Update with New Model


# Use OCR model to extract text
results = reader.readtext(img)


# Cleanup Function
def cleanup_text(text):
    # strip out non-ASCII text so we can draw the text on the image
    # using OpenCV
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()


# Duplicate image
cloneImg = img.copy()


# loop over the results and clean up text
for (bbox, text, prob) in results:
    # display the OCR text and associated probability
    print("[INFO] {:.4f}: {}".format(prob, text))
    # unpack the bounding box
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))
    tr = (int(tr[0]), int(tr[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))
    # cleanup the text and draw the box surrounding the text along
    # with the OCR text itself
    text = cleanup_text(text)
    cv2.rectangle(cloneImg, tl, br, (0, 255, 0), 2)
#	cv2.putText(cloneImg, text, (tl[0], tl[1] - 10),
#   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Print final output text
final_text = ""
for _, text, __ in results:  # _ = bounding box, text = text and __ = confident level
    final_text += " "
    final_text += text
print(final_text)

spell = Speller()

# Calculate image size
sz = img.shape

# Convert height and width to inches
H = 8.5*25.4/0.35
W = 11*25.4/0.35

# Scale image to PDF size
# If height is greater than pdf height scale down
if sz[1] > H:
    scale1 = H/sz[1]
else:
    scale1 = 1

# If width is greater than pdf width scale down
if sz[2] > W:
    scale2 = W/sz[2]
else:
    scale2 = 1

# Use larger scale factor
if scale1 < scale2:
    scale = scale1
elif scale2 < scale1:
    scale = scale2
else:
    scale = 1

# Store length of results
i = len(results)

# Define PDF using FPDF
pdf = FPDF(orientation='L', unit='pt')
pdf.add_page()

# Loop through results and save text to PDF
for x in range(i):
    pdf.set_xy(results[x][0][0][0]*scale, results[x][0][0][1]*scale)
    pdf.set_font("Arial", size=(results[x][0][2][1] - results[x][0][0][1]-3)*scale)
    pdf.cell((results[x][0][1][0] - results[x][0][0][0])*scale, (results[x][0][2][1] - results[x][0][0][1])*scale,
             txt=spell(results[x][1]))

# Save output PDF
pdf.output(img_name + '.pdf')
