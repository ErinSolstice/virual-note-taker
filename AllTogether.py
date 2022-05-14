# from matplotlib import pyplot as plt
import cv2
import easyocr
import numpy as np
from fpdf import FPDF
from autocorrect import Speller
import preprocessing
import argparse
from pathlib import Path

# Create the parser
my_parser = argparse.ArgumentParser(description='perform ocr on image')

# Add the arguments
my_parser.add_argument('--path',
                       action='store',
                       type=str,
                       required=False,
                       default='sampleImages/slide.png',
                       help='the path to image')

# Execute the parse_args() method
args = my_parser.parse_args()

img_path = args.path

# Change img_name to image path without the file type extension
img_name = Path(img_path).stem

# read image into memory
img = cv2.imread(img_path)
# copy image
img_copy = img.copy()
# run preprocessing on the img
prepro_img = preprocessing.preprocess(img)

# Uncomment to display original and processed images
cv2.imshow('img', img)
cv2.imshow('prepro_img', prepro_img)
cv2.waitKey(0)

# Comment out all the readers except the one with the desired recog_network
# this needs to run only once to load the model into memory
# base model
reader = easyocr.Reader(['en'])
# model trained on IAM-onDB dataset
# reader = easyocr.Reader(['en'], recog_network='wb_v2')
# further training of base model using IAM-onDB dataset and synthetic data generated with TextRecognitionDataGenerator
# reader = easyocr.Reader(['en'], recog_network='g2_wb_bel_v2')

# Use OCR model to extract text
# Doesn't use preprocessing
# results = reader.readtext(img_copy)
# Uses preprocessing
results = reader.readtext(prepro_img)

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

# Uncomment to show image wiht bounding boxes
# cv2.imshow("cloneImg", cloneImg)
# cv2.waitKey(0)

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
    pdf.set_font("Arial", size=16)  # Uncomment for uniform text
    # (results[x][0][2][1] - results[x][0][0][1]-3)*scale*0.8)  # Uncomment for scaled text
    pdf.cell((results[x][0][1][0] - results[x][0][0][0])*scale, (results[x][0][2][1] - results[x][0][0][1])*scale,
             # txt=results[x][1])  # Uncomment for no spell check
             txt=spell(results[x][1]))  # Uncomment for spell check

# Save output PDF
pdf.output(f"sampleResults/{img_name}.pdf")
