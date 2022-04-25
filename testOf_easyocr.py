import easyocr
import argparse
import cv2
import numpy as np

reader = easyocr.Reader(['en'])  # this needs to run only once to load the model into memory

img = cv2.imread('sampleImages\dataset.png')
results = reader.readtext(img)


def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()


cloneImg = img.copy()
cv2.imshow('image', cloneImg)

# loop over the results
for (bbox, text, prob) in results:
	# display the OCR'd text and associated probability
	print("[INFO] {:.4f}: {}".format(prob, text))
	# unpack the bounding box
	(tl, tr, br, bl) = bbox
	tl = (int(tl[0]), int(tl[1]))
	tr = (int(tr[0]), int(tr[1]))
	br = (int(br[0]), int(br[1]))
	bl = (int(bl[0]), int(bl[1]))
	# cleanup the text and draw the box surrounding the text along
	# with the OCR'd text itself
	text = cleanup_text(text)
	cv2.rectangle(cloneImg, tl, br, (0, 255, 0), 2)


#	cv2.putText(cloneImg, text, (tl[0], tl[1] - 10),

#   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

final_text = ""
for _, text, __ in results: # _ = bounding box, text = text and __ = confident level
	final_text += " "
	final_text += text
print(final_text)

# show the output image
cv2.imshow('image', cloneImg)
cv2.waitKey(0)
