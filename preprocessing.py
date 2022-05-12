import cv2
import os
import numpy as np


def load_images_from_folder(folder: str) -> list[cv2.mat_wrapper.Mat]:
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def reorder(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), np.int32)
    add = points.sum(1)
    # print("add", add)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    # print("NewPoints",new_points)
    return new_points


def get_warp(img: cv2.mat_wrapper.Mat, biggest):
    biggest = reorder(biggest)
    h = img.shape[0]
    w = img.shape[1]
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (w, h))

    img_cropped = imgOutput[20:imgOutput.shape[0] - 20, 20:imgOutput.shape[1] - 20]
    img_cropped = cv2.resize(img_cropped, (w, h))

    return img_cropped


def preprocess(img: cv2.mat_wrapper.Mat) -> cv2.mat_wrapper.Mat:
    if img is None:
        return None

    # make copy of img to process
    img_copy = img.copy()

    # convert to greyscale (maybe threshold)
    cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY, img_copy)

    # blur
    cv2.GaussianBlur(img_copy, (3, 3), 1, img_copy, 1)

    # canny edge
    cv2.Canny(img_copy, 100, 200)

    # dilate (and maybe erode)
    cv2.dilate(img_copy, (3, 3), img_copy)

    # get contours
    contours, hierarchy = cv2.findContours(img_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # take the biggest rectangle of contours as the ROI
    biggest = np.array([])
    img_contour = img.copy()
    max_area = 0
    contours, hierarchy = cv2.findContours(img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            # cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    cv2.drawContours(img_contour, biggest, -1, (255, 0, 0), 20)

    # deskew and crop original image to squared roi
    if biggest.size != 0:
        img_copy = get_warp(img, biggest)
        # imageArray = ([img,imgThres],
        #           [imgContour,img_warped])
        imageArray = ([img_contour, img_copy])
        cv2.imshow("ImageWarped", img_copy)
    else:
        # imageArray = ([img, imgThres],
        #               [img, img])
        imageArray = ([img_contour, img])

    # stackedImages = stackImages(0.6, imageArray)
    # cv2.imshow("WorkFlow", stackedImages)

    # return warped and cropped img
    return img_copy


def process_img_file(path):
    return preprocess(cv2.imread(path))

if(__name__ == "__main__"):
    img_path = "sampleImages/whiteboard.png"
    process_img_file(img_path)
