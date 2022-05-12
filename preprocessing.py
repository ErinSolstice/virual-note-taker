import math

import cv2
import os
import numpy as np


def stackImages(scale, img_array):
    rows = len(img_array)
    cols = len(img_array[0])
    rowsAvailable = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv2.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]),
                                                 None, scale, scale)
                if len(img_array[x][y].shape) == 2: img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None, scale, scale)
            if len(img_array[x].shape) == 2: img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    return ver


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


def get_warp(img: cv2.mat_wrapper.Mat, biggest, contour_scale_factor):
    biggest = reorder(biggest)
    h = img.shape[0]
    w = img.shape[1]
    pts1 = np.float32(biggest) / contour_scale_factor  # pts1[0][a][x] a={0-BL, 1-TL, 2-BR, 3-TR}; b={0-X, 1-Y}
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (w, h))
    return imgOutput

    # replace return with this to apply a slight crop
    # img_cropped = imgOutput[20:imgOutput.shape[0] - 20, 20:imgOutput.shape[1] - 20]
    # img_cropped = cv2.resize(img_cropped, (w, h))

    # img_cropped


def preprocess(img: cv2.mat_wrapper.Mat, target_width=500, blur_ksize=5, blur_sigma=2, canny_thresh1=54,
               canny_thresh2=75, dil_ero_ksize=5, dil_iter=2, ero_iter=0, wb_area=8000, contour_width=3,
               poly_approx_eps=0.02, show_all=False) -> cv2.mat_wrapper.Mat:

    if img is None:
        return None

    height = img.shape[0]
    width = img.shape[1]
    scale_factor = target_width / width
    target_height = math.floor(height * scale_factor)

    # make resized copy of img to process
    img_copy = img.copy()
    img_resize = cv2.resize(img_copy, (target_width, target_height))

    # convert to greyscale (maybe threshold)
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

    # blur
    img_blur = cv2.GaussianBlur(img_gray, (blur_ksize, blur_ksize), blur_sigma)

    # canny edge
    img_canny = cv2.Canny(img_blur, canny_thresh1, canny_thresh2)

    # dilate (and maybe erode)
    kernel = np.ones((dil_ero_ksize, dil_ero_ksize))
    img_dil = cv2.dilate(img_canny, kernel, iterations=dil_iter)
    img_ero = cv2.erode(img_dil, kernel, iterations=ero_iter)

    # get contours
    contours, hierarchy = cv2.findContours(img_ero, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # take the biggest rectangle of contours as the ROI
    biggest = np.array([])
    img_contour = img_resize.copy()
    # img_contour_orig = img.copy()
    img_contour_all = img_resize.copy()
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > wb_area:
            # cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, poly_approx_eps * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    cv2.drawContours(img_contour, biggest, -1, (255, 0, 0), 50)
    cv2.drawContours(img_contour_all, contours, -1, (255, 0, 0), contour_width)

    # deskew and crop original image to squared roi
    img_warp = img.copy()
    imageArray = ([img_resize, img_gray, img_blur, img_canny, img_dil, img_ero, img_contour, img_contour_all, img_warp])
    if biggest.size != 0:
        img_warp = get_warp(img, biggest, scale_factor)
        imageArray[-1] = img_warp
    else:
        print("Couldn't find whiteboard")

    if show_all:
        stackedImages = stackImages(0.6, imageArray)
        cv2.imshow("WorkFlow", stackedImages)

    # return warped and cropped img
    return img_warp


def process_img_file(path):
    return preprocess(cv2.imread(path))


if __name__ == "__main__":

    img_path = "sampleImages/m9k6y5tvum6x.jpg"
    tune_window = "Tuning"
    cv2.namedWindow(tune_window, cv2.WINDOW_NORMAL)
    img = cv2.imread(img_path)

    iw = 1
    ih = 1
    bk = 1
    bs = 1
    ct1 = 1
    ct2 = 1
    dek = 1
    di = 1
    ei = 1
    wba = 1
    clw = 1
    pae = 1

    def null(x):
        pass

    iw_name = 'image_width'
    cv2.createTrackbar(iw_name, tune_window, 500, 1000, null)

    bk_name = 'blur_ksize'
    cv2.createTrackbar(bk_name, tune_window, 5, 31, null)

    bs_name = 'blur_sig'
    cv2.createTrackbar(bs_name, tune_window, 2, 100, null)

    ct1_name = 'canny_thresh1'
    cv2.createTrackbar(ct1_name, tune_window, 54, 1000, null)

    ct2_name = 'canny_thresh2'
    cv2.createTrackbar(ct2_name, tune_window, 75, 1000, null)

    dek_name = 'dil/ero_ksize'
    cv2.createTrackbar(dek_name, tune_window, 5, 31, null)

    di_name = 'dil_iter'
    cv2.createTrackbar(di_name, tune_window, 2, 10, null)

    ei_name = 'ero_iter'
    cv2.createTrackbar(ei_name, tune_window, 0, 10, null)

    wba_name = 'wb_area'
    cv2.createTrackbar(wba_name, tune_window, 8000, 10000, null)

    clw_name = 'contour_line_width'
    cv2.createTrackbar(clw_name, tune_window, 2, 20, null)

    pae_name = 'poly_approx_eps * 1000'
    cv2.createTrackbar(pae_name, tune_window, 20, 100, null)

    while True:
        iw = cv2.getTrackbarPos(iw_name, tune_window)
        bk = cv2.getTrackbarPos(bk_name, tune_window)
        bk = bk - 1 if bk % 2 == 0 else bk
        bs = cv2.getTrackbarPos(bs_name, tune_window)
        ct1 = cv2.getTrackbarPos(ct1_name, tune_window)
        ct2 = cv2.getTrackbarPos(ct2_name, tune_window)
        dek = cv2.getTrackbarPos(dek_name, tune_window)
        dek = dek - 1 if dek % 2 == 0 else dek
        di = cv2.getTrackbarPos(di_name, tune_window)
        ei = cv2.getTrackbarPos(ei_name, tune_window)
        wba = cv2.getTrackbarPos(wba_name, tune_window)
        clw = cv2.getTrackbarPos(clw_name, tune_window)
        pae = cv2.getTrackbarPos(pae_name, tune_window) / 1000
        preprocess(img, iw, bk, bs, ct1, ct2, dek, di, ei, wba, clw, pae, True)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
