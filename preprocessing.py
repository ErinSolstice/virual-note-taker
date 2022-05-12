import cv2
import os
import numpy as np


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
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


def get_warp(img: cv2.mat_wrapper.Mat, biggest, h, w):
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


def preprocess(img: cv2.mat_wrapper.Mat, target_height=500, target_width=500, blur_ksize=5, blur_sigma=2, canny_thresh1=100, canny_thresh2=0, dil_ero_ksize=3, wb_area=5000, contour_width=3, show_all=False) -> cv2.mat_wrapper.Mat:
    if img is None:
        return None

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
    img_dil = cv2.dilate(img_canny, kernel, iterations=2)
    img_ero = cv2.erode(img_dil, kernel, iterations=1)

    # convert to 8 bit
    #img = img_ero
    #cv2.convertScaleAbs(old_img_copy, img_copy)
    #print(type(old_img_copy), type(img_copy), old_img_copy.shape, img_copy.shape)

    # get contours
    contours, hierarchy = cv2.findContours(img_ero, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # all_contours, all_hierarchy = cv2.findContours(img_ero, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # take the biggest rectangle of contours as the ROI
    biggest = np.array([])
    img_contour = img_resize.copy()
    img_contour_all = img_resize.copy()
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > wb_area:
            # cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    cv2.drawContours(img_contour, biggest, -1, (255, 0, 0), 50)
    cv2.drawContours(img_contour_all, contours, -1, (255, 0, 0), contour_width)

    # deskew and crop original image to squared roi
    img_warp = img_resize.copy()
    imageArray = ([img_resize, img_gray, img_blur, img_canny, img_dil, img_ero, img_contour, img_contour_all, img_warp])
    if biggest.size != 0:
        img_warp = get_warp(img_resize, biggest, target_height, target_width)
        imageArray[-1] = img_warp
        # imageArray.append(img_warp)
        # imageArray = ([img,imgThres],
        #           [imgContour,img_warped])
        #print(imageArray[0])
        #print("warp", img_warp.shape)
        #imageArray[0] += img_warp
        # cv2.imshow("ImageWarped", img_warp)
        #cv2.waitKey(0)
    else:
        print("Couldn't find whiteboard")
        # imageArray = ([img, imgThres],
        #               [img, img])
        #imageArray[0] += [img_contour, img]

    stackedImages = stackImages(0.6, imageArray)
    if show_all:
        cv2.imshow("WorkFlow", stackedImages)
    #cv2.waitKey(0)

    # return warped and cropped img
    return img_copy


def process_img_file(path):
    return preprocess(cv2.imread(path))


if __name__ == "__main__":
    img_path = "sampleImages/whiteboard.png"
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
    wba = 1
    clw = 1

    def null(x):
        pass

    def trackbar_callback():
        print("update")
        iw = cv2.getTrackbarPos(iw_name, tune_window)
        ih = cv2.getTrackbarPos(ih_name, tune_window)
        bk = cv2.getTrackbarPos(bk_name, tune_window)
        bk = bk - 1 if bk % 2 == 0 else bk
        print("bk", bk)
        bs = cv2.getTrackbarPos(bs_name, tune_window)
        ct1 = cv2.getTrackbarPos(ct1_name, tune_window)
        ct2 = cv2.getTrackbarPos(ct2_name, tune_window)
        dek = cv2.getTrackbarPos(dek_name, tune_window)
        dek = dek - 1 if dek % 2 == 0 else dek
        wba = cv2.getTrackbarPos(wba_name, tune_window)
        clw = cv2.getTrackbarPos(clw_name, tune_window)
        preprocess(img, iw, ih, bk, bs, ct1, ct2, dek, wba, clw, True)

    iw_name = 'image_width'
    cv2.createTrackbar(iw_name, tune_window, 500, 1000, null)

    ih_name = 'image_height'
    cv2.createTrackbar(ih_name, tune_window, 500, 1000, null)

    bk_name = 'blur_ksize'
    cv2.createTrackbar(bk_name, tune_window, 3, 31, null)

    bs_name = 'blur_sig'
    cv2.createTrackbar(bs_name, tune_window, 2, 100, null)

    ct1_name = 'canny_thresh1'
    cv2.createTrackbar(ct1_name, tune_window, 100, 1000, null)

    ct2_name = 'canny_thresh2'
    cv2.createTrackbar(ct2_name, tune_window, 1, 1000, null)

    dek_name = 'dil/ero_ksize'
    cv2.createTrackbar(dek_name, tune_window, 3, 31, null)

    wba_name = 'wb_area'
    cv2.createTrackbar(wba_name, tune_window, 1, 10000, null)

    clw_name = 'contour_line_width'
    cv2.createTrackbar(clw_name, tune_window, 2, 20, null)

    trackbar_callback()
    while True:
        print("update")
        iw = cv2.getTrackbarPos(iw_name, tune_window)
        ih = cv2.getTrackbarPos(ih_name, tune_window)
        bk = cv2.getTrackbarPos(bk_name, tune_window)
        bk = bk - 1 if bk % 2 == 0 else bk
        bs = cv2.getTrackbarPos(bs_name, tune_window)
        ct1 = cv2.getTrackbarPos(ct1_name, tune_window)
        ct2 = cv2.getTrackbarPos(ct2_name, tune_window)
        dek = cv2.getTrackbarPos(dek_name, tune_window)
        dek = dek - 1 if dek % 2 == 0 else dek
        wba = cv2.getTrackbarPos(wba_name, tune_window)
        clw = cv2.getTrackbarPos(clw_name, tune_window)
        preprocess(img, iw, ih, bk, bs, ct1, ct2, dek, wba, clw)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
