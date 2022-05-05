import cv2
import os


def load_images_from_folder(folder: str) -> list[cv2.mat_wrapper.Mat]:
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


def preprocess(img: cv2.mat_wrapper.Mat) -> cv2.mat_wrapper.Mat:
    if img is None: return None

    # make copy of img to process
    img_copy = img.copy()

    # convert to greyscale (maybe threshold)
    cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY, img_copy)

    # blur
    cv2.GaussianBlur(img_copy, (3, 3), 1, img_copy, 1)

    # canny edge
    cv2.Canny(img_copy, 1, 1, cv2.RETR_EXTERNAL)

    # dilate (and maybe erode)
    cv2.dilate(img_copy, (3,3), img_copy)

    # take the biggest rectangle of contours as the ROI
    # deskew and crop original image to squared roi
    # return original img
    return img


def process_img_file(path):
    return preprocess(cv2.imread(path))

