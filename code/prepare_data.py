import sys
import math
import cv2 as cv
import numpy as np
import os
from imutils import paths
from matplotlib import pyplot as plt
import argparse


points = list()


def blur_scoring(image):
    blur_val = cv.Laplacian(image, cv.CV_64F).var()
    print(f"Blur Value: {blur_val}")
    return blur_val


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv.INTER_AREA):  #  https://stackoverflow.com/questions/35180764/opencv-python-image-too-big-to-display
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=inter)


def click_event(event, x, y, flags, params):  # https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
        print(f"{x} {y}")
        points.append([x, y])


def manual_segmentation(imag, name, status, folder=None):
    # original image
    # -1 loads as-is so if it will be 3 or 4 channel as the original
    # mask defaulting to black for 3-channel and transparent for 4-channel
    # (of course replace corners with yours)
    b_channel, g_channel, r_channel = cv.split(imag)

    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # creating a dummy alpha channel image.

    img_BGRA = cv.merge((b_channel, g_channel, r_channel, alpha_channel))
    mask = np.zeros(img_BGRA.shape, dtype=np.uint8)
    # print(mask)
    # roi_corners = np.array([[(10, 10), (300, 300), (10, 300)]], dtype=np.int32)
    roi_corners = np.array([points], dtype=np.int32)
    # print(roi_corners)
    # fill the ROI so it doesn't get wiped out when the mask is applied

    print(len(img_BGRA.shape))
    channel_count = img_BGRA.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,) * channel_count
    print(ignore_mask_color)
    cv.fillPoly(mask, roi_corners, ignore_mask_color)
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex

    # apply the mask
    masked_image = cv.bitwise_and(img_BGRA, mask)
    # cv.imshow('masked image', masked_image)
    val = blur_scoring(masked_image)
    if status == 0:
        print("Above Water")
        filename = "above-" + str(val) + "-" + str(name)
    else:
        print("Below Water")
        filename = "below-" + str(val) + "-" + str(name)
    if folder is not None:
        filename = os.path.join(folder, filename)
    cv.imwrite(filename, masked_image)


def main():
    # This is going to process the image and set it to output purely the edges, plan is to run hough line detection on this
    # Which will maybe fix the issue with to many lines being represented
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--images", required=True, help="path to input directory of images")
    # args = vars(ap.parse_args())
    for i in sys.argv[1:]:
        folder_name = input("New Folder Name:\n")
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        images = paths.list_images(i)
        print(i)
        for image in images:
            above_water = cv.imread(image, -1)
            above_water = resize_with_aspect_ratio(above_water, height=720)
            below_water = np.copy(above_water)
            sections = [above_water, below_water]
            for c, s in enumerate(sections):
                print("Select Your Points!")
                cv.imshow('image', s)
                cv.setMouseCallback('image', click_event)
                cv.waitKey(0)
                manual_segmentation(s, os.path.basename(image), c, folder_name)
                points.clear()

            cv.destroyAllWindows()


if __name__ == "__main__":
    main()

