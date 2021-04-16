import sys
import math
import cv2 as cv
import numpy as np
from imutils import paths
from matplotlib import pyplot as plt
import argparse


def blur_scoring(image):
    return cv.Laplacian(image, cv.CV_64F).var()


def otsu_binarization(images):
    linked_values = list()
    for count, imagePath in enumerate(images):
        img = cv.imread(imagePath, 0)
        blur = cv.GaussianBlur(img, (5, 5), 0)

        # find normalized_histogram, and its cumulative distribution function
        hist = cv.calcHist([blur], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.max()
        Q = hist_norm.cumsum()

        bins = np.arange(256)

        fn_min = np.inf
        thresh = -1

        for i in range(1, 256):
            p1, p2 = np.hsplit(hist_norm, [i])  # probabilities
            q1, q2 = Q[i], Q[255] - Q[i]  # cum sum of classes
            b1, b2 = np.hsplit(bins, [i])  # weights

            # finding means and variances
            m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
            v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(((b2 - m2) ** 2) * p2) / q2

            # calculates the minimization function
            fn = v1 * q1 + v2 * q2
            if fn < fn_min:
                fn_min = fn
                thresh = i

        # find otsu's threshold value with OpenCV function
        ret, otsu = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        linked_values.append((thresh, ret))
    return linked_values


def save_image(image_list):                     # image_list = (name/path, img file)
    good_saves = 0
    bad_saves = 0
    for image_tuples in image_list:
        for name, image in image_tuples:
            check = cv.imwrite(name, image)
            if check:
                good_saves += 1
            else:
                print(f"{name} failed to save")
                bad_saves += 1
    print(f"Proper saves: {good_saves}\nFailed saved: {bad_saves}")


def hough_line_drawing(images):
    image_files_to_save = list()
    for count, imgPath in enumerate(images):
        image = cv.imread(imgPath)
        image2 = np.copy(image)
        print(imgPath)
        # gray = cv.imread(cv.samples.findFile(imgPath), cv.IMREAD_GRAYSCALE)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        print("gray")
        blur = cv.medianBlur(gray, 5)
        print("blur")
        adapt_type = cv.ADAPTIVE_THRESH_GAUSSIAN_C
        thresh_type = cv.THRESH_BINARY_INV
        threshold = cv.adaptiveThreshold(blur, 255, adapt_type, thresh_type, 41, 20)
        # threshold, image_result = cv.threshold(blur, 255, 0, cv.THRESH_BINARY + cv.THRESH_OTSU)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        fm = cv.filter2D(threshold, -1, sharpen_kernel)
        print("adaptiveThreshold")
        # fm = variance_of_laplacian(gray)                                        ### Used to calculate the blurriness
        # fm = cv.Canny(image, 50, 200)                                           ### Canny looks promising but it restricts too much, maybe play with thresh?
        rho, theta, thresh = 2, np.pi / 180, 400
        lines = cv.HoughLines(fm, rho, theta, thresh)
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv.line(image2, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
        rho, theta, thresh = 2, np.pi / 180, 400
        lines_p = cv.HoughLinesP(fm, 1, np.pi / 180, 50, None, 50, 10)
        if lines_p is not None:
            for i in range(0, len(lines_p)):
                l = lines_p[i][0]
                cv.line(image, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)
        name = (f"Probability Hough Line - Standard Preprocessing - {imgPath}", image)
        name2 = (f"Hough Line - Standard Preprocessing - {imgPath}", image2)
        name3 = (f"B&W Image - Standard Preprocessing - {imgPath}", fm)
        image_files_to_save.append((name, name2, name3))
        cv.imshow("Probability Hough Line Standard Preprocessing", image)
        cv.putText(fm, f"{fm}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        cv.imshow("Hough Line with Standard Preprocessing", image2)
        cv.imshow("B&W Image", fm)
        key = cv.waitKey(0)
    return image_files_to_save


def main():
    # This is going to process the image and set it to output purely the edges, plan is to run hough line detection on this
    # Which will maybe fix the issue with to many lines being represented

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True, help="path to input directory of images")
    args = vars(ap.parse_args())
    images = paths.list_images(args["images"])
    # thresh_vals = otsu_binarization(images)
    to_save = hough_line_drawing(images)
    save_image(to_save)


if __name__ == "__main__":
    main()
