import cv2 as cv

def sharpen(image):
    blurred_img = cv.blur(image, (25, 25))

    output_img = cv.addWeighted(image, 2.25, blurred_img, -1.25, 0, cv.CV_8UC3)

    return output_img
