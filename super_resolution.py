import os
import natsort
from os import listdir
from os.path import isfile, join
import cv2 as cv
import numpy as np
from image_sharpener import sharpen
from lr_warping import lr_warping

scale = 2
frame_buffer = 10 - 1

def super_resolution(filename, video_name):

    capture = cv.VideoCapture()
    capture.open(filename)

    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
    fps = capture.get(cv.CAP_PROP_FPS)

    out = cv.VideoWriter('Super Resolution/'+video_name+".mp4", cv.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width * scale,frame_height * scale))

    capture.release()
    
    bg_path = "Background/" + video_name +"/"
    # bg_cubic_path = "Background/Cubic Interpolation/" + video_name +"/"
    fg_path = "Foreground/" + video_name +"/"
    mask_path = "Mask/" + video_name +"/"
    
    bg_files = [f for f in listdir(bg_path) if isfile(join(bg_path, f))]
    # bg_cubic_files = [f for f in listdir(bg_cubic_path) if isfile(join(bg_cubic_path, f))]
    fg_files = [f for f in listdir(fg_path) if isfile(join(fg_path, f))]
    mask_files = [f for f in listdir(mask_path) if isfile(join(mask_path, f))]

    sorted_bg_files = natsort.natsorted(bg_files)
    # sorted_bg_cubic_files = natsort.natsorted(bg_cubic_files)
    sorted_fg_files = natsort.natsorted(fg_files)
    sorted_mask_files = natsort.natsorted(mask_files)

    lr_images = None
    fg_mask = None
    reference = None
    foreground = None
    result = None

    for i in range(len(sorted_bg_files)-frame_buffer):
        print("Super Resolution Frame Number: " + str(i))
        # print("Reference: " + background_frames[i])
        reference = cv.imread(bg_path + sorted_bg_files[i])
        foreground = cv.imread(fg_path + sorted_fg_files[i + frame_buffer])
        fg_mask = cv.imread(mask_path + sorted_mask_files[i + frame_buffer])

        lr_images = []

        for j in range(frame_buffer):
            lr_image = cv.imread(bg_path + sorted_bg_files[i+j+1])
            # lr_image = sharpen(lr_image)
            lr_images.append(lr_image)

        # lr_warping(reference, lr_images, video_name)
        # lr_images = []

        # for j in range(frame_buffer):
        #     lr_image = cv.imread("Temp/" + video_name + "_warped_" + str(j) + ".png")
        #     lr_images.append(lr_image)


        result = mean_fusion(reference, lr_images)
        
        result = combine_foreground(result, foreground, fg_mask)

        # result = cv.detailEnhance(result, sigma_s=1, sigma_r=0.05)

        result = result.astype(np.uint8)

        cv.imwrite("Super Resolution/" + video_name + "/sr_" + str(i) + ".png", result)
        out.write(result)
    
    out.release()

def mean_fusion(reference, lr_images):
    initialMat = reference
    sumMat = np.zeros(initialMat.shape)
    np.copyto(sumMat, initialMat)
    sumMat = np.float32(sumMat)

    for initialMat in lr_images:
        initialMat= np.float32(initialMat)
        maskMat = produce_mask(initialMat)
        sumMat = cv.add(sumMat, initialMat, mask=maskMat)


    sumMat = cv.divide(sumMat, (frame_buffer + 1, frame_buffer + 1, frame_buffer + 1, frame_buffer + 1))

    outputMat = sumMat.astype(np.uint8)
    return outputMat


def produce_mask(inputMat):
    dstMask = np.empty(inputMat.shape)
    np.copyto(dstMask, inputMat)

    img_float32 = np.float32(dstMask)
    dstMask = cv.cvtColor(img_float32, cv.COLOR_BGR2GRAY)
    dstMask = dstMask.astype(np.uint8)
    dstMask = cv.threshold(dstMask, 1, 1, cv.THRESH_BINARY)[1]
    return dstMask

def combine_foreground(background, foreground, alpha):
    foreground = np.float32(foreground) / 255.
    background = np.float32(background) / 255.

    alpha = np.float32(alpha) / 255.

    foreground = cv.multiply(alpha, foreground)

    background = cv.multiply(1.0 - alpha, background)

    result = cv.add(foreground, background) * 255

    return result 
