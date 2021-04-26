import os
import natsort
from os import listdir
from os.path import isfile, join
import cv2 as cv
import numpy as np

scale = 2

def super_resolution(filename, video_name):
    frame_buffer = 10 - 1
    
    bg_path = "Background/Median Blur/" + video_name +"/"
    fg_path = "Foreground/Median Blur/" + video_name +"/"
    
    bg_files = [f for f in listdir(bg_path) if isfile(join(bg_path, f))]
    fg_files = [f for f in listdir(fg_path) if isfile(join(fg_path, f))]

    sorted_bg_files = natsort.natsorted(bg_files)
    sorted_fg_files = natsort.natsorted(fg_files)

    for i in range(len(sorted_bg_files)-frame_buffer):
        print("Run: " + str(i))
        print("Reference: " + sorted_bg_files[i])
        reference = cv.imread(bg_path + sorted_bg_files[i])
        foreground = cv.imread(fg_path + sorted_fg_files[i + frame_buffer])
        foreground = perform_interpolation(foreground, scale, cv.INTER_LINEAR)
        lr_images = []

        for j in range(frame_buffer):
            lr_images.append(cv.imread(bg_path + sorted_bg_files[i+j+1]))

        result = mean_fusion(reference, lr_images)
        
        result = np.float32(result)
        foreground = np.float32(foreground)
        result = cv.add(result, foreground)

        result = result.astype(np.uint8)

        cv.imwrite("Super Resolution/" + video_name + "/sr_" + str(i) + ".png", result)

def mean_fusion(reference, lr_images):
    initialMat = reference
    sumMat = perform_interpolation(initialMat, scale, cv.INTER_LINEAR)
    sumMat = np.float32(sumMat)

    for initialMat in lr_images:
        initialMat = perform_interpolation(initialMat, scale, cv.INTER_CUBIC)
        initialMat= np.float32(initialMat)
        maskMat = produce_mask(initialMat)
        sumMat = cv.add(sumMat, initialMat, mask=maskMat)


    sumMat = cv.divide(sumMat, (10, 10, 10, 10))

    outputMat = sumMat.astype(np.uint8)
    return outputMat

def perform_interpolation(fromMat, scale, interpolationType):
    newRows = round(np.shape(fromMat)[1] * scale)
    newCols = round(np.shape(fromMat)[0] * scale)

    shape = [newRows, newCols]
    hrMat = np.zeros(shape, dtype=fromMat.dtype)

    hrMat = cv.resize(fromMat, (newRows, newCols), scale, scale, interpolationType)

    return hrMat

def produce_mask(inputMat):
    dstMask = np.empty(inputMat.shape)
    np.copyto(dstMask, inputMat)

    img_float32 = np.float32(dstMask)
    dstMask = cv.cvtColor(img_float32, cv.COLOR_BGR2GRAY)
    dstMask = dstMask.astype(np.uint8)
    dstMask = cv.threshold(dstMask, 1, 1, cv.THRESH_BINARY)[1]
    return dstMask
