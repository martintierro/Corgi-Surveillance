from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
from matplotlib import pyplot as plt

#contour threshold
threshold_area = 500 

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(2,2))
parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

#Open Video Feed
capture = cv.VideoCapture()
capture.open("383.mp4")
if not capture.isOpened():
    print('Unable to open')
    exit(0)

i = 0

#Get video dimensions
frame_width = int(capture.get(3))
frame_height = int(capture.get(4))
fps = capture.get(cv.CAP_PROP_FPS)

#Save Video
out = out = cv.VideoWriter('output.avi', cv.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

while True:
    ret, frame = capture.read()

    
    if frame is None:
        break
    fgMask=0
    #Get foreground Mask
    fgMask = backSub.apply(frame, fgMask, 0.005)

    #Apply Median Blur
    blur = cv.medianBlur(fgMask, 13)


    #Morphological Transformations
    # opening = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel)
    # closing = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, kernel)
    # dilation = cv.dilate(fgMask,kernel,iterations = 1)
    # erosion = cv.erode(fgMask,kernel,iterations = 2)
    # closeerosion = cv.erode(closing,kernel,iterations = 2)
    # erosionclose = cv.morphologyEx(erosion, cv.MORPH_CLOSE, kernel)  
    # openclose = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)


    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    contours, hierarchy  = cv.findContours(blur, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv.contourArea(cnt)         
        if area > threshold_area:
            x,y,w,h = cv.boundingRect(cnt)
            frame = cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    cv.imshow("Median Blur", blur)
    # cv.imshow('Erosion', erosion)
    # cv.imshow('Opening', opening)
    # cv.imshow('Closing', closing)
    # cv.imshow('Dilation', dilation)
    # cv.imshow('Opening and Closing', openclose)
    # cv.imshow('Erosion then Closing', erosionclose)
    # cv.imshow('Closing then Erosion', closeerosion)
    # cv.imshow("Closing Blur", closingblur)



    #Subtracts the mask overlap region from the image overlap region, puts it in image_sub
    colored_mask = cv.bitwise_and(frame,frame,mask = blur)

    frame_sub = frame-colored_mask

    # Shows diff only:
    cv.imshow('image_sub', frame_sub)

    # Write the frame into the file 'output.avi'
    out.write(frame)
 

    #Exports Frames
    # cv.imwrite('../Frames/Frame '+str(i)+'.jpg',frame)
    # cv.imwrite('../Mask/BG Mask '+str(i)+'.jpg',fgMask)
    # cv.imwrite('../Background/Background '+str(i)+'.jpg',frame_sub)
    i=i+1
    
   

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

# When everything done, release the video capture and video write objects
capture.release()
out.release()

 
# Closes all the frames
cv.destroyAllWindows() 