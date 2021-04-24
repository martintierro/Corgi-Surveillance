from __future__ import print_function

import cv2 as cv
import numpy as np
import argparse
from matplotlib import pyplot as plt

def background_subtraction(filename, video_name):
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


    capture = cv.VideoCapture()
    # capture.open("bike.mp4")
    capture.open(filename)


    if not capture.isOpened():
        print('Unable to open')
        exit(0)

    i = 0

    #Get video dimensions
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
    fps = capture.get(cv.CAP_PROP_FPS)

    #Save Video
    out = out = cv.VideoWriter('BoundingBoxesVideo/'+video_name+".mp4", cv.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

    while True:
        ret, frame = capture.read()

        if frame is None:
            break
        fgMask=0
        #Get foreground Mask
        fgMask = backSub.apply(frame, fgMask, 0.005)

        #Apply Median Blur
        blur = cv.medianBlur(fgMask, 5)
        
        cv.imshow('Frame', frame)
        cv.imwrite('Frames/'+video_name+'/Frame '+str(i)+'.jpg',frame)
        

        cv.imshow('FG Mask', fgMask)
        cv.imshow("Median Blur", blur)


        #Subtracts the mask overlap region from the image overlap region, puts it in image_sub
        colored_mask = cv.bitwise_and(frame,frame,mask = fgMask)
        colored_mask_blur = cv.bitwise_and(frame,frame,mask = blur)

        frame_bg = frame-colored_mask
        frame_bg_blur = frame-colored_mask_blur

        frame_fg = frame - frame_bg
        frame_fg_blur = frame - frame_bg_blur
        # Shows diff only:
        cv.imshow('image_sub', frame_bg)
        cv.imshow('image_fore', frame_fg)
        # Write the frame into the file 'output.avi'

        contours, hierarchy  = cv.findContours(blur, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        for cnt in contours:
            area = cv.contourArea(cnt)         
            if area > threshold_area:
                x,y,w,h = cv.boundingRect(cnt)
                frame_box = cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        out.write(frame)
        
        cv.imshow('Bounding Box', frame_box)
        

        #Exports Frames
        cv.imwrite('BoundingBox/'+video_name+'/Box '+str(i)+'.jpg',frame)
        cv.imwrite('Mask/Raw BG/'+video_name+'/BG Mask '+str(i)+'.jpg',fgMask)
        cv.imwrite('Mask/Median Blur/'+video_name+'/BG Mask '+str(i)+'.jpg',blur)
        cv.imwrite('Background/Raw BG/'+video_name+'/Background '+str(i)+'.jpg',frame_bg)
        cv.imwrite('Background/Median Blur/'+video_name+'/Background '+str(i)+'.jpg',frame_bg_blur)
        cv.imwrite('Foreground/Raw BG/'+video_name+'/Foreground '+str(i)+'.jpg',frame_bg)
        cv.imwrite('Foreground/Median Blur/'+video_name+'/Foreground '+str(i)+'.jpg',frame_bg_blur)

        i=i+1   

        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

    # When everything done, release the video capture and video write objects
    capture.release()
    out.release()

    
    # Closes all the frames
    cv.destroyAllWindows() 