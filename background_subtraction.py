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
    print(fps)

    #Save Video
    out = cv.VideoWriter('BoundingBoxesVideo/'+video_name+".mp4", cv.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

    bg_plate = cv.imread("Background/" + video_name + ".png")

    #Variable declarations
    mb_backgrounds = []
    mb_foregrounds = []
    mb_fg_masks = []
    fgMask = None
    blur = None
    kernel = np.ones((3,3),np.uint8)
    frame = None
    ret = None
    colored_mask = None
    colored_mask_blur = None
    colored_bg_mask = None
    colored_bg_mask_blur = None
    frame_bg = None
    frame_bg_blur = None
    frame_bg_plate = None
    frame_bg_plate_blur = None
    frame_fg = None
    frame_fg_blur = None
    contours = None
    hierarchy = None
    
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        frame = np.float32(frame)
        bg_plate = np.float32(bg_plate)
        fgMask=0.0
        # Get foreground Mask
        fgMask = backSub.apply(frame, fgMask, 0.005)

        #Apply Median Blur
        blur = cv.medianBlur(fgMask, 5)
 
        # blur = cv.morphologyEx(blur,cv.MORPH_CLOSE, kernel)
        blur = cv.morphologyEx(blur,cv.MORPH_OPEN, kernel)
        
        # Save Frame
        frame = frame.astype(np.uint8)
        cv.imwrite('Frames/'+video_name+'/Frame '+str(i)+'.jpg',frame)
        frame = np.float32(frame)
        

        #Subtracts the mask overlap region from the image overlap region, puts it in image_sub
        colored_mask = cv.bitwise_and(frame,frame,mask = fgMask)
        colored_mask_blur = cv.bitwise_and(frame,frame,mask = blur)
        colored_bg_mask = cv.bitwise_and(bg_plate, bg_plate, mask = fgMask)
        colored_bg_mask_blur = cv.bitwise_and(bg_plate, bg_plate, mask = blur)

        colored_mask = np.float32(colored_mask)
        colored_mask_blur = np.float32(colored_mask_blur)
        colored_bg_mask = np.float32(colored_bg_mask)
        colored_bg_mask_blur = np.float32(colored_bg_mask_blur)



        frame_bg = frame-colored_mask
        frame_bg_blur = frame-colored_mask_blur
        
        frame_bg_plate = bg_plate - (bg_plate - colored_bg_mask)
        frame_bg_plate_blur = bg_plate - (bg_plate - colored_bg_mask_blur)

        frame_fg = frame - frame_bg
        frame_fg_blur = frame - frame_bg_blur

        frame_bg = cv.add(frame_bg, frame_bg_plate)
        frame_bg_blur = cv.add(frame_bg_blur, frame_bg_plate_blur)
        # Shows diff only:
        # cv.imshow('image_background', frame_bg)
        # cv.imshow('image_foreground', frame_fg)

        contours, hierarchy  = cv.findContours(blur, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        for cnt in contours:
            area = cv.contourArea(cnt)         
            if area > threshold_area:
                x,y,w,h = cv.boundingRect(cnt)
                frame_box = cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        frame = frame.astype(np.uint8)
        out.write(frame)
        
        fgMask = fgMask.astype(np.uint8)
        blur = blur.astype(np.uint8)
        frame_bg = frame_bg.astype(np.uint8)
        frame_bg_blur = frame_bg_blur.astype(np.uint8)
        frame_fg = frame_fg.astype(np.uint8)
        frame_fg_blur = frame_fg_blur.astype(np.uint8)

        mb_backgrounds.append(frame_bg_blur)
        mb_foregrounds.append(frame_fg_blur)
        mb_fg_masks.append(fgMask)

        #Exports Frames
        cv.imwrite('BoundingBox/'+video_name+'/Box '+str(i)+'.jpg',frame)
        cv.imwrite('Mask/Raw BG/'+video_name+'/BG Mask '+str(i)+'.jpg',fgMask)
        cv.imwrite('Mask/Median Blur/'+video_name+'/BG Mask '+str(i)+'.jpg',blur)
        cv.imwrite('Background/Raw BG/'+video_name+'/Background '+str(i)+'.jpg',frame_bg)
        cv.imwrite('Background/Median Blur/'+video_name+'/Background '+str(i)+'.jpg',frame_bg_blur)
        cv.imwrite('Foreground/Raw BG/'+video_name+'/Foreground '+str(i)+'.jpg',frame_fg)
        cv.imwrite('Foreground/Median Blur/'+video_name+'/Foreground '+str(i)+'.jpg',frame_fg_blur)

        i=i+1   

        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

    # When everything done, release the video capture and video write objects
    capture.release()
    out.release()

    return mb_backgrounds, mb_foregrounds, mb_fg_masks

    
    # Closes all the frames
    cv.destroyAllWindows() 