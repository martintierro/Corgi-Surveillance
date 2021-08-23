import cv2 as cv
import numpy as np
import argparse
from matplotlib import pyplot as plt

scale = 2

model_path = "models/ESPCN_x2.pb"
sr = cv.dnn_superres.DnnSuperResImpl_create()

def background_subtraction(filename, video_name):
    #contour threshold
    threshold_area = 500 

    sr.readModel(model_path)
    sr.setModel("espcn", scale)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(2,2))
    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                                OpenCV. You can process both videos and images.')
    parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
    parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
    args = parser.parse_args()
    
    # if args.algo == 'MOG2':
    #     backSub = cv.createBackgroundSubtractorMOG2()
    # else:
    #     backSub = cv.createBackgroundSubtractorKNN()

    # backSub = cv.createBackgroundSubtractorMOG2()
    backSub = cv.bgsegm.createBackgroundSubtractorGMG()

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
    box_out = cv.VideoWriter('BoundingBox/'+video_name+".mp4", cv.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width * scale,frame_height * scale))
    mask_out = cv.VideoWriter('Mask/'+video_name+".mp4", cv.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width * scale,frame_height * scale))

    bg_plate = cv.imread("Background/" + video_name + ".png")
    # bg_plate = perform_interpolation(bg_plate, scale, cv.INTER_CUBIC)
    bg_plate = sr.upsample(bg_plate)

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
    fgMask = None

    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT)) 

    
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        print("BG Subtraction Frame Number: " + str(i))
        frame = np.float32(frame)
        bg_plate = np.float32(bg_plate)
    
        # Get foreground Mask

        fgMask = cv.imread("E:/Projects/Thesis/Datasets with Generated BG/Masks/" + video_name + "/hr_" + str(i) +".png", cv.IMREAD_GRAYSCALE)
        thresh = 10
        fgMask = cv.threshold(fgMask, thresh, 255, cv.THRESH_BINARY)[1]
        #Apply Median Blur
        # fgMask = perform_interpolation_mask(fgMask, frame, scale, cv.INTER_NEAREST)
        # cv.imshow("FG Mask", fgMask)
        vid_mask = cv.cvtColor(fgMask, cv.COLOR_GRAY2RGB)
        mask_out.write(vid_mask)
        
        frame = sr.upsample(frame)
        
        frame_fg, frame_bg = perform_subtraction(frame, bg_plate, fgMask)

        contours, hierarchy  = cv.findContours(blur, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        for cnt in contours:
            area = cv.contourArea(cnt)         
            if area > threshold_area:
                x,y,w,h = cv.boundingRect(cnt)
                frame_box = cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        frame = frame.astype(np.uint8)
        box_out.write(frame)

        # cv.imshow("Bounding Box", frame)
        
        fgMask = fgMask.astype(np.uint8)
        frame_bg = frame_bg.astype(np.uint8)
        frame_fg = frame_fg.astype(np.uint8)



        cv.imwrite('BoundingBox/'+video_name+'/Box '+str(i)+'.png',frame)
        cv.imwrite('Foreground/'+video_name+'/Foreground '+str(i)+'.png',frame_fg)
        cv.imwrite('Mask/'+video_name+'/BG Mask '+str(i)+'.png',fgMask)
        cv.imwrite('Background/'+video_name+'/Background '+str(i)+'.png',frame_bg)
        # cv.imwrite('Background/Cubic Interpolation/'+video_name+'/Background '+str(i)+'.png',frame_bg_cubic)


        i=i+1   

        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

    # When everything done, release the video capture and video write objects
    capture.release()
    box_out.release()
    mask_out.release()

    # Closes all the frames
    cv.destroyAllWindows() 


def train(filename, backSub):
    video = cv.VideoCapture()
    video.open(filename)
    fgMask = None
    while True:
        ret, frame = video.read()
        if frame is None:
            break
        # frame = sr.upsample(frame)
        # frame = cv.detailEnhance(frame, sigma_s=10, sigma_r=0.15)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        # frame = np.float32(frame)
        fgMask = backSub.apply(frame, fgMask, 0.5)

def perform_subtraction(frame, bg_plate, fgMask):
    colored_mask = cv.bitwise_and(frame,frame,mask = fgMask)
    colored_bg_mask = cv.bitwise_and(bg_plate, bg_plate, mask = fgMask)
    
    colored_mask = np.float32(colored_mask)
    colored_bg_mask = np.float32(colored_bg_mask)

    frame_bg = frame - colored_mask
    frame_bg_plate = bg_plate - (bg_plate - colored_bg_mask)

    frame_fg = frame - frame_bg

    frame_bg = cv.add(frame_bg, frame_bg_plate)


    return frame_fg, frame_bg

def perform_interpolation(fromMat, scale, interpolationType):
    newRows = round(np.shape(fromMat)[1] * scale)
    newCols = round(np.shape(fromMat)[0] * scale)

    shape = [newRows, newCols]
    hrMat = np.zeros(shape, dtype=fromMat.dtype)

    hrMat = cv.resize(fromMat, (newRows, newCols), scale, scale, interpolationType)

    return hrMat

def perform_interpolation_mask(mask, fromMat, scale, interpolationType):
    newRows = round(np.shape(fromMat)[1] * scale)
    newCols = round(np.shape(fromMat)[0] * scale)

    shape = [newRows, newCols]
    hrMat = np.zeros(shape, dtype=mask.dtype)

    hrMat = cv.resize(mask, (newRows, newCols), scale, scale, interpolationType)

    return hrMat

def produce_mask(inputMat):
    dstMask = np.empty(inputMat.shape)
    np.copyto(dstMask, inputMat)

    img_float32 = np.float32(dstMask)
    dstMask = cv.cvtColor(img_float32, cv.COLOR_BGR2GRAY)
    dstMask = dstMask.astype(np.uint8)
    dstMask = cv.threshold(dstMask, 1, 1, cv.THRESH_BINARY)[1]
    return dstMask
