import cv2 as cv
import numpy as np
import argparse
import scipy
from scipy import *
from scipy.sparse import linalg
from matplotlib import pyplot as plt
from trimap_module import trimap, checkImage
import closed_form_matting


scale = 2

model_path = "models/ESPCN_x2.pb"
sr = cv.dnn_superres.DnnSuperResImpl_create()

def background_subtraction(filename, video_name):
    #contour threshold
    threshold_area = 500 

    sr.readModel(model_path)
    sr.setModel("espcn", scale)

    kernel = np.ones( (9,9), np.uint8 )
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

    capture = cv.VideoCapture()
    # capture.open("bike.mp4")
    capture.open(filename)


    if not capture.isOpened():
        print('Unable to open')
        exit(0)

    counter = 0

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
    backSubGSOC = cv.bgsegm.createBackgroundSubtractorGSOC() 
    backSubGMG = cv.bgsegm.createBackgroundSubtractorGMG()
    backSubMOG2 = cv.createBackgroundSubtractorMOG2()

    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT)) 
    if frame_count < 120:
        train(filename, backSubGMG)
        train(filename, backSubGMG)
    else:
        train(filename, backSubGMG)


    
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        print("BG Subtraction Frame Number: " + str(counter) + "/" + str(frame_count))
        frame = sr.upsample(frame)
        unsharp_frame = unsharp_mask(frame)
        cv.imshow("Unsharp Frame", unsharp_frame)
        frame_bw = cv.cvtColor(unsharp_frame, cv.COLOR_RGB2GRAY)
        fgMaskGSOC = backSubGSOC.apply(unsharp_frame)
        fgMaskGMG = backSubGMG.apply(frame_bw, 0.0005)
        fgMaskMOG2 = backSubMOG2.apply(unsharp_frame)
        # fgMaskEdge = get_mask(frame)
        fgMask = np.median([fgMaskGMG, fgMaskGSOC, fgMaskMOG2], axis=0).astype(dtype=np.uint8)  

        frame = np.float32(frame)
        frame_bw = np.float32(frame_bw)
        bg_plate = np.float32(bg_plate)
    
        # Get foreground Mask
        
        # fgMask = cv.morphologyEx(fgMask,cv.MORPH_ERODE, kernel)
        # fgMask = cv.morphologyEx(fgMask,cv.MORPH_ERODE, kernel)
        # fgMask = cv.morphologyEx(fgMask,cv.MORPH_ERODE, kernel)
        # fgMask = cv.morphologyEx(fgMask,cv.MORPH_ERODE, kernel)
        # if checkImage(fgMask):
        #     fgMask = trimap(fgMask, 3, False)
        # fgMask = perform_interpolation_mask(fgMask, frame, scale, cv.INTER_NEAREST)
        vid_mask = cv.cvtColor(fgMask.astype(np.uint8), cv.COLOR_GRAY2RGB)
        mask_out.write(vid_mask)
       
        # fgMask = cv.morphologyEx(fgMask,cv.MORPH_OPEN, kernel) 
        fgMask = cv.morphologyEx(fgMask,cv.MORPH_CLOSE, kernel)
        # fgMask = cv.medianBlur(fgMask, 5)
        # cv.imshow(video_name, fgMask)

        
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
        # cv.imshow("Bounding Box", frame)
        # fgMask = fgMask * 255
        frame_bg = frame_bg.astype(np.uint8)
        frame_fg = frame_fg.astype(np.uint8)



        #Exports Frames
        # cv.imwrite('Mask/Raw BG/'+video_name+'/BG Mask '+str(i)+'.png',fgMask)
        # cv.imwrite('Background/Raw BG/'+video_name+'/Background '+str(i)+'.png',frame_bg)
        # cv.imwrite('Foreground/Raw BG/'+video_name+'/Foreground '+str(i)+'.png',frame_fg)


        cv.imwrite('BoundingBox/'+video_name+'/Box '+str(counter)+'.png',frame)
        cv.imwrite('Foreground/'+video_name+'/Foreground '+str(counter)+'.png',frame_fg)

        cv.imwrite('Mask/'+video_name+'/BG Mask '+str(counter)+'.png',fgMask)
        cv.imwrite('Background/'+video_name+'/Background '+str(counter)+'.png',frame_bg)
        # cv.imwrite('Background/Cubic Interpolation/'+video_name+'/Background '+str(i)+'.png',frame_bg_cubic)


        counter=counter+1   

        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

    # When everything done, release the video capture and video write objects
    capture.release()
    box_out.release()
    mask_out.release()

    # Closes all the frames
    cv.destroyAllWindows() 

def get_mask(img):
    img = img.astype(np.uint8)

    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(img, (9, 9), 0)

    # Edge detection 
    edges = cv.Canny(gray, 100, 200)
    edges = cv.dilate(edges, None)
    edges = cv.erode(edges, None)

    # Find contours in edges, sort by area 
    contour_info = []
    contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        contour_info.append((
            c,
            cv.isContourConvex(c),
            cv.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    # Create empty mask and flood fill
    mask = np.zeros(edges.shape)
    for c in contour_info:
        cv.fillConvexPoly(mask, c[0], (255))

    # Smooth mask and blur it
    # mask = cv.dilate(mask, None, iterations=10)
    # mask = cv.erode(mask, None, iterations=10)
    # mask = cv.GaussianBlur(mask, (21, 21), 0)

    return mask

def train(filename, backSub):
    video = cv.VideoCapture()
    video.open(filename)
    fgMask = None
    while True:
        ret, frame = video.read()
        if frame is None:
            break
        frame = sr.upsample(frame)
        # frame = cv.detailEnhance(frame, sigma_s=10, sigma_r=0.15)
        frame = unsharp_mask(frame)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        frame = np.float32(frame)
        fgMask = backSub.apply(frame, 0.5)

def perform_subtraction(frame, bg_plate, fgMask):
    colored_mask = cv.bitwise_and(frame,frame,mask = fgMask)
    colored_bg_mask = cv.bitwise_and(bg_plate, bg_plate, mask = fgMask)
    # colored_mask = np.concatenate([frame, fgMask[:, :, np.newaxis]], axis=2)
    # colored_bg_mask = np.concatenate([bg_plate, fgMask[:, :, np.newaxis]], axis=2)
    colored_mask = np.float32(colored_mask)
    colored_bg_mask = np.float32(colored_bg_mask)
    colored_mask = cv.cvtColor(colored_mask, cv.COLOR_RGBA2RGB)
    colored_bg_mask = cv.cvtColor(colored_bg_mask, cv.COLOR_RGBA2RGB)

    
    

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

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened