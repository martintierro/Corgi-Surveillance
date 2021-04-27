
# Taken from https://colab.research.google.com/drive/194DC_lyCRafwlJZ5DyRwrK9Gh2RCciVT?usp=sharing#scrollTo=LmYVPLFXrNyW
import cv2 as cv
import numpy as np

def background_detection(filename, video_name):

    video = cv.VideoCapture(filename)
    FOI = video.get(cv.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=30)

    frames = []
    for frameOI in FOI:
        video.set(cv.CAP_PROP_POS_FRAMES, frameOI)
        ret, frame = video.read()
        frames.append(frame)

    #calculate the average
    backgroundFrame = np.median(frames, axis=0).astype(dtype=np.uint8)    
    cv.imwrite("Background/"+video_name+".png",backgroundFrame)
    # cv.imshow("Background Frame",backgroundFrame)