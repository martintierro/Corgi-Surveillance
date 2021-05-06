import os
from tkinter import Tk # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
from background_subtraction import background_subtraction
from super_resolution import super_resolution
from background_detection import background_detection

def init_folders(video_name):

    if not os.path.exists("Background"):
        os.makedirs("Background")
    
    if not os.path.exists("Background/" + video_name):
        os.makedirs("Background/" + video_name)

    # if not os.path.exists("Background/Linear Interpolation"):
    #     os.makedirs("Background/Linear Interpolation")

    # if not os.path.exists("Background/Linear Interpolation/" + video_name):
    #     os.makedirs("Background/Linear Interpolation/" + video_name)
    
    # if not os.path.exists("Background/Cubic Interpolation"):
    #     os.makedirs("Background/Cubic Interpolation")

    # if not os.path.exists("Background/Cubic Interpolation/" + video_name):
    #     os.makedirs("Background/Cubic Interpolation/" + video_name)

    if not os.path.exists("Foreground"):
        os.makedirs("Foreground")

    if not os.path.exists("Foreground/" + video_name):
        os.makedirs("Foreground/" + video_name)

    if not os.path.exists("BoundingBoxesVideo"):
        os.makedirs("BoundingBoxesVideo")

    if not os.path.exists("BoundingBox"):
        os.makedirs("BoundingBox")

    if not os.path.exists("BoundingBox/" + video_name):
        os.makedirs("BoundingBox/" + video_name)
        
    if not os.path.exists("Mask"):
        os.makedirs("Mask")

    if not os.path.exists("Mask/" + video_name):
        os.makedirs("Mask/" + video_name)
    
    if not os.path.exists("Super Resolution"):
        os.makedirs("Super Resolution")

    if not os.path.exists("Super Resolution/" + video_name):
        os.makedirs("Super Resolution/" + video_name)

def main():
    #Open Video Feed
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file


    temp = filename.split("/")
    fn = temp[-1]
    video_name = fn.split(".")[0]

    init_folders(video_name)
    background_detection(filename, video_name)
    background_subtraction(filename, video_name)
    super_resolution(filename, video_name)

if __name__ == "__main__":
    main()
