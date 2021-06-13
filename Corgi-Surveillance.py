import os
import time
import csv
from datetime import datetime
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

    if not os.path.exists("Foreground"):
        os.makedirs("Foreground")

    if not os.path.exists("Foreground/" + video_name):
        os.makedirs("Foreground/" + video_name)

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
    
    if not os.path.exists("Temp"):
        os.makedirs("Temp")

def main():
    start = time.time()
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    
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
    end = time.time()
    
    with open("Processing Time.csv", mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter = ',', quotechar = '"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([video_name, end-start, dt_string])
    print(video_name +": " + str(end - start) + " seconds")

if __name__ == "__main__":
    main()
