<span>
  <img src="img/Game Lab Logo.png" width="50"/>&nbsp;&nbsp;&nbsp;
  <img src="img/Corgi Logo.png" width="50"/>
</span>

# Corgi Eye

## Introduction
Closed-circuit television (CCTV) footage is becoming increasingly used for security and surveillance worldwide. Additionally, they are being used outside of security for purposes such as object or person recognition and tracking to monitor trends and to improve efficiency in a working environment. However, its effectiveness in these scenarios is heavily dependent on the quality of the footage.

This study explores a multi-frame method of enhancing CCTV videos recreated from the Eagle Eye framework of Del Gallego and Ilao using techniques observed from the field of background subtraction and multiframe super resolution (MISR). The application, Corgi Eye, reduces the ghosting artifacts in the video when performing MISR on frames with moving objects.

<img src="img/Comparison KITTI Person.png" width="1000" align="center"/>
<img src="img/Comparison Archer's Eye.png" width="1000" align="center"/>
<img src="img/Comparison Pasig.png" width="1000" align="center"/>

## System Architecture
Upon inputting a video file, the input is subjected to frame extraction where the process of extracting every frame of the video is done. It then undergoes pre-processing, where the system gets an average of randomly selected frames to determine the background, and the background subtractor models the background. This would aid in the main image enhancement process which consists of background subtraction, super resolution, and image blending. The end result of the system is an enhanced video.

<div align="center">
  <img src="img/Framework Architecture.jpg" width="700" />
</div>

### Background Subtraction Architecture
<div align="center">
  <img src="img/Background Subtraction Architecture.jpg" width="700" align="center"/>
</div>

### Super Resolution and Image Blending Architecture
<div align="center">
  <img src="img/Super Resolution and Image Blending Architecture.png" width="700" align="center"/>
</div>

## Prerequisites and Recommended System Requirements
- Windows 10
- Python 3
- MATLAB (for obtaining metrics)
- It is recommended that the script be placed on a large storage device due to the large amount of drive space that the system uses for storing processed frames.
- The use of the software on devices with fewer than 16GB of RAM and 4 CPU cores has not been tested and your mileage may vary.





## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/martintierro/Corgi-Surveillance.git
cd Corgi-Surveillance
```
- Dependencies
  - matplotlib==3.3.4
  - opencv_contrib_python==4.5.2.54
  - natsort==7.1.1
  - numpy==1.21.1 
   
- Install dependencies using the command `pip install -r requirements.txt`.

### To Run
```bash
python Corgi-Surveillance.py
```
- Select chosen video from File Explorer and wait for the program to process the video
- When the program is complete, the results will be found in `/Super Resolution`


## Measuring Results
### To Run
- Open `evaluation.m` in MATLAB
- Edit `video_name` to include all names of the folders that have frames of the outputted video(s)
- Edit the path in `img_hr` to the path containing the ground truth frames. Ensure that the filenames of the frames are numbered sequentially exactly like `hr_0.png`
- Run the script and the results will appear in the MATLAB Command Window

## Acknowledgements
This project is based from [Eagle Eye](https://github.com/NeilDG/EagleEyeSR).
