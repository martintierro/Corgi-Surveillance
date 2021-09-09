# Corgi-Surveillance

## Introduction
Closed-circuit television (CCTV) footage is becoming increasingly used for security and surveillance worldwide. Additionally, they are being used outside of security for purposes such as object or person recognition and tracking to monitor trends and to improve efficiency in a working environment. However, its effectiveness in these scenarios is heavily dependent on the quality of the footage.

This study explores a multi-frame method of enhancing CCTV videos recreated from the Eagle Eye framework of Del Gallego and Ilao using techniques observed from the field of background subtraction and multiframe super resolution (MISR). The application, Corgi Eye, reduces the ghosting artifacts in the video when performing MISR on frames with moving objects.


## Prerequisites
- Windows 10 or above
- Python 3
- MATLAB (for obtaining metrics)


## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/martintierro/Corgi-Surveillance.git
cd Corgi-Surveillance
```

- Install dependencies.
  - Please type the command `pip install -r requirements.txt`.

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
