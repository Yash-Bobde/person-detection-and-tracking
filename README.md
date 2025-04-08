# Person Detection(Yolov8x) & Tracking(DeepSORT) in Video Streams.

By:
Aniket Konkar,
Yash Manish Bobde and
Sakshi Agrawal

## Overview
This project aims to detect and count the number of distinct person in a given input video stream using Yolov8x for object detection and DeepSORT object tracking.

## Execution Configuration
- **GPU**: NVIDIA RTX 3070
- **Model**: YOLOv8x for object detection + DeepSORT for object Tracking
- **Python Version:**: 3.11.10

For detailed list of packages used, check requirements.txt

## Program Descriptions + Sample Outputs

1. Simple object detection using Yolov8x.pt model (Source: **yolov8x.py**)
    ![Simple object detection using Yolov8x.pt model.](https://github.com/insp7/cv-proj3/blob/master/gifs/1Person.gif)

2. Detection + Tracking using Interaction Over Union (IOU) -- Highly inaccurate tracking mechanism. (Source: **yolov8x_plus_iou.py**)
    ![Detection + Tracking using Interaction Over Union (IOU)](https://github.com/insp7/cv-proj3/blob/master/gifs/moti16-01-iou.gif)

3. Detection + Tracking using SORT Tracker https://github.com/abewley/sort -- Significantly more accurate than IOU. (Source: **yolov8x_plus_sort.py**)
    ![Detection + Tracking using SORT Tracker](https://github.com/insp7/cv-proj3/blob/master/gifs/moti16-01-SORT.gif)

4. Detection + Tracking using DeepSORT Tracker https://github.com/nwojke/deep_sort -- Again, a huge improvement over SORT. (Source: **yolov8x_plus_deepsort.py**)
    ![Detection + Tracking using DeepSORT Tracker](https://github.com/insp7/cv-proj3/blob/master/gifs/moti16-01-DeepSORT.gif)

5. Additional Footage: 

![https://github.com/insp7/cv-proj3/blob/master/gifs/moti16-06.gif](https://github.com/insp7/cv-proj3/blob/master/gifs/moti16-06.gif)

![https://github.com/insp7/cv-proj3/blob/master/gifs/moti16-06.gif](https://github.com/insp7/cv-proj3/blob/master/gifs/topview.gif)


## GPU Load Measured while Inferencing (Notice 1200MHz and VRAM gets pushed to 7800MHz)
![https://github.com/insp7/cv-proj3/blob/master/gifs/moti16-12.gif](https://github.com/insp7/cv-proj3/blob/master/gifs/moti16-12.gif)

## Data Sources
- https://motchallenge.net/data/MOT16/
- https://www.youtube.com/watch?v=WvhYuDvH17I
- https://www.youtube.com/watch?v=ZvzKuqSDyG8
