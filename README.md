# Person Detection(Yolov8x) & Tracking(DeepSORT) in Video Streams.

## Overview
This project aims to detect and count the number of distinct person in a given input video stream using Yolov8x for object detection and DeepSORT object tracking.

## Execution Configuration
- **GPU**: NVIDIA RTX 3070
- **Model**: YOLOv8x for object detection + DeepSORT for object Tracking
- **Python Version:**: 3.11.10

For detailed list of packages used, check requirements.txt

## Program Descriptions + Sample Outputs

#### Simple object detection using Yolov8x.pt model (Source: **yolov8x.py**)
![Simple object detection using Yolov8x.pt model.](https://github.com/insp7/cv-proj3/blob/master/gifs/1Person.gif)
- **yolov8x_plus_iou.py**: Detection + Tracking using Interaction Over Union (IOU) -- Highly inaccurate tracking mechanism.
- **yolov8x_plus_sort.py**: Detection + Tracking using SORT Tracker https://github.com/abewley/sort -- Significantly more accurate than IOU.
- **yolov8x_plus_deepsort.py**: Detection + Tracking using DeepSORT Tracker https://github.com/nwojke/deep_sort -- Again, a huge improvement over SORT.
- **yolov8x_plus_deepsort_process_image_frames.py**: Added functionality to accept a directory of images which then can be processed as frames for the input video.

## Data Sources
- https://motchallenge.net/data/MOT16/
- https://www.youtube.com/watch?v=WvhYuDvH17I
- https://www.youtube.com/watch?v=ZvzKuqSDyG8