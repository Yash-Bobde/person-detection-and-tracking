# Person Detection(Yolov8x) & Tracking(DeepSORT) in Video Streams.

## Overview
This project aims to detect and count the number of distinct person in a given input video stream using Yolov8x for object detection and DeepSORT object tracking.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Model Configuration](#model-configuration)
- [Tracking Mechanisms](#tracking-mechanisms)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Execution Configuration
- **GPU**: NVIDIA RTX 3070
- **Model**: YOLOv8x.pt for object detection
- **Python Version:**: 3.11.10
- **Python Version:**: 3.11.10
- Python (version required: `>=3.8`)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/insp7/cv-proj3.git
   cd cv-proj3

2. Set up a virtual environment (optional but recommended):
    ```bash
    # Create a virtual environment
    python -m venv venv

    # Activate the virtual environment
    # Windows
    .\venv\Scripts\activate

    # macOS/Linux
    source venv/bin/activate

To set up a virtual environment using Anaconda:

    ```bash
    # If you don\'t have Anaconda installed, you can download it from the Anaconda website.
    # Create a new conda environment:
    conda create --name person_tracking python=3.11

    # Activate the virtual environment:
    conda activate person_tracking