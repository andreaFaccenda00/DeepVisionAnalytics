# DeepVisionAnalytics
## Introduction 
The Computer Vision and Deep Learning project "Wait Time Optimization and Analysis of Interactions in Public Areas," part of the CTE SQUARE Pesaro, tackles these challenges using advanced computer vision (CV) and Deep Learning (DL) techniques. The goal is to monitor and analyze the flow of people to reduce waiting times and better understand people's behaviors and attention. The project focuses on three main aspects: wait time analysis, optimization of DL models for edge devices, and practical implementation. By installing webcams in Piazza del Popolo in Pesaro and employing YOLOv8 (You Only Look Once) neural networks, the project aims to monitor the time people spend in specific areas to identify critical points and optimize the efficiency of public spaces.

## Project Overview

This project focuses on three main objectives:
1. **Wait Time Analysis**: Monitoring the duration people spend in specific areas to identify bottlenecks.
2. **Model Optimization**: Enhancing DL models for deployment on edge devices.
3. **Practical Implementation**: Installing webcams and deploying models in real-world scenarios.

## Supervision

<div align="center">
  <p>
    <a align="center" href="" target="https://supervision.roboflow.com">
      <img
        width="100%"
        src="https://media.roboflow.com/open-source/supervision/rf-supervision-banner.png?updatedAt=1678995927529"
      >
    </a>
  </p>

  <br>


  <br>
</div>


Supervision is a powerful library used in the "Wait Time Optimization and Analysis of Interactions in Public Areas" project to enhance the capabilities of computer vision and deep learning applications. This library provides a set of tools and utilities designed to simplify the process of training, evaluating, and deploying deep learning models. Its features include data augmentation, model evaluation metrics, and support for various neural network architectures, making it a versatile choice for developing advanced computer vision solutions.
## 💻 install

####  1. Install the Supervision Package via Pip

To install the supervision package in a [**Python>=3.8**](https://www.python.org/) environment, use the following command:
```bash
pip install supervision
```
#### 2. Verify Installation
To verify that the installation was successful, run the following commands:
```bash
import supervision
print(supervision.__version__)
```
#### 3. Install Miniconda
Download and install Miniconda from the official Miniconda website. [Miniconda website](https://docs.conda.io/en/latest/miniconda.html).

#### 4. Add Conda to Your Environment Variables
Ensure that Conda is added to your environment variables during the installation process.

#### 5. Install the Supervision Package via Conda
Once Miniconda is installed and configured, open your terminal (or Anaconda Prompt on Windows) and run the following command:
If no errors occur and the version number is displayed, the installation was successful.
Then to install the necessary package, please follow these steps:
```bash
conda install -c conda-forge supervision
```
#### 6. Clone the Repository and Set Up the Python Environment
Clone the repository and navigate to the root directory:
```bash
git clone https://github.com/andreaFaccenda00/DeepVisionAnalytics.git
```
Set up the Python environment and activate it:
```bash
python3 -m venv venv
source venv\Scripts\activate
pip install --upgrade pip
```
Perform a headless install:
```bash
pip install -e "."
```
For desktop installation, use:
```bash
pip install -e ".[desktop]"
```

#### 7. Install Required Dependencies
Install the required dependencies listed in requirements.txt:
```bash
pip install -r requirements.txt
```

## 🛠 scripts

### `draw_zones`

If you want to test zone time in zone analysis on your own video, you can use this
script to design custom zones and save results as a JSON file. The script will open a
window where you can draw polygons on the source image or video file. The polygons will
be saved as a JSON file.

- `--source_path`: Path to the source image or video file for drawing polygons.
- `--zone_configuration_path`: Path where the polygon annotations will be saved as a JSON file.


- `enter` - finish drawing the current polygon.
- `escape` - cancel drawing the current polygon.
- `q` - quit the drawing window.
- `s` - save zone configuration to a JSON file.

```bash
python scripts/draw_zones.py 
--source_path "data/people.mp4" 
--zone_configuration_path "data/config.json"
```

https://github.com/roboflow/supervision/assets/26109316/9d514c9e-2a61-418b-ae49-6ac1ad6ae5ac

## 🎬 video processing

### `main`

Script to run object detection on a video file using the Ultralytics YOLOv8 model.Key parameters include:

  - `--zone_configuration_path`: Path to the zone configuration JSON file.
  - `--source_video_path`: Path to the source video file.
  - `--weights`: Path to the model weights file. Default is `'yolov8s_pedestrian.pt'`.
  - `--device`: Computation device (`'cpu'`, `'mps'` or `'cuda'`). Default is `'cuda'`.
  - `--classes`: List of class IDs to track. If empty, all classes are tracked.
  - `--confidence_threshold`: Confidence level for detections (`0` to `1`). Default is `0.3`.
  - `--iou_threshold`: IOU threshold for non-max suppression. Default is `0.7`.

#### Running the Code
To run this code, ensure you have all the required libraries installed and the correct file paths set for your video, configuration, and model weights. Execute the script as follows:
```bash
python main.py
```

The script will process the video, detect and track objects, annotate zones of interest, and calculate the time spent in each zone. The output will be saved as an annotated video.

#### video analysis

(https://github.com/andreaFaccenda00/DeepVisionAnalytics/assets/171338421/d424407b-2e08-4e76-8126-5c64b2b2fa58)


