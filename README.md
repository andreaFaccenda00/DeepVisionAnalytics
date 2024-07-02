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

### YOLOv8 Training for People Flow Analysis

#### YOLOv8 Training
We trained YOLOv8 variants (YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l) on the MOTSynth and EuroCity Persons datasets. MOTSynth is a synthetic dataset generated using GTA V, while EuroCity Persons consists of urban images from various European cities.

- **Datasets**: 
  - **MOTSynth**: Large-scale synthetic dataset for pedestrian detection, segmentation, and tracking.
  - **EuroCity Persons**: Urban images from multiple European cities for realistic pedestrian detection.

- **Training Split**: 80% for training, 10% for validation, 10% for testing.
- **Image Size**: 640x640 pixels.

#### Hyperparameters for All YOLOv8 Variants
| Hyperparameter       | YOLOv8n  | YOLOv8s  | YOLOv8m  | YOLOv8l  |
|----------------------|----------|----------|----------|----------|
| Batch size           | 16       | 16       | 16       | 16       |
| Image size           | 640x640  | 640x640  | 640x640  | 640x640  |
| Epochs               | 250      | 250      | 250      | 250      |
| Early stopping       | 30       | 30       | 30       | 30       |
| Optimizer            | SGD      | SGD      | SGD      | SGD      |
| Initial learning rate| 0.01     | 0.01     | 0.01     | 0.01     |
| LR reduction factor  | 0.01     | 0.01     | 0.01     | 0.01     |
| Momentum             | 0.95     | 0.95     | 0.95     | 0.95     |
| Weight decay         | 0.0001   | 0.0001   | 0.0001   | 0.0001   |
| IOU threshold        | 0.7      | 0.7      | 0.7      | 0.7      |
| Detection limit      | 300      | 300      | 300      | 300      |
| Mixed precision      | Yes      | Yes      | Yes      | Yes      |
| Warmup epochs        | 10       | 10       | 10       | 10       |
| Warmup momentum      | 0.5      | 0.5      | 0.5      | 0.5      |
| Warmup LR            | 0.1      | 0.1      | 0.1      | 0.1      |
| Masking ratio        | 4        | 4        | 4        | 4        |

#### Testing and Evaluation
The trained YOLOv8 variants were tested on the SOMPT22 dataset in addition to the MOTSynth and EuroCity Persons datasets. SOMPT22 was used exclusively for testing to provide a rigorous evaluation of the model's capability in urban surveillance scenarios.

| Dataset           | Model   | Inference | mAP@50 | mAP@50-95 | Precision | Recall |
|-------------------|---------|-----------|--------|-----------|-----------|--------|
| MOTSynth          | YOLOv8n | 2.1ms     | 0.841  | 0.659     | 0.943     | 0.705  |
|                   | YOLOv8s | 2.4ms     | 0.863  | 0.708     | 0.956     | 0.742  |
|                   | YOLOv8m | 3.9ms     | 0.873  | 0.733     | 0.964     | 0.757  |
|                   | YOLOv8l | 6.4ms     | 0.879  | 0.75      | 0.97      | 0.768  |
| EuroCity Persons  | YOLOv8n | 2.4ms     | 0.702  | 0.479     | 0.721     | 0.628  |
|                   | YOLOv8s | 3.2ms     | 0.741  | 0.521     | 0.766     | 0.664  |
|                   | YOLOv8m | 6.6ms     | 0.781  | 0.583     | 0.768     | 0.714  |
|                   | YOLOv8l | 10.4ms    | 0.833  | 0.651     | 0.786     | 0.795  |
| MOTSynth EuroCity | YOLOv8n | 2.7ms     | 0.854  | 0.691     | 0.915     | 0.739  |
|                   | YOLOv8s | 2.7ms     | 0.854  | 0.691     | 0.915     | 0.739  |
|                   | YOLOv8m | 4.7ms     | 0.864  | 0.72      | 0.925     | 0.753  |
|                   | YOLOv8l | 8.3ms     | 0.87   | 0.734     | 0.925     | 0.766  |
| Coco              | YOLOv8n | 2.3ms     | 0.721  | 0.591     | 0.891     | 0.667  |
|                   | YOLOv8s | 2.6ms     | 0.745  | 0.612     | 0.902     | 0.684  |
|                   | YOLOv8m | 4.0ms     | 0.762  | 0.635     | 0.911     | 0.698  |
|                   | YOLOv8l | 6.5ms     | 0.781  | 0.652     | 0.918     | 0.712  |

YOLOv8s was chosen for its balance of rapid inference (2.4ms), high precision (0.956), and significant recall (0.742), making it suitable for real-time surveillance and monitoring.

#### Visual Results
The following images demonstrate the performance and evaluation metrics of the YOLOv8s model:

- **Confusion Matrix**:
  ![confusion_matrix](https://github.com/andreaFaccenda00/DeepVisionAnalytics/assets/171338421/5918e13d-4d1f-4c21-bb85-dd2a99a889f0)

- **F1-Confidence Curve**:
  ![F1-Confidence Curve](file:///mnt/data/F1_curve.png)

- **Training Metrics**:
  ![Training Metrics](file:///mnt/data/results.png)

These results illustrate the model's accuracy in detecting pedestrians, its confidence at various thresholds, and the improvements in training metrics over time.
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

https://github.com/andreaFaccenda00/DeepVisionAnalytics/assets/171338421/d424407b-2e08-4e76-8126-5c64b2b2fa58


