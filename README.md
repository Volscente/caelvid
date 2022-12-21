# Introduction
The repository contains several computer vision models

# Object Detection

## Yolo

### Installation

##### Set up Environment Variables
Set up an environment variables called `YOLO_OBJECT_DETECTION_PATH` that hosts the path to the `./object_detection/yolo`
folder.

### Running API Service
Change directory into `object_detection/yolo` and run the following command:
``` bash
uvicorn packages.rest_api.rest_api:app --reload
```