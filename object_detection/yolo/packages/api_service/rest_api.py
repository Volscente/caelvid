# Import Standard Libraries
from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import os

# Set root path
os.chdir(os.environ['YOLO_OBJECT_DETECTION_PATH'])

# Import Package Libraries
from packages.logging_module.logging_module import get_logger
from packages.object_detection.object_detection import ObjectDetector

# Setup logger
logger = get_logger(os.path.basename(__file__).split('.')[0])

# Instance FastAPI object
app = FastAPI()


@app.post('/detect_object/')
def detect_object(image: UploadFile = File(..., description='Image file from which detect the object')):

    # Read the 'SpooledTemporaryFile' in image_file.file.read() as Numpy Array
    image_numpy = np.frombuffer(image.file.read(), np.uint8)

    return {'detected_object': None}