# Import Standard Libraries
from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
import os

# Set root path
os.chdir(os.environ['YOLO_OBJECT_DETECTION_PATH'])

# Import Package Libraries
from packages.logging_module.logging_module import get_logger
from packages.object_detection.object_detection import ObjectDetector
from packages.utils.utils import read_configuration

# Setup logger
logger = get_logger(os.path.basename(__file__).split('.')[0])

# Read Configuration file
config = read_configuration('config.yaml')

# Instance FastAPI object
app = FastAPI()


@app.post('/detect_object/')
def detect_object(image: UploadFile = File(...,
                                           description='Image file from which detect the object')):

    logger.info('detect_object - Check file content type')

    # Check file content type
    if image.content_type not in config['detect_object_valid_content_types']:

        raise HTTPException(403,
                            detail="detect_object - Invalid file type")

    logger.info('detect_object - Read image as Numpy Array')

    # Read the 'SpooledTemporaryFile' in image_file.file.read() as Numpy Array
    image_numpy = np.frombuffer(image.file.read(), np.uint8)

    # Instantiate ObjectDetector
    object_detector = ObjectDetector()

    # Compute detected object
    detected_object = object_detector.detect_single_object(image_numpy)

    return {'detected_object': detected_object}
