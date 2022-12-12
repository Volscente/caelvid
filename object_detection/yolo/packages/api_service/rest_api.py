# Import Standard Libraries
from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np

# Instance FastAPI object
app = FastAPI()


@app.post('/file')
def upload_image(image_file: UploadFile = File(...)):

    # Read the 'SpooledTemporaryFile' in image_file.file.read() as Numpy Array
    numpy_array = np.frombuffer(image_file.file.read(), np.uint8)

    # Read OpenCV Image
    image = cv2.imdecode(numpy_array, cv2.IMREAD_UNCHANGED)

    # Read blob from image
    blob = cv2.dnn.blobFromImage(image=image,
                                 size=(416, 416),
                                 scalefactor=1/255.,
                                 swapRB=True,
                                 crop=False)

    return {'Blob Shape': blob.shape}