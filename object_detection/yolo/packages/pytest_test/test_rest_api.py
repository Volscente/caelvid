# Import Standard Libraries
import os

from fastapi.testclient import TestClient

# Set root path
os.chdir(os.environ['YOLO_OBJECT_DETECTION_PATH'])

# Import Package Modules
from packages.rest_api.rest_api import app

# Instance TestClient object
test_client = TestClient(app)


def test_detect_object():

    #with open('./data/test_images/image_1.jpeg', 'rb') as file:

        # Call the REST API
        #response = test_client.post('/detect_object/',
        #                            files={'file': ('filename', file, 'image/jpeg')})

    files = {"image": open("./data/test_images/image_1.jpeg", "rb")}
    response = test_client.post("/detect_object/", files=files)

    print('Response:')
    print(response)
