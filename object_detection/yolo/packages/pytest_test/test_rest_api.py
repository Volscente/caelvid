# Import Standard Libraries
import os
import pytest
import json

from fastapi.testclient import TestClient

# Set root path
os.chdir(os.environ['YOLO_OBJECT_DETECTION_PATH'])

# Import Package Modules
from packages.rest_api.rest_api import app

# Instance TestClient object
test_client = TestClient(app)



def test_detect_object():

    """
    TODO: Docstrings
    Returns:

    """

    # Define the File to upload
    files = {"image": open("./data/test_images/image_1.jpeg", "rb")}

    # Retrieve the response
    response = test_client.post("/detect_object/", files=files)

    # Parse the response as JSON
    json_response = json.loads(response.content.decode('utf-8'))

    print('Response:')
    print(json_response['detected_object'])
