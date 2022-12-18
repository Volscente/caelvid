# Import Standard Libraries
import os

from fastapi import FastAPI
from fastapi.testclient import TestClient

# Set root path
os.chdir(os.environ['YOLO_OBJECT_DETECTION_PATH'])

# Instance FastAPI & TestClient objects
app = FastAPI()
test_client = TestClient(app)


def test_detect_object():
    pass
