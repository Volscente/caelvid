# Import Libraries Modules
import pytest
import os
import numpy as np

# Set root path
os.chdir(os.environ['YOLO_OBJECT_DETECTION_PATH'])

# Import Package Libraries
from modules.object_detection.object_detection import ObjectDetector
from modules.object_detection.object_detection_utils import read_blob_from_local_image
from modules.utils.utils import read_configuration


@pytest.fixture
def test_configuration() -> dict:
    """
    Fixture for a Dictionary configuration object

    Returns:
        Dictionary configuration
    """

    return read_configuration('config.yaml')


@pytest.fixture
def test_object_detector() -> ObjectDetector:
    """
    Fixture for an instance of the class ObjectDetector

    Returns:
        ObjectDetector instance
    """

    return ObjectDetector()


@pytest.fixture
def test_blob(test_configuration) -> np.ndarray:
    """
    Fixture for a Blob image

    Args:
        test_configuration: Dictionary configuration object

    Returns:
        Numpy.ndarray Blob
    """

    pass