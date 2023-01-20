# Import Libraries Modules
import pytest
import os
import numpy as np

# Set root path
os.chdir(os.environ['YOLO_OBJECT_DETECTION_PATH'])

# Import Package Modules
from src.object_detection_yolov3.object_detection import ObjectDetector
from src.object_detection_yolov3.object_detection_utils import read_image_from_source, read_blob_from_image
from src.utils.utils import read_configuration


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
def test_image(test_configuration: dict) -> np.ndarray:
    """
    Fixture for an Open CV image representation

    Returns:
        Numpy.ndarray Open CV image representation
    """

    return read_image_from_source(test_configuration['test_image'])


@pytest.fixture
def test_blob(test_image: np.ndarray,
              test_configuration: dict) -> np.ndarray:
    """
    Fixture for a Blob image

    Args:
        test_image: Numpy.ndarray Open CV image representation
        test_configuration: Dictionary configuration object

    Returns:
        Numpy.ndarray Blob
    """

    return read_blob_from_image(test_image,
                                test_configuration['blob_size'],
                                test_configuration['blob_scale_factor'],
                                test_configuration['blob_swap_rb'],
                                test_configuration['blob_crop'])
