# Import Standard Modules
import os
import pytest

# Set root path
os.chdir(os.environ['YOLO_OBJECT_DETECTION_PATH'])

# Import Package Libraries
from modules.pytest_test.test_utils_fixtures import test_object_detector
from modules.object_detection.object_detection import ObjectDetector


def test_environment_variable(test_object_detector: ObjectDetector):
    """
    Test the correct set of the environment variables YOLO_OBJECT_DETECTION

    Args:
        test_object_detector: ObjectDetector instance

    Returns:
        Boolean
    """

    assert os.getcwd() == os.environ['YOLO_OBJECT_DETECTION_PATH']