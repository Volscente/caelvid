# Import Standard Modules
import os
import pytest

# Set root path
os.chdir(os.environ['YOLO_OBJECT_DETECTION_PATH'])

# Import Package Libraries
from modules.pytest_test.test_utils_fixtures import test_object_detector
from modules.object_detection.object_detection import ObjectDetector


@pytest.mark.parametrize('input_class', [
    'cow',
    'frisbee',
    'banana'
])
def test__read_classes(test_object_detector: ObjectDetector,
                       input_class: str):
    """
    Test the function modules.object_detection.object_detection.ObjectDetector.__read_classes

    Args:
        test_object_detector: ObjectDetector instance

    Returns:
    """

    assert input_class in test_object_detector.classes

