# Import Standard Modules
import os
import pytest

from typing import Tuple

# Set root path
os.chdir(os.environ['YOLO_OBJECT_DETECTION_PATH'])

# Import Package Libraries
from modules.pytest_test.test_utils_fixtures import test_object_detector
from modules.object_detection.object_detection import ObjectDetector
from modules.object_detection.object_detection_utils import read_blob_from_local_image


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
        input_class: String class name

    Returns:
    """

    assert input_class in test_object_detector.classes


@pytest.mark.parametrize('input_layer', [
    'leaky_3',
    'shortcut_4',
    'conv_9'
])
def test__read_neural_network(test_object_detector: ObjectDetector,
                              input_layer: str):
    """
    Test the function modules.object_detection.object_detection.ObjectDetector.__read_neural_network

    Args:
        test_object_detector: ObjectDetector instance
        input_layer: String layer name

    Returns:
    """

    assert input_layer in test_object_detector.neural_network_layers


@pytest.mark.parametrize('image_path, size, scale_factor, swap_rb, crop, expected_shape', [
    ('./data/test_images/image_1.jpeg', (416, 416), 1/255.0, True, False, (1, 3, 416, 416))
])
def test_read_blob_from_local_image(image_path: str,
                                    size: Tuple[int, int],
                                    scale_factor: float,
                                    swap_rb: bool,
                                    crop: bool,
                                    expected_shape: Tuple[int, int, int, int]):
    """
    Test the function modules.object_detection.object_detection_utils.read_blob_from_local_image

    Args:
        image_path: String image path
        size: Tuple[int, int] resize dimensions
        scale_factor: Float pixel scale factor
        swap_rb: Bool flag for swapping R channel with B channel
        crop: Bool flag to crop the image
        expected_shape: Tuple[int, int, int, int] expected resulting blob shape

    Returns:
    """

    # Apply the function
    blob = read_blob_from_local_image(image_path,
                                      size,
                                      scale_factor,
                                      swap_rb,
                                      crop)

    assert blob.shape == expected_shape


