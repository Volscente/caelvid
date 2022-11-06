# Import Standard Modules
import os
import pytest
import numpy as np

from typing import Tuple

# Set root path
os.chdir(os.environ['YOLO_OBJECT_DETECTION_PATH'])

# Import Package Libraries
from modules.pytest_test.test_utils_fixtures import test_object_detector, test_blob, test_configuration
from modules.object_detection.object_detection import ObjectDetector
from modules.object_detection.object_detection_utils import read_blob_from_local_image, retrieve_neural_network_output


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


@pytest.mark.parametrize('image_path, expected_shape', [
    ('./data/test_images/image_1.jpeg', (1, 3, 416, 416))
])
def test_read_blob_from_local_image(image_path: str,
                                    expected_shape: Tuple[int, int, int, int],
                                    test_configuration: dict):
    """
    Test the function modules.object_detection.object_detection_utils.read_blob_from_local_image

    Args:
        image_path: String image path
        expected_shape: Tuple[int, int, int, int] expected resulting blob shape
        test_configuration: Dictionary configuration object

    Returns:
    """

    # Apply the function
    blob = read_blob_from_local_image(image_path,
                                      test_configuration['blob_size'],
                                      test_configuration['blob_scale_factor'],
                                      test_configuration['blob_swap_rb'],
                                      test_configuration['blob_crop'])

    assert blob.shape == expected_shape


@pytest.mark.parametrize('test_output_layer', [
    'yolo_82',
    'yolo_94',
    'yolo_106'
])
def test__get_output_layers(test_object_detector: ObjectDetector,
                            test_output_layer: str):
    """
    Test the function modules.object_detection.object_detection.ObjectDetector.__get_output_layers
    Args:
        test_object_detector: ObjectDetector instance
        test_output_layer: String output layer name

    Returns:
    """

    assert test_output_layer in test_object_detector.output_layers


def test_retrieve_neural_network_output(test_object_detector: ObjectDetector,
                                        test_blob: np.ndarray):
    """
    Test the function Test the function modules.object_detection.object_detection_utils.retrieve_neural_network_output

    Args:
        test_object_detector: ObjectDetector instance
        test_blob: Numpy.ndarray Blob of the image

    Returns:
    """

    # Feed forward the test_blob into the test_object_detector.neural_network and retrieve outputs
    outputs = retrieve_neural_network_output(test_object_detector.neural_network,
                                             test_blob,
                                             test_object_detector.output_layers)

    assert len(outputs) == 3  # Fixed number of output layers
