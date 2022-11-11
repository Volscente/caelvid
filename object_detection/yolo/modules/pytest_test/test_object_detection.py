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
from modules.object_detection.object_detection_utils import retrieve_image_width_and_height, read_blob_from_local_image, \
    retrieve_neural_network_output, retrieve_all_detected_classes, retrieve_max_confident_class_index


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

    assert input_layer in test_object_detector.neural_network.getLayerNames()


@pytest.mark.parametrize('image_path, expected_dimensions', [
    ('./data/test_images/image_1.jpeg', (768, 576))
])
def test_retrieve_image_width_and_height(image_path: str,
                                         expected_dimensions: Tuple[int, int]):
    """
    Test the function modules.object_detection.object_detection_utils.retrieve_image_width_and_height

    Args:
        image_path: String image path
        expected_dimensions: Tuple[int, int] expected dimensions

    Returns:
    """

    # Retrieve image width and height
    width, height = retrieve_image_width_and_height(image_path)

    assert width == expected_dimensions[0] and height == expected_dimensions[1]


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


@pytest.mark.parametrize('input_image_width, input_image_height, expected_length', [
    (768, 576, 14)
])
def test_retrieve_all_detected_classes(test_object_detector: ObjectDetector,
                                       test_blob: np.ndarray,
                                       test_configuration: dict,
                                       input_image_width: int,
                                       input_image_height: int,
                                       expected_length: int):
    """
    Test the function Test the function modules.object_detection.object_detection_utils.retrieve_all_detected_classes

    Args:
        test_object_detector: ObjectDetector instance
        test_blob: Numpy.ndarray Blob of the image
        test_configuration: Dictionary configuration object
        input_image_width: Integer image width
        input_image_height: Integer image heigth
        expected_length: Integer expected number of detected classes

    Returns:
    """

    # Compute neural network outputs
    outputs = retrieve_neural_network_output(test_object_detector.neural_network,
                                             test_blob,
                                             test_object_detector.output_layers)

    # Retrieve detected classes
    detected_classes, _, _ = retrieve_all_detected_classes(outputs,
                                                        input_image_width,
                                                        input_image_height,
                                                        test_configuration['detection_confidence_threshold'])

    assert len(detected_classes) == expected_length


@pytest.mark.parametrize('input_image_path, expected_class_index', [
    ('./data/test_images/image_1.jpeg', 16),
    ('./data/test_images/image_2.png', 19),
    ('./data/test_images/image_3.png', 47),
])
def test_retrieve_max_confident_class_index(input_image_path: str,
                                            test_configuration: dict,
                                            test_object_detector: ObjectDetector,
                                            expected_class_index: int):

    """
    Test the function Test the function modules.object_detection.object_detection_utils.retrieve_max_confident_class_index

    Args:
        input_image_path: String image path
        test_configuration: Dictionary configuration object
        test_object_detector: ObjectDetector instance
        expected_class_index: Integer max confident class index

    Returns:
    """

    # Retrieve image dimensions
    image_width, image_height = retrieve_image_width_and_height(input_image_path)

    # Compute blob from image
    blob = read_blob_from_local_image(input_image_path,
                                      test_configuration['blob_size'],
                                      test_configuration['blob_scale_factor'],
                                      test_configuration['blob_swap_rb'],
                                      test_configuration['blob_crop'])

    # Retrieve the max confident class index
    class_index = retrieve_max_confident_class_index(image_width,
                                                     image_height,
                                                     test_object_detector.neural_network,
                                                     blob,
                                                     test_object_detector.output_layers,
                                                     test_configuration['detection_confidence_threshold'],
                                                     test_configuration['non_max_suppression_threshold'])

    assert class_index == expected_class_index


@pytest.mark.parametrize('input_image_path, expected_class', [
    ('./data/test_images/image_1.jpeg', 'dog'),
    ('./data/test_images/image_2.png', 'cow'),
    ('./data/test_images/image_3.png', 'apple'),
])
def test_detect_local_single_object(input_image_path: str,
                                    test_object_detector: ObjectDetector,
                                    expected_class: str):
    """
    Test the method 'detect_local_single_object' from the ObjectDetector class

    Args:
        input_image_path: String image path
        test_object_detector: ObjectDetector instance object
        expected_class: String expected detected class

    Returns:
    """

    # Detect the image's object class
    detected_class = test_object_detector.detect_local_single_object(input_image_path)

    assert detected_class == expected_class
    