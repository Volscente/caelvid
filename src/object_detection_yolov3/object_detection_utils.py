# Import Standard Libraries
import os
import sys
import cv2
from typing import Tuple, List
import pathlib
from pathlib import Path
import numpy as np

# Import Package Modules
from src.logging_module.logging_module import get_logger

# Setup logger
logger = get_logger(os.path.basename(__file__).split('.')[0])


def read_image_from_source(image_source: str | np.ndarray | pathlib.PosixPath) -> np.ndarray:
    """
    Read the image with Open CV from the 'image_source'

    Args:
        image_source: String image path from local File System | Numpy.ndarray image representation | pathlib.PosixPath object

    Returns:
        image: Numpy.ndarray OpenCV read image
    """

    logger.info('read_image_from_source - Start')

    logger.info('read_image_from_source - Reading image from source')

    logger.info('read_image_from_source - Type of source: {}'.format(str(type(image_source))))

    # Switch between reading from local file (str or pathlib.PosixPath or from Numpy.ndarray image representation
    if type(image_source) == str or type(image_source) == pathlib.PosixPath:

        # Check if the image file is in the FS
        if os.path.isfile(image_source):

            # Switch between String and pathlib.PosixPath
            if type(image_source) == str:

                # Read image from local file through String path
                image = cv2.imread(image_source)

            else:

                # Read image from local file through pathlib.PosixPath path
                image = cv2.imread(image_source.as_posix())

        else:

            raise FileNotFoundError('read_image_from_source - Unable to find the image at {}'.format(image_source))

    elif type(image_source) == np.ndarray:

        # Read image from Numpy.ndarray image representation
        image_4_channels = cv2.imdecode(image_source,
                                        cv2.IMREAD_UNCHANGED)

        # Convert the image from 4 channels to 3 channels
        image = cv2.cvtColor(image_4_channels,
                             cv2.COLOR_BGRA2BGR)

    else:

        raise TypeError('read_image_from_source - Image source type must be String, Numpy.ndarray or pathlib.PosixPath')

    logger.info('read_image_from_source - Successfully read image from the source')

    logger.info('read_image_from_source - Image Shape: {}'.format(image.shape))

    logger.info('read_image_from_source - End')

    return image


def retrieve_image_width_and_height(image: np.ndarray) -> Tuple[int, int]:
    """
    Retrieve Width and Height of the image

    Args:
        image: Numpy.ndarray image representation

    Returns:
        image_width: Integer width of the image
        image_height: Integer height of the image
    """

    logger.info('retrieve_image_width_and_height - Start')

    logger.info('retrieve_image_width_and_height - Retrieving image width and height')

    # Retrieve image's width and height
    image_width = image.shape[1]
    image_height = image.shape[0]

    logger.info('retrieve_image_width_and_height - Successfully retrieved image width and height')

    logger.info('retrieve_image_width_and_height - End')

    return image_width, image_height


def read_blob_from_image(image: np.ndarray,
                         size: List[int],
                         scale_factor: float,
                         swap_rb: bool,
                         crop: bool) -> np.ndarray:
    """
    Create a 4-dimensional (images, channels, width, height) Blob from a OpenCV image

    Args:
        image: Numpy.ndarray image representation
        size: List integer resize dimensions
        scale_factor: Float pixel scale factor
        swap_rb: Bool flag for swapping R channel with B channel
        crop: Bool flag to crop the image

    Returns:
        blob_image: Numpy.ndarray Blob
    """

    logger.info('read_blob_from_image - Start')

    logger.info('read_blob_from_image - Creating Blob from the image')

    # Create a 4-dimensional (images, channels, width, height) Blob from an image
    blob_image = cv2.dnn.blobFromImage(image=image,
                                       size=(size[0], size[1]),
                                       scalefactor=float(scale_factor),
                                       swapRB=swap_rb,
                                       crop=crop)

    logger.info('read_blob_from_image - Blob from the image successfully created')

    logger.info('read_blob_from_image - End')

    return blob_image


def retrieve_neural_network_output(neural_network: cv2.dnn.Net,
                                   blob: np.ndarray,
                                   output_layers: List[str]) -> Tuple[List, List, List]:
    """
    Set the blob as the neural network input and feed forward through it to retrieve the 3 output nodes predictions

    Args:
        neural_network: cv2.dnn.Net instance of DarkNet Yolo v3
        blob: numpy.ndarray Blob of the image
        output_layers: List of output layer names

    Returns:
        outputs: Tuple of output layers predictions (3 Lists of predictions)
    """

    logger.info('retrieve_neural_network_output - Start')

    logger.info('retrieve_neural_network_output - Setting neural network input')

    try:

        # Set Blob as the neural network input
        neural_network.setInput(blob)

    except Exception:

        raise Exception('retrieve_neural_network_output - Unable to set neural network input')

    else:

        logger.info('retrieve_neural_network_output - Successfully set neural network input')

    logger.info('retrieve_neural_network_output - Feed forwarding the input through the Neural Network')

    try:

        # Compute the model's output of only the three output layers
        outputs = neural_network.forward(output_layers)

    except Exception:

        raise Exception('retrieve_neural_network_output - Unable to feed forward the input through the Neural Network')

    else:

        logger.info('retrieve_neural_network_output - Successfully retrieve neural network outputs')

    logger.info('retrieve_neural_network_output - End')

    return outputs


def retrieve_all_detected_classes(outputs: Tuple[List, List, List],
                                  image_width: int,
                                  image_height: int,
                                  detection_confidence_threshold: float) -> Tuple[List[int],
                                                                                  List[float],
                                                                                  List[List[int]]]:
    """
    Retrieve all the detected class with a detection confidence grater than detection_confidence_threshold

    Args:
        outputs: Tuple[List, List, List] outputs of the Neural Network
        image_width: Integer image width
        image_height: Integer image height
        detection_confidence_threshold: Float detection confidence threshold for discarding low confidence detections

    Returns:
        detected_classes: List[int] detected class indices
        detected_confidences: List[float] detected class confidence levels
        detected_boxes: List[List[int, int, int, int]] detected boxes
    """

    logger.info('retrieve_all_detected_classes - Start')

    # Initialise the list of detected classes and confidence levels
    detected_classes, detected_confidences, detected_boxes = [], [], []

    logger.info('retrieve_all_detected_classes - Fetching outputs')

    # Fetch all the three output layers outputs
    for index, output in enumerate(outputs):

        logger.info('retrieve_all_detected_classes - Fetch output layers {}'.format(index + 1))

        # Fetch all the boxes detected from the output layers
        for detection in output:

            # Retrieve the score of the detection for each class (First 4 values are the box coordinates)
            scores = detection[5:]

            # Get the maximum score, which corresponds to the detected class
            detected_class = np.argmax(scores)

            # Retrieve the detected confidence level
            detected_confidence = scores[detected_class]

            # Check if the confidence is greater than the threshold
            if detected_confidence > detection_confidence_threshold:
                # Retrieve box coordinates
                center_x = int(detection[0] * image_width)
                center_y = int(detection[1] * image_height)
                w = int(detection[2] * image_width)
                h = int(detection[3] * image_height)
                x = center_x - w / 2
                y = center_y - h / 2

                # Update detected_classes. detected_confidences and detected_boxes
                detected_classes.append(detected_class)
                detected_confidences.append(detected_confidence)
                detected_boxes.append([x, y, w, h])

    logger.info('retrieve_all_detected_classes - Total number of detected boxes {}'.format(len(detected_classes)))

    logger.info('retrieve_all_detected_classes - End')

    return detected_classes, detected_confidences, detected_boxes


def retrieve_max_confident_class_index(image_width: int,
                                       image_height: int,
                                       neural_network: cv2.dnn.Net,
                                       blob: np.ndarray,
                                       output_layers: List[str],
                                       detection_confidence_threshold: float,
                                       non_max_suppression_threshold: float) -> int:
    """
    Retrieve the class index with the max confident detection level in the Blob

    Args:
        image_width: Integer image width
        image_height: Integer image height
        neural_network: cv2.dnn.Net DarkNet OpenCV instance
        blob: Numpy.ndarray Blob of the image
        output_layers: List String
        detection_confidence_threshold: Float detection confidence threshold for discarding low confidence detections
        non_max_suppression_threshold: Float threshold for discarding boxes during the Non-max Suppression step

    Returns:
        class_index: Integer class index
    """

    logger.info('retrieve_max_confident_class_index - Start')

    logger.info('retrieve_max_confident_class_index - Computing neural network outputs')

    # Retrieve neural network outputs for the forwarded input (blob)
    outputs = retrieve_neural_network_output(neural_network,
                                             blob,
                                             output_layers)

    try:

        logger.info('retrieve_max_confident_class_index - Retrieving all detected classes')

        # Retrieve all the detected class indices coming from all the three output layers
        detected_classes, detected_confidences, detected_boxes = retrieve_all_detected_classes(outputs,
                                                                                               image_width,
                                                                                               image_height,
                                                                                               detection_confidence_threshold)

    except Exception:

        raise Exception('retrieve_max_confident_class_index - Unable to retrieve detected classes')

    else:

        logger.info('retrieve_max_confident_class_index - Successfully retrieved all detected classes')

    try:

        logger.info('retrieve_max_confident_class_index - Applying Non-max Suppression')

        # Apply Non-Max Suppression
        class_indices = cv2.dnn.NMSBoxes(detected_boxes,
                                         detected_confidences,
                                         detection_confidence_threshold,
                                         non_max_suppression_threshold)

    except Exception:

        raise Exception('retrieve_max_confident_class_index - Unable to apply Non-max Suppression')

    else:

        logger.info('retrieve_max_confident_class_index - Successfully applied Non-max Suppression')

    logger.info('retrieve_max_confident_class_index - End')

    return detected_classes[class_indices[0]]
