# Import Standard Libraries
import os
import cv2

from typing import Tuple, List

# Set root path
import numpy as np

os.chdir(os.environ['YOLO_OBJECT_DETECTION_PATH'])

# Import Package Libraries
from modules.logging_module.logging_module import get_logger

# Setup logger
logger = get_logger(os.path.basename(__file__).split('.')[0])


def retrieve_image_width_and_height(image_path: str) -> Tuple[int, int]:
    """
    Read the image with OpenCV and retrieve Width and Height of the image

    Args:
        image_path: String image path

    Returns:
        image_width: Integer width of the image
        image_height: Integer height of the image
    """

    logger.info('retrieve_image_width_and_height - Start')

    try:

        logger.info('retrieve_image_width_and_height - Reading the image')

        image = cv2.imread(image_path)

    except FileNotFoundError as e:

        logger.error('retrieve_image_width_and_height - Unable to find the image {}'.format(image_path))
        logger.error(e)
        raise FileNotFoundError

    else:

        logger.info('retrieve_image_width_and_height - Image successfully read')

    finally:

        # Compute image's width and height
        image_width = image.shape[1]
        image_height = image.shape[0]

        logger.info('retrieve_image_width_and_height - End')

        return image_width, image_height


def read_blob_from_local_image(image_path: str,
                               size: List[int],
                               scale_factor: float,
                               swap_rb: bool,
                               crop: bool) -> np.ndarray:
    """
    Create a 4-dimensional (images, channels, width, height) Blob from a OpenCV image

    Args:
        image_path: String image path
        size: List integer resize dimensions
        scale_factor: Float pixel scale factor
        swap_rb: Bool flag for swapping R channel with B channel
        crop: Bool flag to crop the image

    Returns:
        blob_image: numpy.ndarray Blob
    """

    logger.info('read_blob_from_local_image - Start')

    try:

        logger.info('read_blob_from_local_image - Reading image from {}'.format(image_path))

        # Read image through Open CV as a 3-D Numpy ndarray
        image = cv2.imread(image_path)

        logger.info('read_blob_from_local_image - Creating Blob from the image')

        # Create a 4-dimensional (images, channels, width, height) Blob from an image
        blob_image = cv2.dnn.blobFromImage(image=image,
                                           size=(size[0], size[1]),
                                           scalefactor=float(scale_factor),
                                           swapRB=swap_rb,
                                           crop=crop)

    except FileNotFoundError as e:

        logger.error('read_blob_from_local_image - Unable to find the image {}'.format(image_path))
        logger.error(e)
        raise FileNotFoundError

    else:

        logger.info('read_blob_from_local_image - Blob from the image successfully created')

    finally:

        logger.info('read_blob_from_local_image - End')

        return blob_image


def read_blob_from_rest_api_image():
    pass


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

    except Exception as e:

        logger.error('retrieve_neural_network_output - Unable to set neural network input')
        logger.error(e)
        sys.exit(1)

    else:

        logger.info('retrieve_neural_network_output - Successfully set neural network input')

    logger.info('retrieve_neural_network_output - Feed forwarding the input through the Neural Network')

    try:

        # Compute the model's output of only the three output layers
        outputs = neural_network.forward(output_layers)

    except Exception as e:

        logger.error('retrieve_neural_network_output - Unable to feed forward the input through the Neural Network')
        logger.error(e)
        sys.exit(1)

    else:

        logger.info('retrieve_neural_network_output - Successfully retrieve neural network outputs')
        
    finally:
        
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


# TODO retrieve_max_confident_class_index
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

    except Exception as e:

        logger.error('retrieve_max_confident_class_index - Unable to retrieve detected classes')
        logger.error(e)
        sys.exit(1)

    else:

        logger.info('retrieve_max_confident_class_index - Successfully retrieved all detected classes')

    try:

        logger.info('retrieve_max_confident_class_index - Applying Non-max Suppression')

        # Apply Non-Max Suppression
        class_indices = cv2.dnn.NMSBoxes(detected_boxes,
                                         detected_confidences,
                                         detection_confidence_threshold,
                                         non_max_suppression_threshold)

    except Exception as e:

        logger.error('retrieve_max_confident_class_index - Unable to apply Non-max Suppression')
        logger.error(e)
        sys.exit(1)

    else:

        logger.info('retrieve_max_confident_class_index - Successfully applied Non-max Suppression')

    finally:

        logger.info('retrieve_max_confident_class_index - End')

        return detected_classes[class_indices[0]]
