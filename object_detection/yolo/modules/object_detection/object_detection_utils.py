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


def read_blob_from_local_image(image_path: str,
                               size: Tuple[int, int],
                               scale_factor: float,
                               swap_rb: bool,
                               crop: bool) -> np.ndarray:
    """
    Create a 4-dimensional (images, channels, width, height) Blob from a OpenCV image

    Args:
        image_path: String image path
        size: Tuple integer resize dimensions
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
                                           size=size,
                                           scalefactor=scale_factor,
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
                                   output_layers: List) -> Tuple[List, List, List]:
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
