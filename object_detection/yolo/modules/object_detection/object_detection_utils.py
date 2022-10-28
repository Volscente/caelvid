# Import Standard Libraries
import os
import cv2

from typing import Tuple

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

    Args:
        image_path:
        size:
        scale_factor:
        swap_rb:
        crop:

    Returns:

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
