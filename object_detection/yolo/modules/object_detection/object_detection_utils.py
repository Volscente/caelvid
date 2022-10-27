# Import Standard Libraries
import os

from typing import Tuple

# Set root path
os.chdir(os.environ['YOLO_OBJECT_DETECTION_PATH'])

# Import Package Libraries
from modules.logging_module.logging_module import get_logger


def read_blob_from_local_image(image: str,
                               size: Tuple[int, int],
                               scale_factor: float,
                               swap_rb: bool,
                               crop: bool):

    logger.info('read_blob_from_local_image - Start')

def read_blob_from_rest_api_image():

    pass