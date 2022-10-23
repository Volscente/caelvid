# Import Standard Libraries
import os

# Set root path
os.chdir(os.environ['YOLO_OBJECT_DETECTION_PATH'])

# Import Package Libraries
from modules.logging_module.logging_module import get_logger


class ObjectDetector:
    """
    A class that implements an object detector through YOLO v3

    Attributes:

    Methods:

    """

    def __init__(self):

        # Setup Logger
        self.logger = get_logger(__class__.__name__)
        self.logger.info('__init__ - Instancing the class')
