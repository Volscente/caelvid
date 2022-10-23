# Import Standard Libraries
import os

# Set root path
os.chdir(os.environ['YOLO_OBJECT_DETECTION_PATH'])

# Import Package Libraries
from modules.logging_module.logging_module import get_logger
from modules.utils.utils import read_configuration


class ObjectDetector:
    """
    A class that implements an object detector through YOLO v3

    Attributes:

    Methods:

    """

    def __init__(self, configuration_file='config.yaml'):
        """
        Initialise a ObjectDetector object for perform object detection operations

        Args:
            configuration_file: String of configuration file name
        """

        # Setup Logger
        self.logger = get_logger(__class__.__name__)
        self.logger.info('__init__ - Instancing the class')

        self.logger.info('__init__ - Read configuration file')

        # Read Configuration file
        self.config = read_configuration(configuration_file)

        #
