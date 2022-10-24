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
        logger: logging.Logger object for log messages
        config: Dictionary object for configuration items
    Methods:

    """

    def __init__(self,
                 configuration_file: str = 'config.yaml'):
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

        # Initialise instance variables
        self.classes = []

        # Read YOLO v3 classes
        self.__read_classes()

    def __read_classes(self):

        self.logger.info('__read_classes - Start')

        try:

            # Open the classes file and extract the list of available classes
            with open(self.config['classes_file_path'], 'r') as classes_file:

                self.classes = [line.strip() for line in classes_file.readlines()]

        except FileNotFoundError as e:

            self.logger.error('__read_classes - File {} not found'.format(self.config['classes_file_path']))
            self.logger.error(e)
            raise FileNotFoundError

        except Exception as e:

            self.logger.error('__read_classes - Unable to read {}'.format(self.config['classes_file_path']))
            self.logger.error(e)
            raise e

        else:

            self.logger.info('__read_classes - Classes file {} read successfully'.format(self.config['classes_file_path']))

        finally:

            self.logger.info('__read_classes - End')
