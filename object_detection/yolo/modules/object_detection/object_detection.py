# Import Standard Libraries
import os
import cv2
import sys
from typing import List

from urllib.request import urlretrieve
from urllib.error import URLError

# Set root path
os.chdir(os.environ['YOLO_OBJECT_DETECTION_PATH'])

# Import Package Libraries
from modules.logging_module.logging_module import get_logger
from modules.utils.utils import read_configuration


class ObjectDetector:
    """
    A class that implements an object detector through YOLO v3

    Attributes:
        rest_api: Boolean identifying if the image comes from a REST API
        logger: logging.Logger object for log messages
        config: Dictionary object for configuration items
        classes: List of String classes
        neural_network: cv2.dnn.Net instance of DarkNet Yolo v3

    Methods:

    """

    def __init__(self,
                 rest_api: bool = False,
                 configuration_file: str = 'config.yaml'):
        """
        Initialise a ObjectDetector object for perform object detection operations

        Args:
            configuration_file: String of configuration file name
        """

        # Setup Logger
        self.logger = get_logger(__class__.__name__)
        self.logger.info('__init__ - Instancing the class')

        # Set rest_api
        self.rest_api = rest_api

        self.logger.info('__init__ - Read configuration file')

        # Read Configuration file
        self.config = read_configuration(configuration_file)

        # Initialise instance variables
        self.classes = []
        self.neural_network = None

        # Read YOLO v3 classes
        self.__read_classes(self.config['classes_file_path'])

        # Read Neural Network
        self.__read_neural_network(self.config['nn_weights_url'],
                                   self.config['model_weights_file_path'],
                                   self.config['model_structure_file_path'])

        # Retrieve neural network layers
        self.neural_network_layers = self.neural_network.getLayerNames()

    def __read_classes(self,
                       classes_file_path: str) -> List:
        """
        Read the 'yolov3_classes.txt' file and retrieve the list of available classes

        Args:
            classes_file_path: String classes file path

        Returns:
            classes: List of String classes
        """

        self.logger.info('__read_classes - Start')

        try:

            self.logger.info('__read_classes - Reading file {}'.format(classes_file_path))

            # Open the classes file and extract the list of available classes
            with open(classes_file_path, 'r') as classes_file:

                classes = [line.strip() for line in classes_file.readlines()]

        except FileNotFoundError as e:

            self.logger.error('__read_classes - File {} not found'.format(classes_file_path))
            self.logger.error(e)
            raise FileNotFoundError

        else:

            self.logger.info('__read_classes - Classes file {} read successfully'.format(classes_file_path))

        finally:

            self.classes = classes

            self.logger.info('__read_classes - End')

    def __read_neural_network(self,
                              nn_weights_url: str,
                              model_weights_file_path: str,
                              model_structure_file_path: str) -> cv2.dnn.Net:
        """
        Read YOLO v3 DarkNet trained neural network from the 'yolov3.weights' and 'yolov3.cfg' files.
        Download the 'yolov3.weights' file if not present

        Args:
            nn_weights_url: String URL for 'yolov3.weights'
            model_weights_file_path: String path for 'yolov3.weights'
            model_structure_file_path: String path for 'yolov3.cfg'

        Returns:
            neural_network: cv2.dnn.Net Trained Neural Network
        """

        self.logger.info('__read_neural_network - Start')

        # Retrieve yolov3.weights if not present
        try:

            self.logger.info('__read_neural_network - Checking if the file {} is already downloaded'.format(model_weights_file_path))

            # Check whatever the 'yolov3.weights' file is not present and download it
            if not os.path.isfile(model_weights_file_path):

                # Download 'yolov3.weights'
                urlretrieve(nn_weights_url, model_weights_file_path)

                self.logger.info('__read_neural_network - File {} download completed'.format(model_weights_file_path))

            else:

                self.logger.info('__read_neural_network - File {} already downloaded'.format(model_weights_file_path))

        except FileNotFoundError as e:

            self.logger.error('__read_neural_network - File {} not found'.format(model_weights_file_path))
            self.logger.error(e)
            raise FileNotFoundError

        except URLError as e:

            self.logger.error('__read_neural_network - Unable to reach the URL'.format(nn_weights_url))
            self.logger.error(e)
            raise URLError

        else:

            self.logger.info('__read_neural_network - File {} is in the File System'.format(model_weights_file_path))

        self.logger.info('__read_neural_network - Instancing Neural Network')

        # Read pr-trained model and configuration file if the required files are available
        if os.path.isfile(model_weights_file_path) and os.path.isfile(model_structure_file_path):

            # Define the Neural Network
            neural_network = cv2.dnn.readNetFromDarknet(model_structure_file_path, model_weights_file_path)

            # Set the Neural Network computation backend
            neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

        else:

            self.logger.error('__read_neural_network - Missing required files: yolov3.weights and yolov3.cfg')
            raise FileNotFoundError

        self.logger.info('__read_neural_network - Neural Network file read successfully')

        self.neural_network = neural_network

        self.logger.info('__read_neural_network - End')
