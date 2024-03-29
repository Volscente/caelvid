# Import Standard Libraries
import os
import pathlib

import cv2
import sys
from typing import List
from typing import Tuple
from pathlib import Path
import numpy as np

from urllib.request import urlretrieve
from urllib.error import URLError

# Import Package Modules
from src.logging_module.logging_module import get_logger
from src.utils.utils import read_configuration
from src.object_detection_yolov3.object_detection_utils import read_image_from_source, retrieve_image_width_and_height, \
    read_blob_from_image, retrieve_max_confident_class_index


class ObjectDetector:
    """
    A class that implements an object detector through YOLO v3

    Attributes:
        rest_api: Boolean identifying if the image comes from a REST API
        logger: logging.Logger object for log messages
        config: Dictionary object for configuration items
        classes: List of String classes
        neural_network: cv2.dnn.Net instance of DarkNet Yolo v3
        output_layers: List of String neural network output layer names

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
        self.neural_network = None
        self.output_layers = []

        # Read YOLO v3 classes
        self.__read_classes(self.config['classes_file_path'])

        # Read Neural Network
        self.__read_neural_network(self.config['nn_weights_url'],
                                   self.config['model_weights_file_path'],
                                   self.config['model_structure_file_path'])

        # Retrieve neural network output layers
        self.output_layers = self.__get_output_layers()

    def __read_classes(self,
                       classes_file_path: Tuple[str]) -> List[str]:
        """
        Read the 'yolov3_classes.txt' file and retrieve the list of available classes

        Args:
            classes_file_path: Tuple[String] classes file path

        Returns:
            classes: List of String classes
        """

        self.logger.info('__read_classes - Start')

        try:

            self.logger.info('__read_classes - Constructing the Classes Filepath')

            # Construct the Path object from 'classes_file_path'
            classes_file_path_pathlib = Path(__file__).parents[2] / classes_file_path[0] / classes_file_path[1]

            self.logger.info('__read_classes - Reading file {}'.format(classes_file_path_pathlib))

            # Open the classes file and extract the list of available classes
            with open(classes_file_path_pathlib, 'r') as classes_file:

                classes = [line.strip() for line in classes_file.readlines()]

        except FileNotFoundError:

            raise FileNotFoundError('__read_classes - File {} not found'.format(classes_file_path_pathlib))

        else:

            self.logger.info('__read_classes - Classes file {} read successfully'.format(classes_file_path_pathlib))

        self.classes = classes

        self.logger.info('__read_classes - End')

    def __read_neural_network(self,
                              nn_weights_url: str,
                              model_weights_file_path: Tuple[str],
                              model_structure_file_path: Tuple[str]) -> cv2.dnn.Net:
        """
        Read YOLO v3 DarkNet trained neural network from the 'yolov3.weights' and 'yolov3.cfg' files.
        Download the 'yolov3.weights' file if not present

        Args:
            nn_weights_url: String URL for 'yolov3.weights'
            model_weights_file_path: Tuple[String] path for 'yolov3.weights'
            model_structure_file_path: Tuple[String] path for 'yolov3.cfg'

        Returns:
            neural_network: cv2.dnn.Net Trained Neural Network
        """

        self.logger.info('__read_neural_network - Start')

        # Retrieve yolov3.weights if not present
        try:

            self.logger.info('__read_neural_network - Constructing the Model Weights Filepath')

            # Construct the Path object from 'model_weights_file_path'
            model_weights_file_path_pathlib = Path(__file__).parents[2] / model_weights_file_path[0] / \
                model_weights_file_path[1]

            self.logger.info(
                '__read_neural_network - Checking if the file {} is already downloaded'.format(
                    model_weights_file_path_pathlib))

            # Check whatever the 'yolov3.weights' file is not present and download it
            if not os.path.isfile(model_weights_file_path_pathlib):

                # Download 'yolov3.weights'
                urlretrieve(nn_weights_url, model_weights_file_path_pathlib)

                self.logger.info(
                    '__read_neural_network - File {} download completed'.format(model_weights_file_path_pathlib))

            else:

                self.logger.info(
                    '__read_neural_network - File {} already downloaded'.format(model_weights_file_path_pathlib))

        except FileNotFoundError as e:

            raise FileNotFoundError('__read_neural_network - File {} not found'.format(model_weights_file_path_pathlib))

        except URLError:

            raise URLError('__read_neural_network - Unable to reach the URL'.format(nn_weights_url))

        else:

            self.logger.info(
                '__read_neural_network - File {} is in the File System'.format(model_weights_file_path_pathlib))

        self.logger.info('__read_neural_network - Instancing Neural Network')

        self.logger.info('__read_neural_network - Constructing the Model Structure Filepath')

        # Construct the Path object from 'model_weights_file_path'
        model_structure_file_path_pathlib = Path(__file__).parents[2] / model_structure_file_path[0] / \
            model_structure_file_path[1]

        # Read pre-trained model and configuration file if the required files are available
        if os.path.isfile(model_weights_file_path_pathlib) and os.path.isfile(model_structure_file_path_pathlib):

            # Define the Neural Network
            neural_network = cv2.dnn.readNetFromDarknet(model_structure_file_path_pathlib.as_posix(),
                                                        model_weights_file_path_pathlib.as_posix())

            # Set the Neural Network computation backend
            neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

        else:

            raise FileNotFoundError('__read_neural_network - Missing required files: yolov3.weights and yolov3.cfg')

        self.logger.info('__read_neural_network - Neural Network file read successfully')

        self.neural_network = neural_network

        self.logger.info('__read_neural_network - End')

    def __get_output_layers(self) -> List[str]:
        """
        Retrieve the list of output layers names

        Returns:
            output_layers: List of output layers names
        """

        self.logger.info('__get_out_layers - Start')

        try:

            self.logger.info('__get_out_layers - Retrieving layers names')

            # Retrieve layer's names
            layer_names = self.neural_network.getLayerNames()

        except Exception:

            raise Exception('__get_out_layers - Unable to retrieve layers names')

        else:

            self.logger.info('__get_out_layers - Successfully retrieved layers names')

        self.logger.info('__get_out_layers - Computing the output layers names')

        # Get output layers names since by the non-output connected ones
        output_layers = [layer_names[i - 1] for i in self.neural_network.getUnconnectedOutLayers()]

        self.logger.info('__get_out_layers - End')

        return output_layers

    def detect_single_object(self,
                             image_source: str | np.ndarray | pathlib.PosixPath) -> str:
        """
        Detect the class of the input image

        Args:
            image_source: String image path from local File System | Numpy.ndarray image representation | pathlib.PosixPath object

        Returns:
            detected_class: String detected class name
        """

        self.logger.info('detect_single_object - Start')

        self.logger.info('detect_single_object - Reading the image from the source')

        # Read the image from a local file path (String or pathlib.PosixPath) or from a Numpy.ndarray image representation
        image = read_image_from_source(image_source)

        # Retrieve image dimensions
        image_width, image_height = retrieve_image_width_and_height(image)

        # Retrieve blobFromImage parameters
        size = self.config['blob_size']
        scale_factor = self.config['blob_scale_factor']
        swap_rb = self.config['blob_swap_rb']
        crop = self.config['blob_crop']

        self.logger.info('detect_single_object - Creating Blob from image')

        # Retrieve Blob image from local file
        blob = read_blob_from_image(image,
                                    size,
                                    scale_factor,
                                    swap_rb,
                                    crop)

        self.logger.info('detect_single_object - Blob from image successfully created')

        self.logger.info('detect_single_object - Retrieving the class with the max confident detection level')

        # Retrieve class index with mac confidence level
        class_index = retrieve_max_confident_class_index(image_width,
                                                         image_height,
                                                         self.neural_network,
                                                         blob,
                                                         self.output_layers,
                                                         self.config['detection_confidence_threshold'],
                                                         self.config['non_max_suppression_threshold'])

        self.logger.info(
            'detect_single_object - Successfully retrieved the class with the max confident detection level')

        # Retrieve class
        detected_class = self.classes[class_index]

        self.logger.info('detect_single_object - End')

        return detected_class
