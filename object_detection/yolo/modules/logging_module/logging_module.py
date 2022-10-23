# Import Standard Libraries
import logging.config
import yaml
import os

# Set root path
os.chdir(os.environ['YOLO_OBJECT_DETECTION_PATH'])


def get_logger(logger_name: str) -> logging.Logger:
    """
    Set the configuration for the logging module and return the requested logger

    Args:
        logger_name: String name of the logger to retrieve from 'log_configuration.yaml' fie

    Returns:
        logging.Logger logger object
    """

    try:

        # Read the log_configuration file
        with open('./configuration/log_configuration.yaml', 'r') as file:
            log_config = yaml.safe_load(file.read())
            logging.config.dictConfig(log_config)

        # Retrieve the requested logger
        logger = logging.getLogger(logger_name)

    except FileNotFoundError:

        raise FileNotFoundError

    return logger
