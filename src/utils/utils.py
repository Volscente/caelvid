# Import Standard Modules
import os
import yaml

# Set root path
# TODO: Remove
os.chdir(os.environ['YOLO_OBJECT_DETECTION_PATH'])

# Import Package Modules
from src.logging_module.logging_module import get_logger

# Setup logger
logger = get_logger(os.path.basename(__file__).split('.')[0])


def read_configuration(file_name: str) -> dict:
    """
    Read and return the specified configuration file from the 'configuration' folder

    Args:
        file_name: String configuration file name to read

    Returns:
        configuration: Dictionary configuration
    """

    logger.info('read_configuration - Start')

    try:

        logger.info('read_configuration - Reading {}'.format(file_name))

        # Read configuration file
        with open('./configuration/' + file_name) as config_file:

            configuration = yaml.safe_load(config_file.read())

    except FileNotFoundError:

        raise FileNotFoundError('read_data - File {} not found'.format(file_name))

    except Exception:

        raise Exception('read_configuration - Unable to read {}'.format(file_name))

    else:

        logger.info('read_configuration - Configuration file {} read successfully'.format(file_name))

    logger.info('read_configuration - End')

    return configuration

