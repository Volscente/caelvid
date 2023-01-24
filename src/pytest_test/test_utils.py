# Import Standard Modules
import os
import pytest

# Import Package Modules
from src.pytest_test.test_fixtures import test_object_detector
from src.object_detection_yolov3.object_detection import ObjectDetector
from src.utils.utils import read_configuration


@pytest.mark.parametrize('test_config_file, test_config, expected_value', [
    ('config.yaml', 'classes_file_path', './../../classes/yolov3_classes.txt'),
    ('config.yaml', 'model_structure_file_path', './../../models/yolov3.cfg'),
    ('config.yaml', 'model_weights_file_path', './../../models/yolov3.weights'),
    ('config.yaml', 'test_data_path', './../../data/test_images/')
])
def test_read_configuration(test_config_file: str,
                            test_config: str,
                            expected_value: str):
    """
    Test the function src.utils.utils.read_configuration

    Args:
        test_config_file: String configuration file name
        test_config: String configuration entry key
        expected_value: String configuration expected value

    Returns:
    """

    # Read configuration file
    config = read_configuration(test_config_file)

    assert config[test_config] == expected_value


@pytest.mark.parametrize('test_config_file, expected_error', [
    ('wrong_config.config', FileNotFoundError)
])
def test_read_configuration_exception(test_config_file: str,
                                      expected_error: FileNotFoundError):
    """
    Test the exceptions to the function src.utils.utils.read_configuration

    Args:
        test_config_file: String configuration file name
        expected_error: Exception instance

    Returns:
    """

    with pytest.raises(expected_error):

        read_configuration(test_config_file)