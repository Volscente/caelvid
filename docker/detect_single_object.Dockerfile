# Base Image
FROM python:3.10.5

# Accept ENVIRONMENT argument for distinguishing between 'production' and 'test'
ARG ENVIRONMENT

# Set Environment Variables
ENV ENVIRONMENT=${ENVIRONMENT} \ 
    # Enable the Python Fault Handler to dump Python tracebacks
    PYTHONFAULTHANDLER=1 \
    # Stream the stdout and stderr directly into the terminal
    PYTHONUNBUFFERED=1 \
    # Set a random value for the hash seed secret
    PYTHONHASHSEED=random \
    # Do not store any pip installation in the cache
    PIP_NO_CACHE_DIR=off \
    # Do not check for pip version
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    # Pip command timeout
    PIP_DEFAULT_TIMEOUT=100 \
    # Specify Poetry version
    POETRY_VERSION=1.3.2

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Install OpenCV required libraries
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Change working directory
WORKDIR /detect_single_object

# Copy required poetry files into the container
COPY poetry.lock pyproject.toml /detect_single_object/

# Copy the code
COPY . /detect_single_object

# Install dependencies
RUN poetry install --without dev --no-interaction --no-ansi

 poetry run uvicorn src.object_detection_yolov3.object_detection_rest_api:app --reload