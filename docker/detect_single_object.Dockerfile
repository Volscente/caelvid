# Base Image
FROM python:3

# Accept ENVIRONMENT argument for distinguishing between 'production' and 'test'
ARG ENVIRONMENT

# Set Environment Variables
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