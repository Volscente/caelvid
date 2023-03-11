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