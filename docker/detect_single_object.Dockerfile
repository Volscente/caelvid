# Base Image
FROM python:3

# Update & Upgrade
RUN apt update -y \
    && apt upgrade -y

# Install Poetry
#RUN curl -sSL https://install.python-poetry.org | python3 -
# Check: https://stackoverflow.com/questions/53835198/integrating-python-poetry-with-docker