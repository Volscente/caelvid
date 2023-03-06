# Base Image
FROM python:3.12.0a3-slim-bullseye

# Update & Upgrade
RUN apt update -y \
    && apt upgrade -y

# Install curl
RUN apt install curl -y

# Install required libraries for GCC
RUN apt install build-essential -y \
    && apt install manpages-dev -y