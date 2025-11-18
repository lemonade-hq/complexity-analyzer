# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY . /app/

# Install any needed packages specified in pyproject.toml
RUN pip install --no-cache-dir .

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV GITHUB_TOKEN ""

ENTRYPOINT ["complexity-cli"]
