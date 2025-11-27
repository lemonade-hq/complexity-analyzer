FROM public.ecr.aws/docker/library/python:3.14.0-slim

WORKDIR /app

COPY . /app/

# Install any needed packages specified in pyproject.toml
RUN pip install --no-cache-dir .

EXPOSE 80

# Define environment variable
ENV GITHUB_TOKEN ""

ENTRYPOINT ["complexity-cli"]
