#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting deployment process..."

# Stop and remove all running containers
echo "Stopping and removing all running containers..."
sudo docker ps -q | xargs -r docker stop
sudo docker ps -aq | xargs -r docker rm

echo "Removing all Docker images..."
sudo docker images -q | xargs -r docker rmi -f


# Pull the latest image from the repository
REPO_NAME="h0w6o2c0/weatherpredictor"
IMAGE_TAG="latest"
echo "Pulling the latest image: public.ecr.aws/${REPO_NAME}:${IMAGE_TAG}..."
sudo docker pull public.ecr.aws/${REPO_NAME}:${IMAGE_TAG}

# Run the new container
CONTAINER_NAME="my-application"
echo "Deploying the new container: ${CONTAINER_NAME}..."
sudo docker run -d --name ${CONTAINER_NAME} -p 8501:8501 public.ecr.aws/${REPO_NAME}:${IMAGE_TAG}

echo "Deployment completed successfully!"