#!/usr/bin/env bash
# This file tags and uploads an image to Docker Hub

# Assumes that an image is built via `run_docker.sh`

# Step 1:
# Create dockerpath
dockerpath=coil/stock_analytics

# Step 2:  
# Authenticate & tag
echo "Docker ID and Image: $dockerpath"

cat ./my_password.txt | docker login --username coil --password-stdin

docker tag stock_analytics ${dockerpath}:latest
docker commit stock_analytics ${dockerpath}:latest

# Step 3:
# Push image to a docker repository
docker push ${dockerpath}:latest
