#!/usr/bin/env bash

## Complete the following steps to get Docker running locally

# Step 1:
# Build image and add a descriptive tag
# dont forget about dot at the end
docker build --tag=stock_analytics:latest .

# Step 2:
# List docker images
docker image ls

# Step 3:
# Run flask app
docker run -p 80:80 stock_analytics
