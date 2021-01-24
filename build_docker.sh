#!/usr/bin/env bash

# Step 1:
# Build image and add a descriptive tag
# dont forget about dot at the end
docker build --tag=stock_analytics .

# Step 2: 
# List docker images
docker image ls

