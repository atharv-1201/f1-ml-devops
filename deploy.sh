#!/usr/bin/env bash
set -e

APP_NAME="f1-predictor"
PORT=8000

echo "🏎️ Deploying ${APP_NAME} on Docker..."

# Build container
docker build -t ${APP_NAME}:latest .

# Stop previous container if running
if [ $(docker ps -q -f name=${APP_NAME}) ]; then
    echo "Stopping running container..."
    docker stop ${APP_NAME}
fi

# Remove old container if exists
if [ $(docker ps -aq -f name=${APP_NAME}) ]; then
    echo "Removing old container..."
    docker rm ${APP_NAME}
fi

# Run new container
echo "Starting new container on port ${PORT}..."
docker run -d \
    --name ${APP_NAME} \
    --restart always \
    -p ${PORT}:8000 \
    ${APP_NAME}:latest

# Health check
echo "Waiting for service to become healthy..."
sleep 5
curl -f http://localhost:${PORT}/health

echo ""
echo "✅ Deployment successful! Service running at http://localhost:${PORT}"
