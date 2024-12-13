#!/bin/bash

echo "Worker Initiated"

echo "Starting ComfyUI API"
python3 /app/ComfyUI/main.py --listen "0.0.0.0" --front-end-version "Comfy-Org/ComfyUI_frontend@latest" &

echo "Starting RunPod Handler"
python3 -u "/app/rp_handler.py"
