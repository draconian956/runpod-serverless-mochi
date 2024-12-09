#!/bin/bash

echo "Worker Initiated"

echo "Starting WebUI API"
python /stable-diffusion-webui/webui.py --skip-python-version-check --skip-torch-cuda-test --skip-install --ckpt /stable-diffusion-webui/models/Stable-diffusion/turbovisionxlSuperFastXLBasedOnNew_tvxlV431Bakedvae.safetensors --opt-sdp-attention --disable-safe-unpickle --port 3000 --api --nowebui --skip-version-check --xformers --no-hashing --no-download-sd-model &

echo "Starting RunPod Handler"
python -u /rp_handler.py
