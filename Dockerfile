# Stage 1: Base image with common dependencies
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
	python3-dev python3-pip pipx git wget \
	zip unzip libgl1 \
	&& ln -sf /usr/bin/python3.12 /usr/bin/python \
	&& ln -sf /usr/bin/pip3 /usr/bin/pip

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# COPY ./conf/pip.conf ~/.config/pip/pip.conf

RUN rm /usr/lib/python3.12/EXTERNALLY-MANAGED

WORKDIR /app

# Install comfy-cli
RUN python3 -m pip install --break-system-packages \
	comfy-cli

# Install ComfyUI
RUN /usr/bin/yes | comfy --workspace /app/ComfyUI install --cuda-version 12.1 --nvidia --version latest

# Change working directory to ComfyUI
WORKDIR /app/ComfyUI

# Install runpod
# RUN pip install runpod requests

# Go back to the root
# WORKDIR /

# Add scripts
# ADD src/start.sh src/restore_snapshot.sh src/rp_handler.py test_input.json ./
# RUN chmod +x /start.sh /restore_snapshot.sh

# Optionally copy the snapshot file
# ADD *snapshot*.json /

# Restore the snapshot to install custom nodes
# RUN /restore_snapshot.sh

# Create necessary directories
RUN mkdir -p models/checkpoints models/vae models/diffusion_models models/clip

# COPY ./mochi_model[s]/mochi1PreviewVideo_previewBF16.safetensor[s] /comfyui/models/diffusion_models/
# COPY ./mochi_model[s]/mochi1PreviewVideo_vae.safetensor[s] /comfyui/models/vae/
# COPY ./mochi_model[s]/stableDiffusion3SD3_textEncoderT5XXLFP16.safetensor[s] /comfyui/models/clip/

EXPOSE 8188 9966

# Start the container
CMD ["comfy","launch","--background","--listen","0.0.0.0","--port","9966"]
