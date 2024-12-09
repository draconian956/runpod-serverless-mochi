# Stage 1: Base image with common dependencies
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04
# FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# 不要檢查 CUDA 版本
ENV NVIDIA_DISABLE_REQUIRE=1

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
	python3-dev python3-pip pipx git wget \
	zip unzip libgl1

# Clean up to reduce image size
RUN apt-get autoremove -y && \
	apt-get clean -y && \
	rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install comfy-cli
RUN --mount=type=cache,target=/root/.cache/pip \
	python3 -m pip install \
	comfy-cli

# Install ComfyUI
# RUN /usr/bin/yes | comfy --workspace /app/ComfyUI install --cuda-version 12.1 --nvidia --version latest

# RUN git clone https://github.com/comfyanonymous/ComfyUI.git ComfyUI
COPY ./ComfyUI /app/ComfyUI

RUN cd ./ComfyUI && \
	--mount=type=cache,target=/root/.cache/pip \
	python3 -m pip install -r requirements.txt --no-cache-dir

# Change working directory to ComfyUI
WORKDIR /app/ComfyUI

# RUN cd ./custom_nodes && \
# 	git clone https://github.com/ltdrdata/ComfyUI-Manager.git && \
# 	git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git

# RUN cd ./custom_nodes/ComfyUI-LTXVideo && \
# 	python3 -m pip install -r requirements.txt

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
	python3 -m pip install --upgrade pip && \
	python3 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
	rm /requirements.txt

COPY src/* /

# Optionally copy the snapshot file
# ADD *snapshot*.json /

# Restore the snapshot to install custom nodes
# RUN /restore_snapshot.sh

# Create necessary directories
RUN mkdir -p models/checkpoints models/vae models/diffusion_models models/clip

COPY ./mochi_models/mochi1PreviewVideo_fp8Scaled.safetensors /app/ComfyUI/models/diffusion_models/
COPY ./mochi_models/mochi1PreviewVideo_vae.safetensors /app/ComfyUI/models/vae/
COPY ./mochi_models/mochi1PreviewVideo_t5xxlFP8E4m3fnScaled.safetensors /app/ComfyUI/models/clip/

EXPOSE 8188

# Start the container
CMD ["python3","main.py","--listen","0.0.0.0","--front-end-version","Comfy-Org/ComfyUI_frontend@latest"]
