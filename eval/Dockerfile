# docker build --build-arg HF_TOKEN=$HF_TOKEN --build-arg DB_PASSWORD=$DB_PASSWORD --build-arg WANDB_API_KEY=$WANDB_API_KEY -t dcft-train:latest .
# docker run --shm-size=2g --gpus all -it dcft-train:latest
# torchrun --nnodes 1 --nproc_per_node 8 dcft/train/llamafactory/src/train.py dcft/train/configs/sample.yaml

# Base Image
FROM nvidia/cuda:12.4.1-devel-ubuntu20.04

ARG HF_TOKEN
ARG DB_PASSWORD
ARG WANDB_API_KEY

ENV PATH="/:${PATH}"
ENV PYTHONPATH="dcft/train/llamafactory/src:${PYTHONPATH}"
ENV HF_TOKEN=${HF_TOKEN}
ENV DB_PASSWORD=${DB_PASSWORD}
ENV WANDB_API_KEY=${WANDB_API_KEY}

# Update the package list and install prerequisites
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \ 
    python3-pip 

# Copy requirements first to prevent having to build again
COPY dcft/train/requirements.txt dcft/train/requirements.txt
RUN pip install -r dcft/train/requirements.txt

COPY dcft/train/ dcft/train/
COPY database/ database/
