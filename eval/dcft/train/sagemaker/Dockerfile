ARG AWS_REGION
FROM 763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker

ENV PATH="/opt/ml/code:${PATH}"
ENV PYTHONPATH="/opt/ml/code/llamafactory/src:${PYTHONPATH}"

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

COPY dcft/train/requirements.txt /opt/ml/code/requirements.txt

RUN pip install -r /opt/ml/code/requirements.txt
RUN pip install --upgrade s3fs
RUN pip install sagemaker-ssh-helper
RUN pip install --upgrade botocore
RUN pip uninstall accelerate -y
RUN pip install accelerate==0.34.2


# /opt/ml and all subdirectories are utilized by SageMaker, use the /code subdirectory to store your user code.
COPY dcft/train/ /opt/ml/code/
COPY database/ /opt/ml/code/database/

# Prevent sagemaker from installing requirements again.
RUN rm /opt/ml/code/requirements.txt

RUN cp /opt/ml/code/llamafactory/src/train.py /opt/ml/code/train.py